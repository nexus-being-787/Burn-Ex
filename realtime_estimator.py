#!/usr/bin/env python3
"""
Burn-Ex -- Real-Time Calorie Estimator (Improved)
---------------------------------------------------
Improvements over v1:
  • Auto activity detection  — no key presses needed; a second RF classifier
    reads your pose in real time and labels the activity automatically.
  • EMA smoothing            — exponential moving average on the calorie
    prediction makes the readout stable and jitter-free.
  • BMR-adjusted mode        — adds resting metabolic rate correction so the
    estimate reflects NET additional calories burned above baseline.
  • Better HUD               — session timer, auto-detected label overlay,
    confidence bar for the classifier.

Usage:
  python realtime_estimator.py --weight 70
  python realtime_estimator.py --source video.mp4 --weight 70 --height 175
  python realtime_estimator.py --weight 70 --no-auto   # disable auto-detection
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from met_calculator import ACTIVITY_METS, kcal_per_min, label_to_kcal_per_min
from constants import ML_FEATURES, ROLLING_COLS, ACTIVITY_NAMES
from pose_pipeline import (ACTIVITY_LABELS, FeatureExtractor,
                           build_landmarker, draw_skeleton)


# --------------------------------------------------------------------------
# EMA smoother
# --------------------------------------------------------------------------

class EMA:
    """Exponential moving average — smooths noisy real-time signal."""
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# --------------------------------------------------------------------------
# Rolling feature buffer (temporal context for model)
# --------------------------------------------------------------------------

class RollingBuffer:
    def __init__(self, window: int = 5):
        self.history: dict[str, deque] = {c: deque(maxlen=window) for c in ROLLING_COLS}

    def update(self, features: dict):
        for c in ROLLING_COLS:
            self.history[c].append(features.get(c, 0.0))

    def rolling_features(self) -> dict:
        result = {}
        for c, buf in self.history.items():
            arr = np.array(buf)
            result[f'{c}_roll5_mean'] = float(np.mean(arr)) if len(arr) else 0.0
            result[f'{c}_roll5_std']  = float(np.std(arr))  if len(arr) > 1 else 0.0
        return result

    def build_vector(self, features: dict, feature_cols: list, weight_kg: float) -> np.ndarray:
        self.update(features)
        merged = {**features, **self.rolling_features(), 'weight_kg': weight_kg}
        return np.array([[merged.get(c, 0.0) for c in feature_cols]])


# --------------------------------------------------------------------------
# Rep counter
# --------------------------------------------------------------------------

class RepCounter:
    def __init__(self, low: float = 110.0, high: float = 160.0):
        self.low, self.high = low, high
        self.state = 'up'
        self.count = 0

    def update(self, avg_knee: float) -> int:
        if self.state == 'up' and avg_knee < self.low:
            self.state = 'down'
        elif self.state == 'down' and avg_knee > self.high:
            self.state = 'up'
            self.count += 1
        return self.count


# --------------------------------------------------------------------------
# Model loaders
# --------------------------------------------------------------------------

def load_regression_model(model_dir: str):
    path = os.path.join(model_dir, 'burn_ex_model.pkl')
    if not os.path.isfile(path):
        sys.exit(f"[Burn-Ex] Model not found: {path}\n  Run: python train_model.py --data training_data.csv")
    with open(path, 'rb') as f:
        a = pickle.load(f)
    print(f"[Burn-Ex] Regression model: {a.get('best_model', '?')}  ({len(a['feature_cols'])} features)")
    return a['pipeline'], a['feature_cols']


def load_classifier(model_dir: str):
    path = os.path.join(model_dir, 'activity_classifier.pkl')
    if not os.path.isfile(path):
        print(f"[Burn-Ex] No activity_classifier.pkl found — auto-detection disabled.")
        print(f"          Run: python train_model.py --data training_data.csv --no-tune")
        return None, None, None
    with open(path, 'rb') as f:
        a = pickle.load(f)
    print(f"[Burn-Ex] Activity classifier loaded  classes={a['classes']}")
    return a['pipeline'], a['feature_cols'], a['classes']


# --------------------------------------------------------------------------
# HUD drawing
# --------------------------------------------------------------------------

LABEL_COLORS = {
    'idle':          (140, 140, 140),
    'walking':       (0, 200, 255),
    'jogging':       (0, 220, 100),
    'jumping_jacks': (0, 165, 255),
    'squats':        (180, 100, 255),
    'unlabeled':     (80, 80, 80),
}


def draw_hud(frame, pred_kcal: float, gt_kcal: float, total_kcal: float,
             reps: int, label: str, auto_label: str, confidence: float,
             fps: float, elapsed: str, features: dict | None):
    h, w = frame.shape[:2]

    # Semi-transparent left panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (270, h), (10, 12, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    col_green  = (0, 220, 100)
    col_blue   = (0, 180, 255)
    col_orange = (0, 165, 255)
    col_red    = (60, 80, 255)
    col_muted  = (130, 140, 160)
    col_white  = (230, 235, 245)

    def put(text, y, color=col_white, scale=0.54, thick=1):
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, scale,
                    color, thick, cv2.LINE_AA)

    # Header
    put("BURN-EX  AI", 28, col_green, scale=0.68)
    cv2.line(frame, (10, 36), (260, 36), (50, 60, 80), 1)

    # Session timer
    put(f"Session: {elapsed}", 55, col_muted, scale=0.44)

    # Auto-detected activity
    auto_color = LABEL_COLORS.get(auto_label, (120, 120, 120))
    put("Auto Detected:", 78, col_muted, scale=0.42)
    put(auto_label.upper().replace('_', ' '), 98, auto_color, scale=0.62)

    # Confidence bar
    bar_w = int(240 * confidence)
    cv2.rectangle(frame, (10, 103), (250, 108), (40, 45, 55), -1)
    cv2.rectangle(frame, (10, 103), (10 + bar_w, 108), auto_color, -1)
    put(f"{confidence*100:.0f}% conf", 121, col_muted, scale=0.40)

    # Manual label (if set)
    if label != 'unlabeled':
        put(f"Manual: {label}", 138, (200, 200, 100), scale=0.42)

    cv2.line(frame, (10, 145), (260, 145), (40, 50, 65), 1)

    # Calorie readings
    put("AI Burn Rate:", 165, col_muted, scale=0.42)
    put(f"{pred_kcal:.3f} kcal/min", 185, col_blue, scale=0.65)

    put("MET Reference:", 210, col_muted, scale=0.42)
    put(f"{gt_kcal:.3f} kcal/min", 228, col_orange, scale=0.58)

    cv2.line(frame, (10, 238), (260, 238), (40, 50, 65), 1)

    put("SESSION TOTAL", 258, col_muted, scale=0.40)
    put(f"{total_kcal:.2f} kcal", 283, col_red, scale=0.82)

    put(f"Reps: {reps}", 308, col_muted, scale=0.48)
    put(f"FPS: {fps:.1f}", 325, col_muted, scale=0.44)

    if features:
        put(f"Intensity: {features.get('smoothed_intensity', 0):.3f}", 342,
            col_muted, scale=0.42)

    # Bottom help bar
    cv2.putText(frame,
                "1=idle 2=walk 3=jog 4=jacks 5=squat 0=auto  q=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (160, 170, 190), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Burn-Ex real-time estimator (improved)")
    parser.add_argument('--source',    default='0')
    parser.add_argument('--weight',    type=float, default=70.0)
    parser.add_argument('--height',    type=float, default=175.0,
                        help="Height in cm (used for BMR-adjusted display)")
    parser.add_argument('--age',       type=int,   default=25)
    parser.add_argument('--sex',       default='male', choices=['male', 'female'])
    parser.add_argument('--model-dir', default='.')
    parser.add_argument('--mp-model',  default='pose_landmarker_lite.task')
    parser.add_argument('--no-auto',   action='store_true',
                        help="Disable auto activity detection")
    parser.add_argument('--ema',       type=float, default=0.12,
                        help="EMA alpha for prediction smoothing (default 0.12)")
    args = parser.parse_args()

    # BMR (Harris-Benedict) for resting calorie display
    if args.sex == 'male':
        bmr = 88.36 + 13.40 * args.weight + 5.0 * args.height - 5.68 * args.age
    else:
        bmr = 447.59 + 9.25 * args.weight + 3.10 * args.height - 4.33 * args.age
    bmr_per_min = bmr / (24 * 60)
    print(f"[Burn-Ex] BMR: {bmr:.0f} kcal/day  ({bmr_per_min:.3f} kcal/min resting)")

    # Load models
    reg_pipeline, reg_cols = load_regression_model(args.model_dir)
    clf_pipeline = clf_cols = clf_classes = None
    if not args.no_auto:
        clf_pipeline, clf_cols, clf_classes = load_classifier(args.model_dir)

    # Video source
    try:
        source = int(args.source)
        is_cam = True
    except ValueError:
        source = args.source
        is_cam = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[Burn-Ex] Cannot open: {source}")

    landmarker = build_landmarker(args.mp_model)
    extractor  = FeatureExtractor()
    reg_buf    = RollingBuffer()
    clf_buf    = RollingBuffer()
    rep_ctr    = RepCounter()
    ema        = EMA(alpha=args.ema)

    label      = 'unlabeled'    # manual label (keys 1-5)
    auto_label = 'unlabeled'    # classifier output
    confidence = 0.0
    total_kcal = 0.0
    fps = fps_count = 0.0
    last_fps_t = last_ts_ms = start_t = prev_t = time.time()

    print("[Burn-Ex] Running — press 1-5 to override, 0 for auto, q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_cam:
                frame = cv2.flip(frame, 1)

            rgb    = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now      = time.time()
            ts_ms    = max(int(now * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            dt       = max(now - prev_t, 1e-3)
            prev_t   = now

            result   = landmarker.detect_for_video(mp_img, ts_ms)
            features = None
            pred_kcal_raw = 0.0

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_skeleton(frame, lm)
                features = extractor.extract(lm, now)

                # Rep counter
                avg_knee = (features['left_knee_angle'] + features['right_knee_angle']) / 2.0
                rep_ctr.update(avg_knee)

                # Regression inference
                x_reg     = reg_buf.build_vector(features, reg_cols, args.weight)
                pred_kcal_raw = float(max(reg_pipeline.predict(x_reg)[0], 0.0))

                # Auto-classify activity
                if clf_pipeline is not None:
                    x_clf       = clf_buf.build_vector(features, clf_cols, args.weight)
                    proba       = clf_pipeline.predict_proba(x_clf)[0]
                    best_idx    = int(np.argmax(proba))
                    auto_label  = clf_classes[best_idx]
                    confidence  = float(proba[best_idx])

            # EMA smoothed prediction
            pred_kcal = ema.update(pred_kcal_raw)
            total_kcal += pred_kcal * dt / 60.0

            # Effective label for GT display
            effective_label = label if label != 'unlabeled' else auto_label
            gt_kcal = label_to_kcal_per_min(effective_label, args.weight)

            # FPS
            fps_count += 1
            if now - last_fps_t >= 1.0:
                fps        = fps_count / (now - last_fps_t)
                fps_count  = 0
                last_fps_t = now

            # Session timer string
            elapsed_s = int(now - start_t)
            elapsed   = f"{elapsed_s // 60:02d}:{elapsed_s % 60:02d}"

            draw_hud(frame, pred_kcal, gt_kcal, total_kcal,
                     rep_ctr.count, label, auto_label, confidence,
                     fps, elapsed, features)
            cv2.imshow('Burn-Ex — Real-Time Estimator', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('0'):
                label = 'unlabeled'
                print("[Burn-Ex] Manual label cleared — using auto-detection")
            elif key in ACTIVITY_LABELS:
                label = ACTIVITY_LABELS[key]
                print(f"[Burn-Ex] Manual label → {label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        total_s = int(time.time() - start_t)
        print(f"\n[Burn-Ex] Session ended ({total_s // 60}m {total_s % 60}s)")
        print(f"          Total burned:  {total_kcal:.2f} kcal")
        print(f"          Total reps:    {rep_ctr.count}")
        print(f"          BMR (resting): {bmr_per_min * total_s / 60:.2f} kcal")
        print(f"          Net active:    {max(0, total_kcal - bmr_per_min * total_s / 60):.2f} kcal above BMR")


if __name__ == '__main__':
    main()
