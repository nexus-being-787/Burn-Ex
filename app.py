#!/usr/bin/env python3
"""
Burn-Ex -- Flask Web Application
----------------------------------
Provides a modern dark-mode web dashboard for real-time calorie monitoring.
Streams MJPEG video with pose skeleton overlay via /video_feed endpoint,
and pushes JSON telemetry every second via Server-Sent Events /sse.

Usage:
  python app.py --weight 70 --source 0
  python app.py --weight 70 --source demo.mp4 --no-browser

Then open: http://localhost:5000
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
import json
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, stream_with_context

# Import local modules
from met_calculator import ACTIVITY_METS, label_to_kcal_per_min
from pose_pipeline import FeatureExtractor, draw_skeleton, build_landmarker, ACTIVITY_LABELS
from constants import ML_FEATURES, ROLLING_COLS, ACTIVITY_NAMES
from realtime_estimator import EMA, RollingBuffer, RepCounter, load_regression_model, load_classifier

app = Flask(__name__)

# --------------------------------------------------------------------------
# Global state (thread-safe via lock)
# --------------------------------------------------------------------------

state_lock = threading.Lock()
state = {
    'label':           'unlabeled',
    'auto_label':      'unlabeled',
    'confidence':      0.0,
    'pred_kcal':       0.0,
    'gt_kcal':         0.0,
    'total_kcal':      0.0,
    'reps':            0,
    'fps':             0.0,
    'intensity':       0.0,
    'left_knee':       0.0,
    'right_knee':      0.0,
    'session_active':  False,
    'weight_kg':       70.0,
    'calorie_history': [],  # (timestamp_s, kcal_per_min) pairs
}
latest_frame: bytes | None = None
frame_lock = threading.Lock()


def get_state():
    with state_lock:
        return dict(state)


def set_state(**kwargs):
    with state_lock:
        state.update(kwargs)


# --------------------------------------------------------------------------
# Video capture + inference thread
# --------------------------------------------------------------------------

class VideoPipeline(threading.Thread):
    def __init__(self, source, weight_kg: float, model_dir: str, mp_model: str, ema_alpha: float = 0.12, no_auto: bool = False):
        super().__init__(daemon=True)
        self.source    = source
        self.weight_kg = weight_kg
        self.model_dir = model_dir
        self.mp_model  = mp_model
        self.ema_alpha = ema_alpha
        self.no_auto   = no_auto
        self.running   = True

    def run(self):
        global latest_frame

        reg_pipeline, reg_cols = load_regression_model(self.model_dir)
        clf_pipeline = clf_cols = clf_classes = None
        if not self.no_auto:
            clf_pipeline, clf_cols, clf_classes = load_classifier(self.model_dir)

        try:
            source = int(self.source)
            is_cam = True
        except (ValueError, TypeError):
            source = self.source
            is_cam = False

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[App] Cannot open source: {source}")
            return

        landmarker = build_landmarker(self.mp_model)
        extractor  = FeatureExtractor()
        reg_buf    = RollingBuffer()
        clf_buf    = RollingBuffer()
        rep_ctr    = RepCounter()
        ema        = EMA(alpha=self.ema_alpha)

        label       = 'unlabeled'
        auto_label  = 'unlabeled'
        confidence  = 0.0
        total_kcal  = 0.0
        fps         = 0.0
        fps_count   = 0
        last_fps_t  = time.time()
        last_ts_ms  = -1
        prev_t      = time.time()
        start_t     = time.time()
        cal_history: list[tuple] = []

        set_state(session_active=True, weight_kg=self.weight_kg)

        try:
            while self.running:
                ok_frame, frame = cap.read()
                if not ok_frame:
                    if is_cam:
                        continue
                    break
                if is_cam:
                    frame = cv2.flip(frame, 1)

                rgb    = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                now     = time.time()
                ts_ms   = max(int(now * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                dt      = max(now - prev_t, 1e-3)
                prev_t  = now

                result   = landmarker.detect_for_video(mp_img, ts_ms)
                features = None
                pred_kcal_raw = 0.0

                with state_lock:
                    label = state['label']
                    if 'reset_flag' in state and state['reset_flag']:
                        total_kcal = 0.0
                        rep_ctr.count = 0
                        cal_history.clear()
                        start_t = time.time()
                        state['reset_flag'] = False
                        state['reps'] = 0

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    draw_skeleton(frame, lm)
                    features = extractor.extract(lm, now)

                    # Rep counter
                    avg_knee = (features['left_knee_angle'] + features['right_knee_angle']) / 2.0
                    rep_ctr.update(avg_knee)

                    # ML inference
                    if reg_pipeline is not None and reg_cols is not None:
                        x_reg     = reg_buf.build_vector(features, reg_cols, self.weight_kg)
                        pred_kcal_raw = float(max(reg_pipeline.predict(x_reg)[0], 0.0))

                    # Auto-classify activity
                    if clf_pipeline is not None:
                        x_clf       = clf_buf.build_vector(features, clf_cols, self.weight_kg)
                        proba       = clf_pipeline.predict_proba(x_clf)[0]
                        best_idx    = int(np.argmax(proba))
                        auto_label  = clf_classes[best_idx]
                        confidence  = float(proba[best_idx])

                # EMA smoothed prediction
                pred_kcal = ema.update(pred_kcal_raw)
                total_kcal += pred_kcal * dt / 60.0

                if len(cal_history) == 0 or now - cal_history[-1][0] >= 1.0:
                    cal_history.append((now - start_t, pred_kcal))
                    if len(cal_history) > 300:
                        cal_history.pop(0)

                # FPS
                fps_count += 1
                if now - last_fps_t >= 1.0:
                    fps        = fps_count / (now - last_fps_t)
                    fps_count  = 0
                    last_fps_t = now

                # Effective label for GT
                effective_label = label if label != 'unlabeled' else auto_label
                gt_kcal = label_to_kcal_per_min(effective_label, self.weight_kg)

                # Update state
                set_state(
                    pred_kcal=pred_kcal,
                    gt_kcal=gt_kcal,
                    total_kcal=total_kcal,
                    reps=rep_ctr.count,
                    fps=round(fps, 1),
                    intensity=features['smoothed_intensity'] if features else 0.0,
                    left_knee=features['left_knee_angle'] if features else 0.0,
                    right_knee=features['right_knee_angle'] if features else 0.0,
                    auto_label=auto_label,
                    confidence=confidence,
                    calorie_history=list(cal_history)
                )

                # JPEG encode
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    with frame_lock:
                        latest_frame = buf.tobytes()

        finally:
            cap.release()
            landmarker.close()
            set_state(session_active=False)
            print("[App] Video pipeline stopped.")


# --------------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def gen_frames():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sse')
def sse():
    def event_stream():
        last_total = -1
        last_reps  = -1
        last_pred  = -1
        while True:
            st = get_state()
            # To save bandwidth, we send cal_history only, and everything else
            yield f"data: {json.dumps(st)}\n\n"
            time.sleep(1.0)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route('/api/state')
def api_state():
    return jsonify(get_state())


@app.route('/api/label/<new_label>', methods=['POST'])
def api_set_label(new_label):
    if new_label not in ACTIVITY_NAMES and new_label != 'unlabeled':
        return jsonify(error="Invalid label"), 400
    set_state(label=new_label)
    return jsonify(ok=True, label=new_label)


@app.route('/api/reset', methods=['POST'])
def api_reset():
    set_state(reset_flag=True)
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Burn-Ex Web Dashboard")
    parser.add_argument('--source',    default='0')
    parser.add_argument('--weight',    type=float, default=70.0)
    parser.add_argument('--model-dir', default='.')
    parser.add_argument('--mp-model',  default='pose_landmarker_lite.task')
    parser.add_argument('--port',      type=int, default=5000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    print(f"[App] Starting video pipeline (source={args.source}, weight={args.weight}kg)")
    pipeline = VideoPipeline(args.source, args.weight, args.model_dir, args.mp_model)
    pipeline.start()

    if not args.no_browser:
        import webbrowser
        # Give Flask a second to start
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{args.port}')).start()

    print(f"[App] Starting Flask server on port {args.port}...")
    app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
