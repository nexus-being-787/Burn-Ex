#!/usr/bin/env python3
"""
Burn-Ex -- Pose Pipeline (Day 1)
----------------------------------
Live webcam (or video file) pose tracking using MediaPipe PoseLandmarker
(Tasks API). Extracts 25+ biomechanical features per frame and logs them
to a labeled CSV for model training.

Features extracted (accuracy-focused):
  Joint angles   : elbow, knee, hip (both sides)          6 features
  Velocities     : wrist, ankle (both sides)              4 features
  Angular vel.   : elbow, knee angle rate-of-change       4 features
  Torso          : torso length, torso lean angle         2 features
  Arm spread     : shoulder width ratio                   1 feature
  Hip center     : vertical position, oscillation         2 features
  Intensities    : movement intensity, smoothed           2 features
  Symmetry       : angle symmetry (L-R diff)              3 features
  Temporal       : cadence (hip oscillation freq)         1 feature

Total: 25 features

Usage:
  python pose_pipeline.py --weight 70 --output session1.csv
  python pose_pipeline.py --source video.mp4 --weight 70
  python pose_pipeline.py --source 1

Keys:
  1=idle  2=walking  3=jogging  4=jumping_jacks  5=squats  0=clear  q=quit
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Landmark indices
# --------------------------------------------------------------------------
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW,    RIGHT_ELBOW    = 13, 14
LEFT_WRIST,    RIGHT_WRIST    = 15, 16
LEFT_HIP,      RIGHT_HIP      = 23, 24
LEFT_KNEE,     RIGHT_KNEE     = 25, 26
LEFT_ANKLE,    RIGHT_ANKLE    = 27, 28

SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),   (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),     (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),         (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),       (RIGHT_KNEE, RIGHT_ANKLE),
]

ACTIVITY_LABELS = {
    ord('0'): 'unlabeled',
    ord('1'): 'idle',
    ord('2'): 'walking',
    ord('3'): 'jogging',
    ord('4'): 'jumping_jacks',
    ord('5'): 'squats',
}

# All 25 feature fields + metadata
FEATURE_FIELDS = [
    # Metadata
    'timestamp', 'frame_index', 'label', 'weight_kg',
    # Joint angles (degrees)
    'left_elbow_angle',   'right_elbow_angle',
    'left_knee_angle',    'right_knee_angle',
    'left_hip_angle',     'right_hip_angle',
    # Angle symmetry (absolute left-right difference)
    'elbow_symmetry', 'knee_symmetry', 'hip_symmetry',
    # Body scale
    'torso_length',
    # Torso lean (angle from vertical)
    'torso_lean_angle',
    # Arm spread
    'shoulder_width_ratio',
    # Hip center
    'hip_center_y',
    'vertical_hip_oscillation',
    # Movement
    'movement_intensity',
    'smoothed_intensity',
    # Joint velocities (torso-normalized, per second)
    'left_wrist_velocity',  'right_wrist_velocity',
    'left_ankle_velocity',  'right_ankle_velocity',
    # Angular velocities (degrees per second)
    'left_elbow_angular_vel',  'right_elbow_angular_vel',
    'left_knee_angular_vel',   'right_knee_angular_vel',
]


def calculate_angle(a, b, c) -> float:
    """Angle in degrees at vertex b, formed by rays b→a and b→c."""
    a, b, c = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_val = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def calculate_distance(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)))


def angle_between_vector_and_vertical(pt_top, pt_bottom) -> float:
    """Angle (degrees) between (pt_bottom→pt_top) and straight up (0, -1)."""
    v = np.array([pt_top[0] - pt_bottom[0], pt_top[1] - pt_bottom[1]], dtype=np.float64)
    up = np.array([0.0, -1.0])
    denom = np.linalg.norm(v) + 1e-8
    cos_val = np.clip(np.dot(v, up) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


class FeatureExtractor:
    """
    Extracts 25 scale-invariant biomechanical features from raw pose landmarks.

    All positional features are normalised by torso_length so that the same
    person closer or further from the camera produces consistent numbers.
    Temporal features use actual wall-clock dt for real-world physical units.
    """

    def __init__(self, smoothing_window: int = 15, hip_osc_window: int = 30):
        self.prev_pts:           dict | None = None
        self.prev_time:          float | None = None
        self.prev_angles:        dict | None = None
        self.intensity_history   = deque(maxlen=smoothing_window)
        self.hip_y_history       = deque(maxlen=hip_osc_window)

    @staticmethod
    def _xy(landmarks, idx) -> tuple:
        lm = landmarks[idx]
        return (float(lm.x), float(lm.y))

    def extract(self, landmarks, now: float) -> dict:
        idx_map = {
            'l_shoulder': LEFT_SHOULDER,  'r_shoulder': RIGHT_SHOULDER,
            'l_elbow':    LEFT_ELBOW,     'r_elbow':    RIGHT_ELBOW,
            'l_wrist':    LEFT_WRIST,     'r_wrist':    RIGHT_WRIST,
            'l_hip':      LEFT_HIP,       'r_hip':      RIGHT_HIP,
            'l_knee':     LEFT_KNEE,      'r_knee':     RIGHT_KNEE,
            'l_ankle':    LEFT_ANKLE,     'r_ankle':    RIGHT_ANKLE,
        }
        pts = {name: self._xy(landmarks, idx) for name, idx in idx_map.items()}

        # ---- Body scale reference ----------------------------------------
        mid_shoulder = ((pts['l_shoulder'][0] + pts['r_shoulder'][0]) / 2,
                        (pts['l_shoulder'][1] + pts['r_shoulder'][1]) / 2)
        mid_hip      = ((pts['l_hip'][0] + pts['r_hip'][0]) / 2,
                        (pts['l_hip'][1] + pts['r_hip'][1]) / 2)
        torso_length = calculate_distance(mid_shoulder, mid_hip) + 1e-6

        # ---- Joint angles ------------------------------------------------
        l_elbow_ang = calculate_angle(pts['l_shoulder'], pts['l_elbow'], pts['l_wrist'])
        r_elbow_ang = calculate_angle(pts['r_shoulder'], pts['r_elbow'], pts['r_wrist'])
        l_knee_ang  = calculate_angle(pts['l_hip'],      pts['l_knee'],  pts['l_ankle'])
        r_knee_ang  = calculate_angle(pts['r_hip'],      pts['r_knee'],  pts['r_ankle'])
        l_hip_ang   = calculate_angle(pts['l_shoulder'], pts['l_hip'],   pts['l_knee'])
        r_hip_ang   = calculate_angle(pts['r_shoulder'], pts['r_hip'],   pts['r_knee'])

        angles = {
            'l_elbow': l_elbow_ang, 'r_elbow': r_elbow_ang,
            'l_knee':  l_knee_ang,  'r_knee':  r_knee_ang,
        }

        # ---- Symmetry (L-R angular difference) --------------------------
        elbow_sym = abs(l_elbow_ang - r_elbow_ang)
        knee_sym  = abs(l_knee_ang  - r_knee_ang)
        hip_sym   = abs(l_hip_ang   - r_hip_ang)

        # ---- Torso lean ---------------------------------------------------
        torso_lean = angle_between_vector_and_vertical(mid_shoulder, mid_hip)

        # ---- Shoulder width ratio ----------------------------------------
        shoulder_width = calculate_distance(pts['l_shoulder'], pts['r_shoulder'])
        shoulder_width_ratio = shoulder_width / torso_length

        # ---- Hip center ---------------------------------------------------
        hip_center_y = mid_hip[1]  # normalised 0..1 in frame height
        self.hip_y_history.append(hip_center_y)
        vertical_hip_osc = float(np.std(self.hip_y_history)) if len(self.hip_y_history) > 1 else 0.0

        # ---- Time-dependent features ------------------------------------
        dt = 0.0
        movement_intensity    = 0.0
        l_wrist_vel = r_wrist_vel = 0.0
        l_ankle_vel = r_ankle_vel = 0.0
        l_elbow_ang_vel = r_elbow_ang_vel = 0.0
        l_knee_ang_vel  = r_knee_ang_vel  = 0.0

        if self.prev_pts is not None and self.prev_time is not None:
            dt = max(now - self.prev_time, 1e-3)

            # Movement intensity: mean normalised joint displacement per second
            total_disp = sum(
                calculate_distance(pts[k], self.prev_pts[k]) / torso_length
                for k in pts
            )
            movement_intensity = (total_disp / len(pts)) / dt

            # Joint velocities
            l_wrist_vel = calculate_distance(pts['l_wrist'], self.prev_pts['l_wrist']) / torso_length / dt
            r_wrist_vel = calculate_distance(pts['r_wrist'], self.prev_pts['r_wrist']) / torso_length / dt
            l_ankle_vel = calculate_distance(pts['l_ankle'], self.prev_pts['l_ankle']) / torso_length / dt
            r_ankle_vel = calculate_distance(pts['r_ankle'], self.prev_pts['r_ankle']) / torso_length / dt

            # Angular velocities (deg/s)
            if self.prev_angles is not None:
                l_elbow_ang_vel = abs(l_elbow_ang - self.prev_angles['l_elbow']) / dt
                r_elbow_ang_vel = abs(r_elbow_ang - self.prev_angles['r_elbow']) / dt
                l_knee_ang_vel  = abs(l_knee_ang  - self.prev_angles['l_knee'])  / dt
                r_knee_ang_vel  = abs(r_knee_ang  - self.prev_angles['r_knee'])  / dt

        self.intensity_history.append(movement_intensity)
        smoothed_intensity = float(np.mean(self.intensity_history)) if self.intensity_history else 0.0

        self.prev_pts   = pts
        self.prev_time  = now
        self.prev_angles = angles

        return {
            'left_elbow_angle':       l_elbow_ang,
            'right_elbow_angle':      r_elbow_ang,
            'left_knee_angle':        l_knee_ang,
            'right_knee_angle':       r_knee_ang,
            'left_hip_angle':         l_hip_ang,
            'right_hip_angle':        r_hip_ang,
            'elbow_symmetry':         elbow_sym,
            'knee_symmetry':          knee_sym,
            'hip_symmetry':           hip_sym,
            'torso_length':           torso_length,
            'torso_lean_angle':       torso_lean,
            'shoulder_width_ratio':   shoulder_width_ratio,
            'hip_center_y':           hip_center_y,
            'vertical_hip_oscillation': vertical_hip_osc,
            'movement_intensity':     movement_intensity,
            'smoothed_intensity':     smoothed_intensity,
            'left_wrist_velocity':    l_wrist_vel,
            'right_wrist_velocity':   r_wrist_vel,
            'left_ankle_velocity':    l_ankle_vel,
            'right_ankle_velocity':   r_ankle_vel,
            'left_elbow_angular_vel': l_elbow_ang_vel,
            'right_elbow_angular_vel':r_elbow_ang_vel,
            'left_knee_angular_vel':  l_knee_ang_vel,
            'right_knee_angular_vel': r_knee_ang_vel,
        }


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    used = set(i for pair in SKELETON_CONNECTIONS for i in pair)
    px = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in used}
    for a, b in SKELETON_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 220, 100), 2)
    for (x, y) in px.values():
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def draw_overlay(frame, features, label, fps):
    h = frame.shape[0]
    lines = [f"Label: {label}", f"FPS: {fps:.1f}"]
    if features:
        lines += [
            f"Intensity: {features['smoothed_intensity']:.3f}",
            f"Knee L/R: {features['left_knee_angle']:.0f}/{features['right_knee_angle']:.0f} deg",
            f"Elbow L/R: {features['left_elbow_angle']:.0f}/{features['right_elbow_angle']:.0f} deg",
            f"Hip Osc: {features['vertical_hip_oscillation']:.4f}",
            f"Wrist vel L/R: {features['left_wrist_velocity']:.2f}/{features['right_wrist_velocity']:.2f}",
        ]
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1, cv2.LINE_AA)
        y += 22
    cv2.putText(frame,
                "1=idle 2=walk 3=jog 4=jacks 5=squat 0=clear  q=quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def build_landmarker(model_path: str):
    if not os.path.isfile(model_path):
        sys.exit(
            f"\n[Burn-Ex] Cannot find '{model_path}'.\n"
            f"Download it with:\n"
            f"  curl -L -o {model_path} https://storage.googleapis.com/mediapipe-models/"
            f"pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        )
    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(description="Burn-Ex pose tracking + data logger")
    parser.add_argument('--source', default='0',
                        help="Camera index (default 0) or path to video file")
    parser.add_argument('--weight', type=float, default=70.0,
                        help="Body weight in kg (recorded in CSV for calorie labeling)")
    parser.add_argument('--output', default='session1.csv',
                        help="CSV file to append logged features to")
    parser.add_argument('--model', default='pose_landmarker_lite.task',
                        help="Path to PoseLandmarker .task model file")
    args = parser.parse_args()

    try:
        source = int(args.source)
        is_camera = True
    except ValueError:
        source = args.source
        is_camera = False

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"[Burn-Ex] Could not open video source: {source}")

    landmarker = build_landmarker(args.model)
    extractor   = FeatureExtractor()
    label       = 'unlabeled'

    file_exists = os.path.isfile(args.output)
    csv_file    = open(args.output, 'a', newline='')
    writer      = csv.DictWriter(csv_file, fieldnames=FEATURE_FIELDS)
    if not file_exists:
        writer.writeheader()

    frame_index     = 0
    fps             = 0.0
    fps_count       = 0
    last_fps_time   = time.time()
    last_ts_ms      = -1

    print(f"[Burn-Ex] Logging → {args.output} | Weight: {args.weight} kg | 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_camera:
                frame = cv2.flip(frame, 1)

            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now        = time.time()
            ts_ms      = max(int(now * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms

            result   = landmarker.detect_for_video(mp_image, ts_ms)
            features = None

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                draw_skeleton(frame, lm)
                features = extractor.extract(lm, now)
                writer.writerow({
                    'timestamp':  now,
                    'frame_index': frame_index,
                    'label':      label,
                    'weight_kg':  args.weight,
                    **features,
                })

            fps_count += 1
            if now - last_fps_time >= 1.0:
                fps           = fps_count / (now - last_fps_time)
                fps_count     = 0
                last_fps_time = now

            draw_overlay(frame, features, label, fps)
            cv2.imshow('Burn-Ex — Pose Pipeline', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in ACTIVITY_LABELS:
                label = ACTIVITY_LABELS[key]
                print(f"[Burn-Ex] Label → {label}")

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        landmarker.close()
        print(f"[Burn-Ex] Saved {frame_index} frames → {args.output}")


if __name__ == '__main__':
    main()
