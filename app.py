#!/usr/bin/env python3
"""
Burn-Ex -- Flask Web Application
----------------------------------
Provides a modern dark-mode web dashboard for real-time calorie monitoring.
Now uses SaaS-style architecture: client browser captures webcam, 
and backend processes frames on-demand via /api/process_frame.
"""

from __future__ import annotations

import argparse
import os
import time
import json
import threading
import base64

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, stream_with_context, request

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

def get_state():
    with state_lock:
        return dict(state)

def set_state(**kwargs):
    with state_lock:
        state.update(kwargs)

# --------------------------------------------------------------------------
# Global ML Models & Buffers
# --------------------------------------------------------------------------
models_loaded = False
landmarker = None
extractor = None
reg_pipeline = None
reg_cols = None
clf_pipeline = None
clf_cols = None
clf_classes = None

reg_buf = None
clf_buf = None
rep_ctr = None
ema = None

last_ts_ms = -1
prev_t = 0
start_t = 0
fps_count = 0
last_fps_t = 0
cal_history = []

def init_models(model_dir: str, mp_model: str, weight_kg: float, ema_alpha: float = 0.08, no_auto: bool = False):
    global models_loaded, landmarker, extractor, reg_pipeline, reg_cols
    global clf_pipeline, clf_cols, clf_classes, reg_buf, clf_buf, rep_ctr, ema
    global prev_t, start_t, last_fps_t
    
    print("[App] Initializing ML Models...")
    reg_pipeline, reg_cols = load_regression_model(model_dir)
    if not no_auto:
        clf_pipeline, clf_cols, clf_classes = load_classifier(model_dir)
        
    landmarker = build_landmarker(mp_model)
    extractor = FeatureExtractor()
    reg_buf = RollingBuffer()
    clf_buf = RollingBuffer()
    rep_ctr = RepCounter()
    ema = EMA(alpha=ema_alpha)
    
    now = time.time()
    prev_t = now
    start_t = now
    last_fps_t = now
    
    set_state(session_active=True, weight_kg=weight_kg)
    models_loaded = True
    print("[App] ML Models loaded successfully.")

# --------------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    global last_ts_ms, prev_t, start_t, fps_count, last_fps_t, cal_history
    
    if not models_loaded:
        return jsonify({"error": "Models not loaded"}), 503
        
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
        
    # Decode base64 image
    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    now = time.time()
    ts_ms = max(int(now * 1000), last_ts_ms + 1)
    last_ts_ms = ts_ms
    dt = max(now - prev_t, 1e-3)
    prev_t = now
    
    result = landmarker.detect_for_video(mp_img, ts_ms)
    features = None
    pred_kcal_raw = 0.0
    
    with state_lock:
        label = state['label']
        weight_kg = state['weight_kg']
        auto_label = state['auto_label']
        confidence = state['confidence']
        total_kcal = state['total_kcal']
        
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
            x_reg = reg_buf.build_vector(features, reg_cols, weight_kg)
            pred_kcal_raw = float(max(reg_pipeline.predict(x_reg)[0], 0.0))
            
        # Auto-classify activity
        if clf_pipeline is not None:
            x_clf = clf_buf.build_vector(features, clf_cols, weight_kg)
            proba = clf_pipeline.predict_proba(x_clf)[0]
            best_idx = int(np.argmax(proba))
            auto_label = clf_classes[best_idx]
            confidence = float(proba[best_idx])
            
    # EMA smoothed prediction
    pred_kcal = ema.update(pred_kcal_raw)
    total_kcal += pred_kcal * dt / 60.0
    
    if len(cal_history) == 0 or now - cal_history[-1][0] >= 1.0:
        cal_history.append((now - start_t, pred_kcal))
        if len(cal_history) > 300:
            cal_history.pop(0)
            
    # FPS
    fps_count += 1
    fps = state.get('fps', 0.0)
    if now - last_fps_t >= 1.0:
        fps = round(fps_count / (now - last_fps_t), 1)
        fps_count = 0
        last_fps_t = now
        
    # Effective label for GT
    effective_label = label if label != 'unlabeled' else auto_label
    gt_kcal = label_to_kcal_per_min(effective_label, weight_kg)
    
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
    
    # Encode result frame back to base64 with high quality for clear overlay
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    res_image = ""
    if ret:
        res_image = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
        
    return jsonify({
        "image": res_image,
        "state": get_state()
    })

@app.route('/sse')
def sse():
    def event_stream():
        while True:
            st = get_state()
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
    parser.add_argument('--weight',    type=float, default=70.0)
    parser.add_argument('--model-dir', default='.')
    parser.add_argument('--mp-model',  default='pose_landmarker_lite.task')
    parser.add_argument('--port',      type=int, default=5000)
    parser.add_argument('--no-browser', action='store_true')
    # Backward compatibility args
    parser.add_argument('--source', default='0', help='Ignored in SaaS mode')
    args = parser.parse_args()

    init_models(args.model_dir, args.mp_model, args.weight)

    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f'http://localhost:{args.port}')).start()

    print(f"[App] Starting Flask server on port {args.port}...")
    # Increase max content length for base64 images (e.g. 5MB)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 
    app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)

if __name__ == '__main__':
    main()
