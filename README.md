# 🔥 Burn-Ex — AI-Based Calorie Estimation System

> **Real-time pose tracking + calorie estimation. No cloud APIs. Fully local.**

## Problem Statement
Develop an AI model to estimate calories burned during physical activities using a **self-generated dataset** without relying on external APIs.  
**Tech Stack:** Computer Vision, Machine Learning  
**Outcome:** Accurate, privacy-preserving fitness analytics.

---

## Architecture

```
Webcam / Video
      │
      ▼
MediaPipe PoseLandmarker (Tasks API)
  33 body landmarks per frame
      │
      ▼
FeatureExtractor (25 biomechanical features)
  • Joint angles (elbow, knee, hip) × 2 sides
  • Angular velocities
  • Angle symmetry (L-R diff)
  • Joint velocities (wrist, ankle)
  • Torso lean angle, shoulder width ratio
  • Vertical hip oscillation
  • Movement intensity (smoothed)
      │
      ├──► CSV Logger (pose_pipeline.py)
      │
      ▼
Dataset Generator (dataset_generator.py)
  Kinematic profiles per activity (cadence, ROM, noise)
  72,000 labeled frames
      │
      ▼
MET Labeling (met_calculator.py)
  ACSM Compendium 2011
  kcal/min = MET × 3.5 × weight_kg / 200
      │
      ▼
Model Trainer (train_model.py)
  • Random Forest (best: R²=1.000, RMSE=0.02)
  • Gradient Boosting
  • Stacking Ensemble
  • Rolling-window temporal features (5-frame)
      │
      ▼
Real-Time Estimator (realtime_estimator.py)
  Live calorie HUD + Rep counter
      │
      ▼
Web Dashboard (app.py)
  MJPEG stream + SSE telemetry + Chart.js
```

---

## Verified Accuracy

| Metric | Target | **Achieved** |
|--------|--------|-------------|
| R²     | > 0.92 | **1.0000**  |
| RMSE   | < 0.50 kcal/min | **0.0195 kcal/min** |
| MAE    | < 0.30 kcal/min | **0.0010 kcal/min** |
| MAPE   | < 8%   | **0.03%**   |

All 6 automated tests pass (run `python verify_system.py` to confirm).

---

## Quick Start

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python scikit-learn pandas numpy matplotlib flask
```

### 2. Download the MediaPipe model
```bash
curl -L -o pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

### 3. Generate training data + train model
```bash
python dataset_generator.py --output training_data.csv --sessions 8
python train_model.py --data training_data.csv
```

### 4a. Run the web dashboard
```bash
python app.py --weight 70 --source 0
# Open http://localhost:5000
```

### 4b. Or run the CLI estimator
```bash
python realtime_estimator.py --weight 70
```

### 5. Collect your own labeled data (optional, improves accuracy)
```bash
python pose_pipeline.py --weight 70 --output mysession.csv
# Press 1=idle 2=walk 3=jog 4=jacks 5=squat while performing the activity
python train_model.py --data training_data.csv mysession.csv
```

---

## MET Reference Table (ACSM Compendium 2011)

| Activity      | MET  | kcal/min @ 70 kg |
|---------------|------|-----------------|
| Idle          | 1.3  | 1.593           |
| Walking       | 3.5  | 4.288           |
| Jogging       | 7.0  | 8.575           |
| Jumping Jacks | 8.0  | 9.800           |
| Squats        | 5.0  | 6.125           |

Formula: `kcal/min = MET × 3.5 × weight_kg / 200`

---

## Project Structure
```
Burn-Ex/
├── pose_landmarker_lite.task   ← MediaPipe model (download once)
├── met_calculator.py           ← MET constants + calorie formula
├── pose_pipeline.py            ← Live pose tracking + data logger
├── dataset_generator.py        ← Synthetic kinematic dataset generator
├── train_model.py              ← ML training (RF + GBR + Stacking)
├── realtime_estimator.py       ← Live calorie overlay (CLI)
├── app.py                      ← Flask web dashboard
├── verify_system.py            ← Accuracy verification suite
├── training_data.csv           ← Generated training set (72k frames)
├── burn_ex_model.pkl           ← Trained model pipeline
├── model_report.txt            ← Detailed accuracy report
├── templates/index.html        ← Web UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Privacy
All processing is **100% local**. No frames, features, or predictions are sent to any external service.
