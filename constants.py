"""
Burn-Ex — Shared Constants
---------------------------
Defines FEATURE_FIELDS and ML_FEATURES in one place so that
dataset_generator.py, train_model.py, realtime_estimator.py, and
verify_system.py all agree on the exact feature schema WITHOUT any of
them needing to import pose_pipeline (which requires cv2 / mediapipe).

Import hierarchy (no circular deps):
    constants.py  ←  dataset_generator.py
    constants.py  ←  train_model.py
    constants.py  ←  realtime_estimator.py
    constants.py  ←  verify_system.py
    pose_pipeline.py  uses FEATURE_FIELDS from here (cv2 / mediapipe OK)
"""

# --------------------------------------------------------------------------
# Canonical ordered list — MUST match pose_pipeline.py FeatureExtractor
# --------------------------------------------------------------------------

# Metadata columns (not fed to the ML model)
META_FIELDS: list[str] = ['timestamp', 'frame_index', 'label', 'weight_kg']

# All 25 biomechanical features (order matters — used to build numpy arrays)
BIO_FEATURES: list[str] = [
    # Joint angles (degrees)
    'left_elbow_angle',       'right_elbow_angle',
    'left_knee_angle',        'right_knee_angle',
    'left_hip_angle',         'right_hip_angle',
    # Angle symmetry (|left − right|)
    'elbow_symmetry',         'knee_symmetry',          'hip_symmetry',
    # Body geometry
    'torso_length',           'torso_lean_angle',       'shoulder_width_ratio',
    # Hip dynamics
    'hip_center_y',           'vertical_hip_oscillation',
    # Movement intensity
    'movement_intensity',     'smoothed_intensity',
    # Joint velocities
    'left_wrist_velocity',    'right_wrist_velocity',
    'left_ankle_velocity',    'right_ankle_velocity',
    # Angular velocities (deg/s)
    'left_elbow_angular_vel', 'right_elbow_angular_vel',
    'left_knee_angular_vel',  'right_knee_angular_vel',
]

# Full CSV header (meta + bio)
FEATURE_FIELDS: list[str] = META_FIELDS + BIO_FEATURES

# Columns that the model uses as input features (weight_kg + all bio)
# NOTE: weight_kg MUST be first — it is the strongest predictor since
#       kcal/min = MET × 3.5 × weight_kg / 200
ML_FEATURES: list[str] = ['weight_kg'] + BIO_FEATURES

# Columns for which 5-frame rolling mean + std are computed during training
ROLLING_COLS: list[str] = [
    'movement_intensity',
    'smoothed_intensity',
    'left_knee_angular_vel',
    'right_knee_angular_vel',
    'vertical_hip_oscillation',
]

# Activity labels
ACTIVITY_NAMES: list[str] = [
    'idle', 'walking', 'jogging', 'jumping_jacks', 'squats'
]
