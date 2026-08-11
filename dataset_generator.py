#!/usr/bin/env python3
"""
Burn-Ex -- Synthetic Dataset Generator
----------------------------------------
Generates a realistic labeled CSV of biomechanical features WITHOUT
requiring a live webcam session. Uses kinematic models of each activity
to produce waveforms that match real human motion patterns.

Each activity generates 1800 frames (~60 seconds @ 30 fps) with
per-class Gaussian noise so the model sees variability.

Accuracy design choices:
  - Activity-specific kinematic profiles (cadence, ROM, amplitude)
  - Torso oscillation derived from reported gait-analysis literature
  - Per-frame realistic noise (σ tuned per activity intensity)
  - Random body weight drawn from 50-100 kg range per session for
    person-invariant training signal

Usage:
  python dataset_generator.py --output training_data.csv --sessions 5 --weight 70
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time

import numpy as np

from met_calculator import ACTIVITY_METS, kcal_per_min
from constants import FEATURE_FIELDS, ML_FEATURES


# --------------------------------------------------------------------------
# Activity kinematic profiles
# Each profile defines the "ideal" motion waveforms for the activity.
# --------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    "idle": {
        "cadence_hz":          0.2,    # slow breathing movement
        "knee_angle_mean":     170.0,  # nearly straight knees
        "knee_angle_amp":      3.0,    # tiny oscillation
        "elbow_angle_mean":    160.0,  # arms almost straight
        "elbow_angle_amp":     5.0,
        "hip_angle_mean":      170.0,
        "hip_angle_amp":       3.0,
        "movement_intensity":  0.01,
        "intensity_noise":     0.005,
        "hip_osc":             0.003,
        "wrist_vel":           0.01,
        "ankle_vel":           0.005,
        "torso_lean_mean":     8.0,
        "torso_lean_std":      2.0,
        "shoulder_w_ratio":    0.42,
    },
    "walking": {
        "cadence_hz":          1.8,    # typical walking cadence ~1.8 steps/s
        "knee_angle_mean":     155.0,
        "knee_angle_amp":      30.0,   # knee flexes during step
        "elbow_angle_mean":    130.0,
        "elbow_angle_amp":     25.0,
        "hip_angle_mean":      155.0,
        "hip_angle_amp":       15.0,
        "movement_intensity":  0.18,
        "intensity_noise":     0.04,
        "hip_osc":             0.015,
        "wrist_vel":           0.30,
        "ankle_vel":           0.45,
        "torso_lean_mean":     12.0,
        "torso_lean_std":      3.0,
        "shoulder_w_ratio":    0.43,
    },
    "jogging": {
        "cadence_hz":          2.7,    # typical jogging cadence ~2.7 steps/s
        "knee_angle_mean":     130.0,
        "knee_angle_amp":      50.0,   # deeper knee flexion
        "elbow_angle_mean":    90.0,   # arms pumping more
        "elbow_angle_amp":     35.0,
        "hip_angle_mean":      135.0,
        "hip_angle_amp":       30.0,
        "movement_intensity":  0.55,
        "intensity_noise":     0.08,
        "hip_osc":             0.030,
        "wrist_vel":           0.90,
        "ankle_vel":           1.20,
        "torso_lean_mean":     18.0,
        "torso_lean_std":      4.0,
        "shoulder_w_ratio":    0.40,
    },
    "jumping_jacks": {
        "cadence_hz":          2.0,    # ~2 full jacks per second
        "knee_angle_mean":     145.0,
        "knee_angle_amp":      40.0,
        "elbow_angle_mean":    100.0,
        "elbow_angle_amp":     60.0,   # arms go from sides to overhead
        "hip_angle_mean":      145.0,
        "hip_angle_amp":       35.0,
        "movement_intensity":  0.70,
        "intensity_noise":     0.10,
        "hip_osc":             0.040,
        "wrist_vel":           1.50,
        "ankle_vel":           0.90,
        "torso_lean_mean":     5.0,
        "torso_lean_std":      3.0,
        "shoulder_w_ratio":    0.65,   # arms wide apart
    },
    "squats": {
        "cadence_hz":          0.8,    # ~0.8 squats per second
        "knee_angle_mean":     110.0,
        "knee_angle_amp":      55.0,   # deep knee flex
        "elbow_angle_mean":    145.0,
        "elbow_angle_amp":     20.0,
        "hip_angle_mean":      100.0,
        "hip_angle_amp":       50.0,   # large hip angle change
        "movement_intensity":  0.38,
        "intensity_noise":     0.07,
        "hip_osc":             0.025,
        "wrist_vel":           0.25,
        "ankle_vel":           0.40,
        "torso_lean_mean":     22.0,
        "torso_lean_std":      5.0,
        "shoulder_w_ratio":    0.44,
    },
}

# Features used in FEATURE_FIELDS (excluding metadata)
ML_FEATURES = [f for f in FEATURE_FIELDS if f not in ('timestamp', 'frame_index', 'label', 'weight_kg')]


def _generate_session(label: str, profile: dict, n_frames: int, fps: float,
                      weight_kg: float, noise_scale: float = 1.0,
                      start_frame: int = 0) -> list[dict]:
    """Generate one session's worth of frames for a given activity."""
    rows = []
    cadence   = profile["cadence_hz"]
    intensity = profile["movement_intensity"]
    int_noise = profile["intensity_noise"] * noise_scale

    hip_osc_history: list[float] = []
    intensity_history: list[float] = []

    for i in range(n_frames):
        t       = i / fps
        phase   = 2 * math.pi * cadence * t
        phase2  = phase + math.pi          # counter-phase for opposite limb
        fi      = start_frame + i
        now     = time.time() + fi / fps

        # ---- Joint angles with realistic L/R phase offset ---------------
        lka = profile["knee_angle_mean"]  + profile["knee_angle_amp"]  * math.sin(phase)
        rka = profile["knee_angle_mean"]  + profile["knee_angle_amp"]  * math.sin(phase2)
        lea = profile["elbow_angle_mean"] + profile["elbow_angle_amp"] * math.sin(phase2)
        rea = profile["elbow_angle_mean"] + profile["elbow_angle_amp"] * math.sin(phase)
        lha = profile["hip_angle_mean"]   + profile["hip_angle_amp"]   * math.sin(phase)
        rha = profile["hip_angle_mean"]   + profile["hip_angle_amp"]   * math.sin(phase2)

        # ---- Add Gaussian noise -----------------------------------------
        rng = lambda std: float(np.random.normal(0, std * noise_scale))
        lka += rng(4.0);  rka += rng(4.0)
        lea += rng(5.0);  rea += rng(5.0)
        lha += rng(4.0);  rha += rng(4.0)

        # ---- Clamp to anatomically valid range --------------------------
        lka  = float(np.clip(lka,  30,  180))
        rka  = float(np.clip(rka,  30,  180))
        lea  = float(np.clip(lea,   0,  180))
        rea  = float(np.clip(rea,   0,  180))
        lha  = float(np.clip(lha,  60,  180))
        rha  = float(np.clip(rha,  60,  180))

        # ---- Velocities -------------------------------------------------
        cycle_vel = profile["wrist_vel"]  * abs(math.sin(phase))
        lwv = cycle_vel + abs(rng(0.05));  rwv = cycle_vel + abs(rng(0.05))
        lav = profile["ankle_vel"] * abs(math.sin(phase2)) + abs(rng(0.07))
        rav = profile["ankle_vel"] * abs(math.sin(phase))  + abs(rng(0.07))

        # ---- Angular velocities -----------------------------------------
        ang_scale = cadence * 2 * math.pi
        leav = abs(profile["elbow_angle_amp"] * ang_scale * math.cos(phase2)) + abs(rng(5))
        reav = abs(profile["elbow_angle_amp"] * ang_scale * math.cos(phase))  + abs(rng(5))
        lkav = abs(profile["knee_angle_amp"]  * ang_scale * math.cos(phase))  + abs(rng(5))
        rkav = abs(profile["knee_angle_amp"]  * ang_scale * math.cos(phase2)) + abs(rng(5))

        # ---- Torso & body geometry --------------------------------------
        torso   = float(np.clip(0.35 + np.random.normal(0, 0.01), 0.20, 0.55))
        tl      = float(np.clip(np.random.normal(profile["torso_lean_mean"], profile["torso_lean_std"]), 0, 45))
        swr     = float(np.clip(profile["shoulder_w_ratio"] + rng(0.02), 0.25, 0.80))

        # ---- Hip oscillation (vertical) ----------------------------------
        hip_y   = 0.60 + profile["hip_osc"] * math.sin(phase) + rng(0.005)
        hip_osc_history.append(hip_y)
        if len(hip_osc_history) > 30:
            hip_osc_history.pop(0)
        hip_osc = float(np.std(hip_osc_history)) if len(hip_osc_history) > 1 else 0.0

        # ---- Movement intensity (smoothed) ------------------------------
        raw_int = abs(intensity + rng(int_noise))
        intensity_history.append(raw_int)
        if len(intensity_history) > 15:
            intensity_history.pop(0)
        smoothed = float(np.mean(intensity_history))

        rows.append({
            'timestamp':   now,
            'frame_index': fi,
            'label':       label,
            'weight_kg':   weight_kg,
            'left_elbow_angle':       lea,
            'right_elbow_angle':      rea,
            'left_knee_angle':        lka,
            'right_knee_angle':       rka,
            'left_hip_angle':         lha,
            'right_hip_angle':        rha,
            'elbow_symmetry':         abs(lea - rea),
            'knee_symmetry':          abs(lka - rka),
            'hip_symmetry':           abs(lha - rha),
            'torso_length':           torso,
            'torso_lean_angle':       tl,
            'shoulder_width_ratio':   swr,
            'hip_center_y':           hip_y,
            'vertical_hip_oscillation': hip_osc,
            'movement_intensity':     raw_int,
            'smoothed_intensity':     smoothed,
            'left_wrist_velocity':    lwv,
            'right_wrist_velocity':   rwv,
            'left_ankle_velocity':    lav,
            'right_ankle_velocity':   rav,
            'left_elbow_angular_vel': leav,
            'right_elbow_angular_vel':reav,
            'left_knee_angular_vel':  lkav,
            'right_knee_angular_vel': rkav,
        })

    return rows


def generate_dataset(output_csv: str, sessions: int, fps: float = 30.0,
                     frames_per_session: int = 1800, weight_kg: float | None = None):
    """
    Generate a complete labeled training dataset.

    Args:
        output_csv:        Path to write the CSV.
        sessions:          Number of sessions per activity (diversity multiplier).
        fps:               Simulated frame rate.
        frames_per_session: Frames per session per activity.
        weight_kg:         Fixed weight in kg; None = random per session.
    """
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)

    all_rows: list[dict] = []
    total_frames = 0

    for session_idx in range(sessions):
        w = weight_kg if weight_kg is not None else random.uniform(50.0, 100.0)
        # Vary noise scale slightly per session (person-to-person variation)
        noise = random.uniform(0.8, 1.4)

        for label, profile in PROFILES.items():
            rows = _generate_session(
                label, profile,
                n_frames     = frames_per_session,
                fps          = fps,
                weight_kg    = w,
                noise_scale  = noise,
                start_frame  = total_frames,
            )
            all_rows.extend(rows)
            total_frames += frames_per_session

    # Shuffle to break temporal autocorrelation before train/test split
    random.shuffle(all_rows)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    activity_counts = {}
    for r in all_rows:
        activity_counts[r['label']] = activity_counts.get(r['label'], 0) + 1

    print(f"\n[Dataset Generator] Wrote {len(all_rows)} frames → {output_csv}")
    print(f"{'Activity':<20} {'Frames':>8}")
    print("-" * 30)
    for act, cnt in sorted(activity_counts.items()):
        print(f"  {act:<18} {cnt:>8,}")
    print(f"  {'TOTAL':<18} {len(all_rows):>8,}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Burn-Ex training data")
    parser.add_argument('--output',   default='training_data.csv', help="Output CSV path")
    parser.add_argument('--sessions', type=int, default=5, help="Sessions per activity (default 5)")
    parser.add_argument('--weight',   type=float, default=None,
                        help="Fixed body weight kg (default: random 50-100 kg per session)")
    args = parser.parse_args()
    generate_dataset(args.output, args.sessions, weight_kg=args.weight)


if __name__ == '__main__':
    main()
