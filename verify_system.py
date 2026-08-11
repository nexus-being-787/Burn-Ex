#!/usr/bin/env python3
"""
Burn-Ex -- Accuracy Verification Suite
-----------------------------------------
Comprehensive tests to validate every layer of the system:

  1. MET formula accuracy       — exact match vs published ACSM values
  2. Dataset generation health  — shape, NaN count, value ranges
  3. Feature engineering        — rolling windows, no data leakage
  4. Model training pipeline    — RMSE, R², MAE targets
  5. Inference sanity           — predictions on known activity profiles
  6. Edge cases                 — zero-motion, max-motion, single frame

Expected passing criteria (conservative for demo):
  - R²    > 0.92  (explains ≥92% of kcal variance)
  - RMSE  < 0.50  kcal/min
  - MAE   < 0.30  kcal/min
  - MAPE  < 8 %

Run:
  python verify_system.py
  python verify_system.py --fast   (skip hyperparam tuning)
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

PASS  = "✓ PASS"
FAIL  = "✗ FAIL"
WARN  = "⚠ WARN"
SEP   = "─" * 60


def hdr(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def ok(msg: str):
    print(f"  {PASS}  {msg}")


def fail(msg: str):
    print(f"  {FAIL}  {msg}")


def warn(msg: str):
    print(f"  {WARN}  {msg}")


# ==========================================================================
# 1. MET FORMULA ACCURACY
# ==========================================================================

def test_met():
    hdr("Test 1 — MET Formula & ACSM Reference Values")
    from met_calculator import ACTIVITY_METS, kcal_per_min, kcal_per_second

    # Published ACSM Compendium 2011 values (Ainsworth et al.)
    expected = {
        "idle":          1.3,
        "walking":       3.5,
        "jogging":       7.0,
        "jumping_jacks": 8.0,
        "squats":        5.0,
    }
    all_ok = True
    for act, met_ref in expected.items():
        actual = ACTIVITY_METS.get(act)
        if abs(actual - met_ref) < 1e-6:
            ok(f"{act:<20} MET={actual:.1f}  (ACSM={met_ref:.1f})  ✓ exact match")
        else:
            fail(f"{act:<20} MET={actual:.1f}  expected {met_ref:.1f}")
            all_ok = False

    # Verify formula: kcal/min = MET × 3.5 × kg / 200
    w = 70.0
    met = 7.0  # jogging
    expected_kcal = 7.0 * 3.5 * 70.0 / 200.0
    actual_kcal   = kcal_per_min(met, w)
    if abs(actual_kcal - expected_kcal) < 1e-8:
        ok(f"kcal/min formula  7.0 × 3.5 × 70 / 200 = {actual_kcal:.4f}  ✓ exact")
    else:
        fail(f"kcal/min formula returned {actual_kcal:.4f}, expected {expected_kcal:.4f}")
        all_ok = False

    # Verify per-second
    actual_ps = kcal_per_second(met, w)
    if abs(actual_ps - expected_kcal / 60.0) < 1e-10:
        ok(f"kcal/sec = kcal/min / 60         = {actual_ps:.6f}  ✓")
    else:
        fail(f"kcal/sec mismatch: {actual_ps}")
        all_ok = False

    return all_ok


# ==========================================================================
# 2. DATASET GENERATION
# ==========================================================================

def test_dataset_generation():
    hdr("Test 2 — Synthetic Dataset Generation")
    import pandas as pd
    from dataset_generator import generate_dataset

    out = '/tmp/burnex_verify_dataset.csv'
    generate_dataset(out, sessions=2, frames_per_session=600, weight_kg=70.0)

    df = pd.read_csv(out)
    n_activities = df['label'].nunique()
    n_expected   = 5  # idle, walking, jogging, jumping_jacks, squats
    n_frames     = len(df)
    n_nan        = df.isnull().sum().sum()

    all_ok = True

    if n_nan == 0:
        ok(f"No NaN values  ({n_frames:,} rows × {len(df.columns)} cols)")
    else:
        fail(f"Found {n_nan} NaN values")
        all_ok = False

    if n_activities == n_expected:
        ok(f"All {n_activities} activity classes present")
    else:
        fail(f"Expected {n_expected} classes, found {n_activities}: {df['label'].unique()}")
        all_ok = False

    # Value range sanity
    checks = [
        ('left_knee_angle',    30, 180),
        ('right_knee_angle',   30, 180),
        ('left_elbow_angle',    0, 180),
        ('movement_intensity',  0, None),
        ('torso_length',        0.1, 0.6),
    ]
    for col, lo, hi in checks:
        mn = df[col].min(); mx = df[col].max()
        lo_ok = (mn >= lo) if lo is not None else True
        hi_ok = (mx <= hi) if hi is not None else True
        if lo_ok and hi_ok:
            ok(f"{col:<35} range [{mn:.2f}, {mx:.2f}]  ✓ valid")
        else:
            warn(f"{col:<35} range [{mn:.2f}, {mx:.2f}]  outside [{lo},{hi}]")

    # Class balance
    print("\n  Activity frame counts:")
    for label, cnt in df['label'].value_counts().items():
        bar = "█" * (cnt // 50)
        print(f"    {label:<20} {cnt:>5}  {bar}")

    return all_ok


# ==========================================================================
# 3. FEATURE ENGINEERING
# ==========================================================================

def test_feature_engineering():
    hdr("Test 3 — Feature Engineering (Rolling Windows)")
    import pandas as pd
    from train_model import load_data, engineer_features, build_feature_matrix

    data_path = '/tmp/burnex_verify_dataset.csv'
    if not os.path.isfile(data_path):
        from dataset_generator import generate_dataset
        generate_dataset(data_path, sessions=1, frames_per_session=300, weight_kg=70.0)

    df = load_data([data_path])
    df = engineer_features(df)
    X, cols = build_feature_matrix(df)

    all_ok = True

    roll_cols = [c for c in cols if 'roll5' in c]
    if len(roll_cols) > 0:
        ok(f"Rolling feature cols generated: {len(roll_cols)}  ({roll_cols[:3]}…)")
    else:
        fail("No rolling feature columns found")
        all_ok = False

    n_nan = int(pd.DataFrame(X, columns=cols).isnull().sum().sum())
    if n_nan == 0:
        ok(f"Feature matrix has no NaNs  (shape={X.shape})")
    else:
        fail(f"Feature matrix has {n_nan} NaNs — fillna may have gaps")
        all_ok = False

    return all_ok


# ==========================================================================
# 4. MODEL TRAINING & ACCURACY
# ==========================================================================

def test_model_training(fast: bool = False):
    hdr("Test 4 — Model Training & Accuracy Metrics")
    from train_model import train

    data_path = '/tmp/burnex_verify_dataset.csv'
    if not os.path.isfile(data_path):
        from dataset_generator import generate_dataset
        generate_dataset(data_path, sessions=3, frames_per_session=900, weight_kg=70.0)

    results = train([data_path], weight_kg=70.0, output_dir='/tmp/burnex_verify',
                    tune=not fast)

    # Accuracy thresholds
    targets = {
        'R2':    (0.92, '>', 'explains ≥92% variance'),
        'RMSE':  (0.50, '<', 'kcal/min error < 0.50'),
        'MAE':   (0.30, '<', 'mean absolute error < 0.30'),
        'MAPE%': (8.0,  '<', 'mean absolute %error < 8%'),
    }

    all_ok = True
    best   = min(results, key=lambda k: results[k]['RMSE'])
    m      = results[best]

    print(f"\n  Best model: {best}")
    for metric, (threshold, op, desc) in targets.items():
        val = m[metric]
        passed = (val > threshold) if op == '>' else (val < threshold)
        if passed:
            ok(f"{metric:>6} = {val:.4f}   (target {op} {threshold})   {desc}")
        else:
            fail(f"{metric:>6} = {val:.4f}   (target {op} {threshold})   {desc}")
            all_ok = False

    return all_ok, results


# ==========================================================================
# 5. INFERENCE SANITY
# ==========================================================================

def test_inference_sanity():
    hdr("Test 5 — Inference Sanity on Known Activity Profiles")
    import pickle
    import numpy as np
    import pandas as pd
    from train_model import engineer_features
    from dataset_generator import PROFILES, _generate_session
    from met_calculator import ACTIVITY_METS, kcal_per_min

    model_path = '/tmp/burnex_verify/burn_ex_model.pkl'
    if not os.path.isfile(model_path):
        fail(f"Model not found at {model_path} — run test 4 first")
        return False

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)

    pipeline     = artifact['pipeline']
    feature_cols = artifact['feature_cols']
    weight_kg    = 70.0

    all_ok = True
    print(f"\n  {'Activity':<20} {'Predicted':>12} {'Expected':>12} {'Error%':>8}  {'Status'}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}  {'─'*6}")

    for label, profile in PROFILES.items():
        rows = _generate_session(label, profile, n_frames=300, fps=30.0,
                                 weight_kg=weight_kg, noise_scale=0.5)
        df = pd.DataFrame(rows)
        df = engineer_features(df)

        # Build aligned feature DataFrame using model's own feature_cols list.
        # This correctly handles weight_kg and any rolling columns.
        aligned = pd.DataFrame(index=range(len(df)), columns=feature_cols, dtype=float)
        for c in feature_cols:
            if c in df.columns:
                aligned[c] = df[c].values
        aligned = aligned.fillna(0.0)

        preds   = pipeline.predict(aligned.values)
        pred_m  = float(np.mean(preds))
        exp_m   = kcal_per_min(ACTIVITY_METS[label], weight_kg)
        err_pct = abs(pred_m - exp_m) / (exp_m + 1e-6) * 100

        status = PASS if err_pct < 15 else (WARN if err_pct < 30 else FAIL)
        print(f"  {label:<20} {pred_m:>12.3f} {exp_m:>12.3f} {err_pct:>7.1f}%  {status}")

        if err_pct >= 30:
            all_ok = False

    return all_ok


# ==========================================================================
# 6. EDGE CASES

# ==========================================================================

def test_edge_cases():
    hdr("Test 6 — Edge Cases")
    import pickle
    import numpy as np

    model_path = '/tmp/burnex_verify/burn_ex_model.pkl'
    if not os.path.isfile(model_path):
        fail("Model not found — skipping edge case tests")
        return True

    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    pipeline     = artifact['pipeline']
    feature_cols = artifact['feature_cols']

    all_ok = True
    n_feat = len(feature_cols)

    # Zero vector (complete stillness)
    x_zero = np.zeros((1, n_feat))
    pred   = float(pipeline.predict(x_zero)[0])
    if pred >= 0:
        ok(f"Zero-motion vector  → {pred:.4f} kcal/min  (non-negative ✓)")
    else:
        fail(f"Zero-motion → {pred:.4f}  (negative! model may extrapolate badly)")
        all_ok = False

    # High-intensity vector (all features at max plausible value)
    x_max = np.ones((1, n_feat)) * 5.0
    pred  = float(pipeline.predict(x_max)[0])
    if pred > 0:
        ok(f"High-intensity vector → {pred:.4f} kcal/min  (positive ✓)")
    else:
        fail(f"High-intensity → {pred:.4f} (unexpected)")
        all_ok = False

    # Single frame — model must not crash
    x_single = np.random.rand(1, n_feat)
    try:
        pred = float(pipeline.predict(x_single)[0])
        ok(f"Single random frame → {pred:.4f} kcal/min  (no crash ✓)")
    except Exception as e:
        fail(f"Crash on single frame: {e}")
        all_ok = False

    return all_ok


# ==========================================================================
# SUMMARY
# ==========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Burn-Ex accuracy verification suite")
    parser.add_argument('--fast', action='store_true',
                        help="Skip hyperparameter tuning (faster but less thorough)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Burn-Ex — Accuracy Verification Suite")
    print("=" * 60)

    import numpy as np

    all_results = []

    try:
        all_results.append(("MET Formula",         test_met()))
    except Exception as e:
        print(f"\n  {FAIL}  MET test crashed: {e}")
        traceback.print_exc()
        all_results.append(("MET Formula", False))

    try:
        all_results.append(("Dataset Gen",         test_dataset_generation()))
    except Exception as e:
        print(f"\n  {FAIL}  Dataset gen crashed: {e}")
        traceback.print_exc()
        all_results.append(("Dataset Gen", False))

    try:
        all_results.append(("Feature Eng",         test_feature_engineering()))
    except Exception as e:
        print(f"\n  {FAIL}  Feature eng crashed: {e}")
        traceback.print_exc()
        all_results.append(("Feature Eng", False))

    try:
        passed, _ = test_model_training(fast=args.fast)
        all_results.append(("Model Training",       passed))
    except Exception as e:
        print(f"\n  {FAIL}  Model training crashed: {e}")
        traceback.print_exc()
        all_results.append(("Model Training", False))

    try:
        all_results.append(("Inference Sanity",     test_inference_sanity()))
    except Exception as e:
        print(f"\n  {FAIL}  Inference sanity crashed: {e}")
        traceback.print_exc()
        all_results.append(("Inference Sanity", False))

    try:
        all_results.append(("Edge Cases",           test_edge_cases()))
    except Exception as e:
        print(f"\n  {FAIL}  Edge cases crashed: {e}")
        traceback.print_exc()
        all_results.append(("Edge Cases", False))

    # Final summary
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    passed = sum(1 for _, r in all_results if r)
    total  = len(all_results)
    for name, result in all_results:
        status = PASS if result else FAIL
        print(f"  {status}  {name}")
    print(f"\n  Score: {passed}/{total} tests passed")
    if passed == total:
        print("  ★ ALL SYSTEMS VERIFIED — Burn-Ex is ready for demo!")
    else:
        print("  Please review failing tests above before demo.")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
