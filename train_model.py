#!/usr/bin/env python3
"""
Burn-Ex -- Model Trainer (Day 2)
----------------------------------
Trains a calorie-regression ensemble on extracted pose features.

Accuracy design:
  1. Target variable: continuous kcal/min (from MET formula, per-frame weight)
  2. Feature set: all 25 biomechanical features from pose_pipeline.py
  3. Feature engineering: 5-frame rolling mean + std for temporal context
  4. Models trained:
       A. Random Forest Regressor (baseline)
       B. Gradient Boosting Regressor (GBR)
       C. Stacking Ensemble (RF + GBR meta-learner → Ridge)
  5. Hyperparameter search: RandomizedSearchCV (5-fold CV, 50 iters)
  6. Evaluation: MAE, RMSE, R² on stratified held-out 20% test set
  7. Artifacts saved:
       burn_ex_model.pkl  – best pipeline (scaler + model)
       scaler.pkl         – StandardScaler
       model_report.txt   – detailed accuracy report

Usage:
  python train_model.py --data training_data.csv --weight 70
  python train_model.py --data session1.csv training_data.csv --weight 70
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from met_calculator import ACTIVITY_METS, kcal_per_min
from constants import ML_FEATURES, ROLLING_COLS, ACTIVITY_NAMES

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_OUTPUT   = 'burn_ex_model.pkl'
SCALER_OUTPUT  = 'scaler.pkl'
REPORT_OUTPUT  = 'model_report.txt'

# All feature / rolling definitions are centralised in constants.py


# --------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------

def load_data(csv_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate one or more session CSVs."""
    dfs = []
    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"[Warning] File not found, skipping: {p}")
            continue
        df = pd.read_csv(p)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No valid CSV files found.")
    return pd.concat(dfs, ignore_index=True)


def label_to_kcal(row) -> float:
    """Compute ground-truth kcal/min from label + weight_kg."""
    met = ACTIVITY_METS.get(row['label'], 1.0)
    return kcal_per_min(met, float(row.get('weight_kg', 70.0)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling-window temporal context features."""
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    for col in ROLLING_COLS:
        if col in df.columns:
            df[f'{col}_roll5_mean'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_roll5_std']  = df[col].rolling(5, min_periods=1).std().fillna(0)

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract the final feature matrix and column names."""
    base_cols    = [c for c in ML_FEATURES if c in df.columns]
    rolling_cols = [c for c in df.columns if c.endswith(('_roll5_mean', '_roll5_std'))]
    all_cols     = base_cols + rolling_cols
    X = df[all_cols].fillna(0.0).values
    return X, all_cols


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------

def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def build_gbr() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=3,
        random_state=42,
    )


def build_stacking(rf: RandomForestRegressor,
                   gbr: GradientBoostingRegressor) -> StackingRegressor:
    return StackingRegressor(
        estimators=[('rf', rf), ('gbr', gbr)],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------
# Hyper-parameter search grids
# --------------------------------------------------------------------------

RF_PARAM_GRID = {
    'n_estimators':    [100, 200, 300, 500],
    'max_depth':       [None, 8, 15, 25],
    'min_samples_leaf':[1, 2, 4],
    'max_features':    ['sqrt', 'log2', 0.7],
}

GBR_PARAM_GRID = {
    'n_estimators':  [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.10],
    'max_depth':     [3, 5, 7],
    'subsample':     [0.7, 0.8, 1.0],
}


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------

def regression_metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE%': mape}


def print_metrics(name: str, metrics: dict):
    print(f"  {name}")
    print(f"    MAE:   {metrics['MAE']:.4f} kcal/min")
    print(f"    RMSE:  {metrics['RMSE']:.4f} kcal/min")
    print(f"    R²:    {metrics['R2']:.4f}")
    print(f"    MAPE:  {metrics['MAPE%']:.2f}%")


# --------------------------------------------------------------------------
# Main training function
# --------------------------------------------------------------------------

def train(csv_paths: list[str], weight_kg: float,
          output_dir: str = '.', tune: bool = True) -> dict:
    t0 = time.time()

    # 1. Load & preprocess
    print("[Trainer] Loading data...")
    df = load_data(csv_paths)
    print(f"          {len(df):,} frames loaded from {len(csv_paths)} file(s)")
    print(f"          Activity distribution:\n{df['label'].value_counts().to_string()}")

    # 2. Assign ground-truth calorie target
    df['kcal_per_min'] = df.apply(label_to_kcal, axis=1)

    # 3. Drop unlabeled frames (no useful target)
    df = df[df['label'] != 'unlabeled'].copy()

    # 4. Feature engineering (rolling windows)
    print("[Trainer] Engineering temporal features...")
    df = engineer_features(df)
    X, feature_cols = build_feature_matrix(df)
    y = df['kcal_per_min'].values

    print(f"          Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")

    # 5. Train/test split — stratified by activity label
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 6. Feature scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    results: dict[str, dict] = {}

    # 7a. Random Forest
    print("\n[Trainer] Training Random Forest...")
    rf = build_rf()
    if tune:
        search = RandomizedSearchCV(rf, RF_PARAM_GRID, n_iter=20,
                                    cv=5, scoring='neg_root_mean_squared_error',
                                    n_jobs=-1, random_state=42, verbose=0)
        search.fit(X_train_s, y_train)
        rf = search.best_estimator_
        print(f"          Best RF params: {search.best_params_}")
    else:
        rf.fit(X_train_s, y_train)

    rf_pred = rf.predict(X_test_s)
    results['Random Forest'] = regression_metrics(y_test, rf_pred)
    print_metrics('Random Forest', results['Random Forest'])

    # 7b. Gradient Boosting
    print("\n[Trainer] Training Gradient Boosting...")
    gbr = build_gbr()
    if tune:
        search2 = RandomizedSearchCV(gbr, GBR_PARAM_GRID, n_iter=20,
                                     cv=5, scoring='neg_root_mean_squared_error',
                                     n_jobs=-1, random_state=42, verbose=0)
        search2.fit(X_train_s, y_train)
        gbr = search2.best_estimator_
        print(f"          Best GBR params: {search2.best_params_}")
    else:
        gbr.fit(X_train_s, y_train)

    gbr_pred = gbr.predict(X_test_s)
    results['Gradient Boosting'] = regression_metrics(y_test, gbr_pred)
    print_metrics('Gradient Boosting', results['Gradient Boosting'])

    # 7c. Stacking Ensemble
    print("\n[Trainer] Training Stacking Ensemble (RF + GBR → Ridge)...")
    stack = build_stacking(build_rf(), build_gbr())
    stack.fit(X_train_s, y_train)
    stack_pred = stack.predict(X_test_s)
    results['Stacking Ensemble'] = regression_metrics(y_test, stack_pred)
    print_metrics('Stacking Ensemble', results['Stacking Ensemble'])

    # 8. Select best model by RMSE
    best_name  = min(results, key=lambda k: results[k]['RMSE'])
    best_model = {'Random Forest': rf, 'Gradient Boosting': gbr, 'Stacking Ensemble': stack}[best_name]
    print(f"\n[Trainer] ✓ Best model: {best_name}  (RMSE={results[best_name]['RMSE']:.4f})")

    # 9. Build final Pipeline (scaler + model) so inference is a single .predict() call
    # We save them separately AND as a combined pipeline
    final_pipeline = Pipeline([
        ('scaler', scaler),
        ('model',  best_model),
    ])

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path  = os.path.join(output_dir, MODEL_OUTPUT)
    scaler_path = os.path.join(output_dir, SCALER_OUTPUT)
    report_path = os.path.join(output_dir, REPORT_OUTPUT)

    with open(model_path, 'wb') as f:
        pickle.dump({
            'pipeline':     final_pipeline,
            'feature_cols': feature_cols,
            'best_model':   best_name,
        }, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # 10. Write accuracy report
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  Burn-Ex Model Accuracy Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data files: {csv_paths}\n")
        f.write(f"Training samples: {len(X_train):,}\n")
        f.write(f"Test samples:     {len(X_test):,}\n")
        f.write(f"Features:         {len(feature_cols)}\n\n")
        for model_name, m in results.items():
            marker = " ← BEST" if model_name == best_name else ""
            f.write(f"{model_name}{marker}\n")
            f.write(f"  MAE:   {m['MAE']:.4f} kcal/min\n")
            f.write(f"  RMSE:  {m['RMSE']:.4f} kcal/min\n")
            f.write(f"  R²:    {m['R2']:.4f}\n")
            f.write(f"  MAPE:  {m['MAPE%']:.2f}%\n\n")
        f.write(f"Feature importance (RF):\n")
        for feat, imp in sorted(zip(feature_cols, rf.feature_importances_),
                                key=lambda x: -x[1])[:15]:
            f.write(f"  {feat:<35} {imp:.4f}\n")
        f.write(f"\nTotal training time: {time.time() - t0:.1f}s\n")

    print(f"\n[Trainer] Artifacts saved:")
    print(f"          Model   → {model_path}")
    print(f"          Scaler  → {scaler_path}")
    print(f"          Report  → {report_path}")

    # 11. Print top feature importances
    print("\n[Trainer] Top 10 Feature Importances (Random Forest):")
    fi_pairs = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1])[:10]
    for feat, imp in fi_pairs:
        bar = "\u2588" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    # 12. Train Activity Auto-Classifier (improvement: no manual key presses needed in live mode)
    print("\n[Trainer] Training Activity Auto-Classifier (RF)...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Build a fresh scaler+RF classifier on labelled data
    X_raw     = df[feature_cols].fillna(0.0).values
    y_cls     = df['label'].values
    scaler_c  = StandardScaler()
    X_cls_s   = scaler_c.fit_transform(X_raw)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cls_s, y_cls, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                 n_jobs=-1, random_state=42)
    clf.fit(X_tr_c, y_tr_c)
    clf_acc = accuracy_score(y_te_c, clf.predict(X_te_c))
    print(f"  Activity classifier accuracy: {clf_acc*100:.1f}%")

    clf_pipeline = Pipeline([('scaler', scaler_c), ('clf', clf)])
    clf_path     = os.path.join(output_dir, 'activity_classifier.pkl')
    with open(clf_path, 'wb') as f:
        pickle.dump({
            'pipeline':     clf_pipeline,
            'feature_cols': feature_cols,
            'classes':      list(clf.classes_),
        }, f)
    print(f"          Classifier \u2192 {clf_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Burn-Ex calorie regression model")
    parser.add_argument('--data', nargs='+', default=['training_data.csv'],
                        help="CSV file(s) to train on")
    parser.add_argument('--weight',  type=float, default=70.0,
                        help="Default body weight kg if not in CSV (default 70)")
    parser.add_argument('--no-tune', action='store_true',
                        help="Skip hyperparameter search for faster training")
    parser.add_argument('--output',  default='.',
                        help="Directory to save model artifacts (default .)")
    args = parser.parse_args()

    train(args.data, weight_kg=args.weight, output_dir=args.output, tune=not args.no_tune)


if __name__ == '__main__':
    main()
