#!/usr/bin/env bash
# ============================================================
# Burn-Ex — Universal launcher (bash, works from any shell)
# ============================================================
# Usage: bash run.sh [app|cli|train|verify] [--weight 70] [--height 175]
#
# Does NOT require venv activation — calls ./venv/bin/python directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
  echo "[Burn-Ex] ERROR: venv not found. Create it with:"
  echo "  python3 -m venv venv"
  echo "  ./venv/bin/pip install mediapipe opencv-python scikit-learn pandas numpy matplotlib flask"
  exit 1
fi

CMD="${1:-app}"
shift || true

case "$CMD" in
  app)
    echo "[Burn-Ex] Starting Web Dashboard → http://localhost:5000"
    "$PYTHON" "$SCRIPT_DIR/app.py" "$@"
    ;;
  cli)
    echo "[Burn-Ex] Starting CLI Estimator"
    "$PYTHON" "$SCRIPT_DIR/realtime_estimator.py" "$@"
    ;;
  train)
    echo "[Burn-Ex] Training model on training_data.csv"
    "$PYTHON" "$SCRIPT_DIR/train_model.py" --data "$SCRIPT_DIR/training_data.csv" "$@"
    ;;
  generate)
    echo "[Burn-Ex] Generating synthetic training data"
    "$PYTHON" "$SCRIPT_DIR/dataset_generator.py" "$@"
    ;;
  collect)
    echo "[Burn-Ex] Starting pose data collection session"
    "$PYTHON" "$SCRIPT_DIR/pose_pipeline.py" "$@"
    ;;
  verify)
    echo "[Burn-Ex] Running accuracy verification suite"
    "$PYTHON" "$SCRIPT_DIR/verify_system.py" --fast "$@"
    ;;
  *)
    echo "Usage: bash run.sh [app|cli|train|generate|collect|verify] [options]"
    echo ""
    echo "  app      Launch web dashboard (default)"
    echo "  cli      CLI live estimator"
    echo "  train    Train model on training_data.csv"
    echo "  generate Generate synthetic training data"
    echo "  collect  Start webcam data-collection session"
    echo "  verify   Run 6-test accuracy suite"
    echo ""
    echo "Examples:"
    echo "  bash run.sh app --weight 70"
    echo "  bash run.sh cli --weight 70 --height 175 --age 25"
    echo "  bash run.sh train --no-tune"
    echo "  bash run.sh verify"
    exit 1
    ;;
esac
