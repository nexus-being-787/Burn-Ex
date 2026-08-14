#!/bin/bash
set -e
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PIP_PROGRESS_BAR=off

echo "=== Sapiens2 Phase 1 Setup (VENV) ==="
VENV_DIR="/home/titan/sapiens_venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 1. Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install opencv-python huggingface_hub safetensors transformers timm accelerate

# 2. Setup Sapiens2 Repo
echo "[2/4] Setting up Sapiens2 repository..."
cd ~/sapiens2
pip install -e .

# 3. Download Detector
echo "[3/4] Downloading DETR Person Detector..."
export SAPIENS_CHECKPOINT_ROOT=${HOME}/sapiens2_host
mkdir -p ${SAPIENS_CHECKPOINT_ROOT}/detector
cd ${SAPIENS_CHECKPOINT_ROOT}/detector
huggingface-cli download facebook/detr-resnet-101-dc5 --local-dir detr-resnet-101-dc5

# 4. Download Test Image
echo "[4/4] Downloading test image..."
cd ~/React/Burn-Ex/sapiens_validation
wget -qO test_person.jpg "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?q=80&w=1000&auto=format&fit=crop"

echo "=== Setup Complete! Running validation... ==="
export PYTHONPATH=~/sapiens2:$PYTHONPATH
python validate_sapiens2.py --image test_person.jpg
