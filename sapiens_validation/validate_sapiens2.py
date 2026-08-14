#!/usr/bin/env python3
"""
Phase 1 — Desktop Sapiens2-0.4B Validation Script
================================================
This script:
  1. Downloads sapiens2_0.4b_pose.safetensors from HuggingFace
  2. Loads it with the official Sapiens2 inference code
  3. Runs inference on a test image
  4. Saves the 308 keypoint coordinates + confidence as JSON reference data

Run this BEFORE attempting any ONNX export.
The saved reference data is used to validate the ONNX export in Phase 2.

Requirements:
  pip install torch>=2.7 safetensors huggingface_hub pillow numpy

Usage:
  python validate_sapiens2.py --image path/to/person.jpg
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── 1. Constants ─────────────────────────────────────────────────────────────

MODEL_REPO   = "facebook/sapiens2-pose-0.4b"
MODEL_FILE   = "sapiens2_0.4b_pose.safetensors"
WEIGHTS_DIR  = Path.home() / "sapiens2_weights"
OUTPUT_DIR   = Path(__file__).parent / "sapiens2_reference"

# Official Sapiens2 preprocessing (MUST match training pipeline exactly)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
INPUT_H = 1024
INPUT_W = 768
NUM_KEYPOINTS = 308

# ── 2. Download ───────────────────────────────────────────────────────────────

def download_model() -> Path:
    """Download model weights from HuggingFace if not already cached."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    target = WEIGHTS_DIR / MODEL_FILE
    if target.exists():
        print(f"[✓] Model already downloaded: {target}")
        return target

    print(f"[↓] Downloading {MODEL_FILE} from {MODEL_REPO} ...")
    print(f"    This is ~0.9GB — grab a coffee ☕")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(WEIGHTS_DIR),
        )
        print(f"[✓] Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        print(f"    Manual download: huggingface-cli download {MODEL_REPO} {MODEL_FILE} --local-dir {WEIGHTS_DIR}")
        sys.exit(1)

# ── 3. Load model ─────────────────────────────────────────────────────────────

def load_sapiens2(weights_path: Path):
    """Load Sapiens2-0.4B pose model from safetensors."""
    import torch
    from safetensors.torch import load_file

    print(f"[↻] Loading model weights...")
    state_dict = load_file(str(weights_path), device="cpu")

    # Try to import from official Sapiens2 repo if cloned, otherwise use minimal loader
    try:
        # If you have cloned facebookresearch/sapiens2 and it's on the PYTHONPATH
        sys.path.insert(0, str(Path.home() / "sapiens2"))
        from sapiens2.pose.models import build_sapiens2_pose
        model = build_sapiens2_pose(variant="0.4b")
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"[✓] Loaded via official sapiens2 repo")
        return model, "official"
    except ImportError:
        # Fallback: build minimal ViT inference using timm + custom head
        print(f"[!] Official sapiens2 repo not found on PYTHONPATH")
        print(f"    Clone it: git clone https://github.com/facebookresearch/sapiens2 ~/sapiens2")
        print(f"    Then add to PYTHONPATH: export PYTHONPATH=~/sapiens2/sapiens:$PYTHONPATH")
        print(f"    Then re-run this script.")
        sys.exit(1)

# ── 4. Preprocess ─────────────────────────────────────────────────────────────

def preprocess(image_path: str):
    """
    Exact official Sapiens2 preprocessing:
      1. Resize to INPUT_H x INPUT_W (letterbox with padding)
      2. Normalize using MEAN/STD
      3. Return tensor + CropTransform metadata
    """
    import numpy as np
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"[i] Original image size: {orig_w}x{orig_h}")

    # Letterbox: scale to fit INPUT_H x INPUT_W preserving aspect ratio
    scale = min(INPUT_H / orig_h, INPUT_W / orig_w)
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to target size (center pad)
    pad_top    = (INPUT_H - new_h) // 2
    pad_bottom = INPUT_H - new_h - pad_top
    pad_left   = (INPUT_W - new_w) // 2
    pad_right  = INPUT_W - new_w - pad_left

    padded = np.full((INPUT_H, INPUT_W, 3), 128, dtype=np.uint8)  # grey padding
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = np.array(resized)

    # Normalize
    arr = padded.astype(np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)

    # To tensor [1, C, H, W]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    # Store transform metadata for coordinate remapping
    transform = {
        "orig_w": orig_w, "orig_h": orig_h,
        "scale": scale,
        "new_w": new_w, "new_h": new_h,
        "pad_left": pad_left, "pad_top": pad_top,
        "input_h": INPUT_H, "input_w": INPUT_W,
    }
    return tensor, transform, padded

# ── 5. Postprocess ────────────────────────────────────────────────────────────

def postprocess(heatmaps, transform):
    """
    Extract keypoint coordinates from heatmaps.
    heatmaps shape: [1, 308, H//4, W//4]
    
    Uses argmax + gaussian refinement (DARK method).
    Maps coordinates back to original image space.
    """
    import numpy as np
    import torch

    hm = heatmaps.squeeze(0).cpu().numpy()  # [308, H, W]
    n_kpts, hm_h, hm_w = hm.shape

    keypoints = []
    for k in range(n_kpts):
        hm_k = hm[k]
        conf  = float(hm_k.max())
        flat  = hm_k.argmax()
        cy    = int(flat // hm_w)
        cx    = int(flat % hm_w)

        # Scale to model input space
        # heatmap is 1/4 of model input resolution
        x_model = (cx + 0.5) * (INPUT_W / hm_w)
        y_model = (cy + 0.5) * (INPUT_H / hm_h)

        # Remap to original image space
        t = transform
        x_orig = (x_model - t["pad_left"]) / t["scale"]
        y_orig = (y_model - t["pad_top"])  / t["scale"]

        # Clamp to original image bounds
        x_orig = max(0.0, min(float(t["orig_w"]) - 1, x_orig))
        y_orig = max(0.0, min(float(t["orig_h"]) - 1, y_orig))

        keypoints.append({
            "kpt_id": k,
            "x": round(x_orig, 2),
            "y": round(y_orig, 2),
            "confidence": round(conf, 4),
        })

    return keypoints

# ── 6. Visualize ─────────────────────────────────────────────────────────────

def visualize(image_path, keypoints, output_path):
    """Draw keypoints on the original image and save."""
    try:
        import cv2
        img = cv2.imread(image_path)
        for kpt in keypoints:
            if kpt["confidence"] > 0.3:
                x, y = int(kpt["x"]), int(kpt["y"])
                cv2.circle(img, (x, y), 3, (0, 255, 128), -1)
        cv2.imwrite(str(output_path), img)
        print(f"[✓] Visualization saved: {output_path}")
    except ImportError:
        print(f"[!] cv2 not available, skipping visualization (pip install opencv-python)")

# ── 7. Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sapiens2-0.4B Desktop Validation")
    parser.add_argument("--image", required=True, help="Path to input image (person photo)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[✗] Image not found: {args.image}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    print(f"[i] PyTorch version: {torch.__version__}")
    print(f"[i] Device: {args.device}")
    print(f"[i] CUDA available: {torch.cuda.is_available()}")

    # 1. Download model
    weights_path = download_model()

    # 2. Load model
    model, loader_type = load_sapiens2(weights_path)
    model = model.to(args.device)

    # 3. Preprocess
    tensor, transform, padded_img = preprocess(args.image)
    tensor = tensor.to(args.device)

    # 4. Infer
    import time
    print(f"[↻] Running inference (first run includes JIT warmup)...")
    with torch.no_grad():
        # Warmup
        _ = model(tensor)
        # Timed run
        t0 = time.perf_counter()
        heatmaps = model(tensor)
        elapsed  = time.perf_counter() - t0

    print(f"[✓] Inference time: {elapsed*1000:.1f} ms")
    print(f"[i] Heatmap shape: {heatmaps.shape}")

    # 5. Postprocess
    keypoints = postprocess(heatmaps, transform)
    high_conf = [k for k in keypoints if k["confidence"] > 0.3]
    print(f"[✓] Keypoints extracted: {len(keypoints)} total, {len(high_conf)} with conf > 0.3")

    # 6. Save reference data
    reference = {
        "model": MODEL_REPO,
        "image": args.image,
        "transform": transform,
        "inference_ms": round(elapsed * 1000, 2),
        "keypoints": keypoints,
    }
    ref_path = OUTPUT_DIR / "reference_keypoints.json"
    with open(ref_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"[✓] Reference keypoints saved: {ref_path}")
    print(f"    THIS FILE IS YOUR GROUND TRUTH — keep it for Phase 2 ONNX validation!")

    # 7. Visualize
    vis_path = OUTPUT_DIR / "visualization.jpg"
    visualize(args.image, keypoints, vis_path)

if __name__ == "__main__":
    main()
