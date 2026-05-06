"""
RAFT Optical Flow Visualizer
============================
Moving elements → colored (hue = direction, saturation = speed)
Static elements → white background

Just set INPUT_VIDEO_PATH below and run:
    python raft.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURE HERE
# ════════════════════════════════════════════════════════════════════════════

INPUT_VIDEO_PATH = "test102.mp4"      # Example: "C:/Users/YourName/Videos/test102.mp4"

OUTPUT_DIR       = "output"          # folder where result is saved
MODEL_NAME       = "raft-small"      # "raft-small" (faster) or "raft-things" (accurate)
THRESHOLD        = 2   #1.5               # lower = more pixels colored, higher = only fast motion
SAVE_FRAMES      = False             # True = also save every frame as PNG
DEVICE           = "auto"            # "auto", "cuda", "mps", or "cpu"

# ════════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("❌  PyTorch not found. Run:  pip install torch torchvision")

try:
    from torchvision.models.optical_flow import (
        raft_large, raft_small,
        Raft_Large_Weights, Raft_Small_Weights,
    )
except ImportError:
    sys.exit("❌  torchvision ≥ 0.13 required. Run:  pip install --upgrade torchvision")


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str, device: torch.device):
    print(f"📦  Loading RAFT ({model_name}) …")
    if model_name == "raft-small":
        weights = Raft_Small_Weights.DEFAULT
        model   = raft_small(weights=weights)
    else:
        weights = Raft_Large_Weights.DEFAULT
        model   = raft_large(weights=weights)
    model = model.to(device).eval()
    transforms = weights.transforms()
    return model, transforms


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(frame_bgr: np.ndarray, transforms, device: torch.device):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t       = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    t_trans = transforms(t, t)[0]          # transforms() returns (img1, img2)
    return t_trans.to(device)


# ── Optical flow inference ───────────────────────────────────────────────────

@torch.no_grad()
def compute_flow(model, img1_t, img2_t) -> np.ndarray:
    flow_list = model(img1_t, img2_t)
    flow = flow_list[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
    return flow   # [H, W, 2]


# ── Flow → motion visualisation ──────────────────────────────────────────────

def flow_to_motion_vis(flow: np.ndarray, threshold: float) -> np.ndarray:
    """
    Moving pixels  → HSV color  (hue = direction, brightness = magnitude)
    Static pixels  → white
    Returns BGR uint8 image.
    """
    u, v       = flow[..., 0], flow[..., 1]
    magnitude  = np.sqrt(u**2 + v**2)
    angle      = np.arctan2(v, u)                          # [-π, π]

    hue        = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    max_mag    = np.percentile(magnitude, 99) + 1e-6
    norm_mag   = np.clip(magnitude / max_mag, 0, 1)
    saturation = (norm_mag * 255).astype(np.uint8)
    value      = (norm_mag * 255).astype(np.uint8)

    hsv     = np.stack([hue, saturation, value], axis=-1)
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    colored[magnitude < threshold] = [255, 255, 255]       # static → white
    return colored


# ── Main processing loop ─────────────────────────────────────────────────────

def choose_device(device_name: str) -> torch.device:
    requested = device_name.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        print("⚠️   CUDA not available on this system, falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_available:
            print("⚠️   MPS not available on this system, falling back to CPU.")
            return torch.device("cpu")

    return torch.device(requested)


def process_video():
    # ── Device setup ────────────────────────────────────────────────────────
    device = choose_device(DEVICE)
    print(f"🖥️   Device : {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, transforms = load_model(MODEL_NAME, device)

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open video: {INPUT_VIDEO_PATH}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎬  Input  : {INPUT_VIDEO_PATH}")
    print(f"   Size   : {width}×{height}  |  FPS: {fps:.1f}  |  Frames: {total}")

    # ── Output writer ────────────────────────────────────────────────────────
    stem        = Path(INPUT_VIDEO_PATH).stem
    output_path = os.path.join(OUTPUT_DIR, f"{stem}_flow.mp4")
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if SAVE_FRAMES:
        frames_dir = os.path.join(OUTPUT_DIR, f"{stem}_frames")
        os.makedirs(frames_dir, exist_ok=True)

    # ── Read first frame ─────────────────────────────────────────────────────
    ret, prev_frame = cap.read()
    if not ret:
        sys.exit("❌  Video appears to be empty.")

    prev_t    = preprocess(prev_frame, transforms, device)
    frame_idx = 0

    print("⚙️   Processing …")
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        curr_t = preprocess(curr_frame, transforms, device)
        flow   = compute_flow(model, prev_t, curr_t)

        # Resize flow map if shapes differ (safety check)
        if flow.shape[:2] != (height, width):
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)

        vis = flow_to_motion_vis(flow, threshold=THRESHOLD)
        writer.write(vis)

        if SAVE_FRAMES:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.png"), vis)

        prev_t = curr_t

        if frame_idx % 5 == 0 or frame_idx == 1:
            pct = frame_idx / max(total - 1, 1) * 100
            print(f"   Frame {frame_idx:>5} / {total-1}  ({pct:.1f}%)", end="\r")

    cap.release()
    writer.release()
    print(f"\n✅  Saved → {output_path}")
    if SAVE_FRAMES:
        print(f"🖼️   Frames → {frames_dir}/")


if __name__ == "__main__":
    process_video()
