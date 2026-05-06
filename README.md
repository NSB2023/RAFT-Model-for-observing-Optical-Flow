# RAFT-Model-for-observing-Optical-Flow
## Project Summary

This project implements a RAFT-based optical flow visualizer for video motion analysis. It uses pretrained RAFT models from `torchvision` to estimate pixel-level motion between consecutive video frames, then converts that motion into a color-coded visualization.

Moving regions are shown in color, where the color represents motion direction and brightness/intensity represents motion speed. Static or low-motion regions are converted to a white background, making moving objects easier to observe.

## What Has Been Done So Far

- Built a Python script for processing video files frame by frame.
- Integrated pretrained RAFT optical flow models from `torchvision`.
- Added support for both `raft-small` and larger RAFT models.
- Implemented automatic device selection:
  - CUDA for supported NVIDIA GPUs
  - MPS for Apple Silicon Macs
  - CPU fallback for unsupported systems
- Added video input support using OpenCV.
- Added output video generation in `.mp4` format.
- Implemented motion thresholding to hide static regions.
- Converted optical flow vectors into HSV-based color visualization.
- Added optional frame-by-frame PNG export.
- Made the input path more portable for Windows, macOS, and Linux.
- Added progress logging during video processing.

## Current Features

- Motion-based color visualization
- Static background removal using thresholding


## Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- RAFT optical flow model

## Future Improvement Scope

- Add command-line arguments instead of editing values inside the script.
- Add a simple GUI for selecting input videos and output folders.
- Support batch processing for multiple videos.
- Add real-time webcam optical flow visualization.
- Add side-by-side output showing original video and motion visualization together.
- Improve output video codec compatibility across different systems.
- Add support for saving optical flow data as `.npy` files for research use.
- Add adjustable color maps and visualization styles.
- Add object-level motion tracking on top of optical flow.
- Add performance optimizations for large videos.
- Add automatic resizing options for faster processing.
- Add better error handling for unsupported video formats.
- Add a `requirements.txt` file for easier installation.
- Add example input/output videos or screenshots to the repository.
- Add documentation explaining how RAFT optical flow works.
