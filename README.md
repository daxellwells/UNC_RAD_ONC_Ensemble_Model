## PyTorch installation (required)

# This project intentionally does NOT pin PyTorch in requirements.txt.
# You MUST install PyTorch separately with CUDA support.
# CPU-only PyTorch is NOT supported.
#
# Machine-dependent requirements (Windows):
# - NVIDIA driver must be installed
# - PyTorch wheel must match the machine (choose the right cuXXX build)

# ------------------------------------------------------------
# GPU (required)
# ------------------------------------------------------------

# Install PyTorch with CUDA support.
# On Windows, PyTorch wheels bundle the CUDA runtime, so you generally pick a cuXXX wheel
# based on NVIDIA driver compatibility (not a separately installed CUDA Toolkit).
# Use the official selector for the correct command:
#   https://pytorch.org/get-started/locally/

# Example for CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# Install remaining dependencies (Windows)
# ------------------------------------------------------------
#
# From this folder (PowerShell):
#
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   # Recommended: Python 3.11 (or 3.12). Avoid 3.13+ unless you know all wheels exist.
#   python -m pip install --upgrade pip setuptools wheel
#   pip install -r .\requirements_windows.txt

# ------------------------------------------------------------
# CPU-only (NOT supported)
# ------------------------------------------------------------

# WARNING:
# CPU-only execution is NOT supported.
# SwinUNETR and Auto3DSeg ensemble inference require CUDA
# and will fail in CPU-only mode.

# ------------------------------------------------------------
# Verify CUDA availability
# ------------------------------------------------------------

python -c "import torch; print('Torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() > 0"

# ------------------------------------------------------------
# Run inference (Windows)
# ------------------------------------------------------------
#
# 1) Put NIfTIs into:
#    infer_job\input\*.nii or *.nii.gz
#
# 2) Run:
#    python .\inference_windows.py --n-best 1
#
# Outputs are copied to:
#   infer_job\output\
