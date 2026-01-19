# README_CONDA.md
# =================
# Auto3DSeg Inference — Conda Environment (GPU-Only)
#
# This document describes how to set up and run the conda-based environment
# for the Auto3DSeg inference pipeline.
#
# IMPORTANT:
# - CPU-only execution is NOT supported
# - CUDA + NVIDIA GPU are REQUIRED
# - PyTorch is installed separately from the conda environment
# - This mirrors the behavior of the pip/venv setup
#
# Machine-dependent requirements (Windows):
# - NVIDIA driver must be installed
# - PyTorch wheel must match the machine (choose the right cuXXX build)

# ----------------------------------------------------------------------
# 1. Prerequisites (Required)
# ----------------------------------------------------------------------

# NVIDIA GPU + Driver
# -------------------
# - An NVIDIA GPU compatible with CUDA is required
# - NVIDIA driver must be new enough for the chosen CUDA version
#   Example: CUDA 12.1 → driver ≥ 530.xx
#
# You do NOT need to install the CUDA Toolkit system-wide.

# Conda
# -----
# Any of the following are acceptable:
# - Anaconda
# - Miniconda
# - Mambaforge
#
# Verify conda is installed:
#   conda --version

# ----------------------------------------------------------------------
# 2. Environment Creation (Torch NOT included)
# ----------------------------------------------------------------------

# The conda environment installs all dependencies EXCEPT PyTorch.
# This is intentional.

# NOTE (Windows):
# This folder does not ship an environment.yml. Use a simple conda env + pip installs:
#
#   conda create -n auto3dseg_infer python=3.11 -y
#   conda activate auto3dseg_infer
#   python -m pip install --upgrade pip setuptools wheel

# ----------------------------------------------------------------------
# 3. Install PyTorch (GPU REQUIRED)
# ----------------------------------------------------------------------

# PyTorch must be installed manually AFTER activating the environment.

# Example: CUDA 12.1 wheel (PowerShell):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the remaining deps for this repo (Windows):
pip install -r .\requirements_windows.txt

# Other CUDA versions (e.g. cu118) may be used if required,
# but MUST match the installed NVIDIA driver.

# ----------------------------------------------------------------------
# 4. Verify CUDA Is Available (MANDATORY)
# ----------------------------------------------------------------------

python -c "import torch; print('Torch:', torch.__version__); print('CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available()"

# If this fails:
# - NVIDIA driver is missing or incompatible
# - Incorrect PyTorch CUDA wheel installed

# ----------------------------------------------------------------------
# 5. Running Inference
# ----------------------------------------------------------------------

python .\inference_windows.py --n-best 1

# Use the same CLI arguments as the pip-based version.

# ----------------------------------------------------------------------
# 6. CUDA vs PyTorch vs Drivers
# ----------------------------------------------------------------------

# This setup uses PyTorch wheels with a bundled CUDA runtime.
#
# - No system CUDA Toolkit required
# - CUDA libraries live inside the Python environment
# - NVIDIA DRIVER must still be installed on the system

# ----------------------------------------------------------------------
# 7. Intentional Limitations
# ----------------------------------------------------------------------

# - CPU-only execution is NOT supported
# - No automatic CUDA fallback
# - No hardcoded paths (everything is relative or argument-driven)
#
# These constraints are intentional for consistency and performance.

# ----------------------------------------------------------------------
# 8. Troubleshooting
# ----------------------------------------------------------------------

# CUDA available == False:
# - Run: nvidia-smi
# - Check driver version
# - Verify correct cuXXX PyTorch wheel

# Conda solver issues:
# - Prefer: conda env update -f environment.yml
# - Avoid mixing additional pip-installed scientific libraries

# ----------------------------------------------------------------------
# 9. Summary
# ----------------------------------------------------------------------
#
# Component        Installed By
# --------------------------------
# Python + deps    Conda
# PyTorch          Manual (pip)
# CUDA runtime     Bundled with PyTorch
# NVIDIA driver    System
#
# ----------------------------------------------------------------------
# End of README_CONDA.md
# ----------------------------------------------------------------------
