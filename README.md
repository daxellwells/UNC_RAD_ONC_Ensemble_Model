## Auto3DSeg Inference (Windows)

### What you need (machine-dependent)
- NVIDIA **driver** installed (so `nvidia-smi` works)
- **CUDA-enabled PyTorch** (choose the right `cuXXX` wheel for your machine/driver)
- **Python 3.11** (recommended: install via the Windows Python Launcher `py`)

### Step-by-step setup (PowerShell)

#### 1) Create and activate a venv

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip setuptools wheel
```

#### 2) Install PyTorch (GPU / CUDA)
PyTorch wheels bundle the CUDA runtime on Windows. Use the official selector:
`https://pytorch.org/get-started/locally/`

Example (CUDA 12.4):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 3) Install the remaining dependencies
`requirements_windows.txt` intentionally **does not** include `torch*`.

```powershell
pip install -r .\requirements_windows.txt
```

#### 4) Verify CUDA is visible to PyTorch

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() > 0"
```

### Run inference

#### 1) Put inputs here
- `infer_job\input\*.nii` or `infer_job\input\*.nii.gz`

#### 2) Run

```powershell
python .\inference_windows.py --n-best 1
```

#### 3) Get outputs here
- `infer_job\output\`

### CLI arguments (inference_windows.py)
- **`--work-dir`**: Auto3DSeg exported work directory. **Default**: `.\work_dir`
- **`--input-cfg`**: Job config YAML (expects an `infer_job` folder next to it). **Default**: `.\infer_job\input.yaml`
- **`--n-best`**: Number of top models to ensemble. **Default**: `5`
- **`--num-fold`**: Number of folds in the exported work_dir. **Default**: `5`
- **`--single-gpu`**: Force single-GPU mode. **Default**: off
- **`--gpu-id`**: GPU index to use when `--single-gpu` is set. **Default**: `0`

### Recommendation
- Leave everything at defaults except **`--n-best`**.
- **`--n-best`** trades off quality vs speed:
  - Smaller `--n-best` → faster inference
  - Larger `--n-best` → better ensemble (slower)
- For **best inference quality**, run all models: **`--n-best 10`**.
