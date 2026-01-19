#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from infer_utils import generate_datalist, copy_outputs, self_fix

def _require_cuda() -> None:
    """
    Fail fast with a clear message if CUDA isn't available.
    Auto3DSeg SwinUNETR bundles call torch.cuda.mem_get_info() during pre_operation(),
    which crashes cryptically when no CUDA device is visible.
    """
    try:
        import torch
    except Exception as e:
        raise SystemExit(
            "[ERROR] PyTorch is not installed in this environment.\n"
            "Install a CUDA-enabled PyTorch wheel (torch/torchvision/torchaudio) and retry.\n"
            f"Details: {e}"
        )

    ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
    if ok:
        return

    raise SystemExit(
        "[ERROR] CUDA is not available to PyTorch (torch.cuda.is_available() == False or device_count == 0).\n"
        "This inference pipeline is GPU-only.\n\n"
        "Fix:\n"
        "  1) Ensure an NVIDIA driver is installed and working (run: nvidia-smi)\n"
        "  2) Install a CUDA-enabled PyTorch wheel (NOT cpu-only). Example (CUDA 12.1):\n"
        "       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
        "  3) Re-run:\n"
        "       python .\\inference_windows.py --n-best 1\n"
    )

def force_single_gpu(gpu_id=0):
    """
    Hard-disable distributed execution and force a single visible GPU.
    Must be called BEFORE importing torch or monai.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Kill any distributed env that torchrun / SLURM may inject
    for k in [
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
    ]:
        os.environ.pop(k, None)

    # Extra safety
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"


def main():
    parser = argparse.ArgumentParser("Auto3DSeg Ensemble Runner")
    parser.add_argument("--work-dir", default="./work_dir")
    parser.add_argument("--input-cfg", default="./infer_job/input.yaml")
    parser.add_argument("--n-best", type=int, default=5)
    parser.add_argument("--num-fold", type=int, default=5)
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU mode")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id to use in single-GPU mode")

    args = parser.parse_args()

    if args.single_gpu:
        print(f"[INFO] Forcing single-GPU mode on GPU {args.gpu_id}")
        force_single_gpu(args.gpu_id)

    # Make sure CUDA is available before MONAI bundles attempt GPU auto-scaling.
    _require_cuda()

    input_cfg_path = Path(args.input_cfg).resolve()
    job_dir = input_cfg_path.parent
    datalist_path = generate_datalist(job_dir)
    work_dir = Path(args.work_dir).resolve()
    print(f"[INFO] Wrote datalist: {datalist_path}")

    self_fix(
        work_dir,
        datalist_path=datalist_path,
        base_dir=job_dir,
        patch_scripts=True,
        patch_pkls=True,
        dry_run=False,
        verbose=True,
        backup=True,
    )

    # 🚨 imports AFTER env is sanitized
    from monai.apps.auto3dseg.ensemble_builder import EnsembleRunner

    ensemble_runner = EnsembleRunner(
        data_src_cfg_name=str(input_cfg_path),
        work_dir=str(work_dir),
        num_fold=args.num_fold,
        ensemble_method_name="AlgoEnsembleBestN",
        mgpu=False,               # critical
        n_best=args.n_best,
        mode="mean",
        sigmoid=False,
    )

    ensemble_runner.run()

    print("[INFO] Ensemble predictions saved under:", work_dir / "ensemble_output")
    copy_outputs(work_dir, job_dir)


if __name__ == "__main__":
    main()