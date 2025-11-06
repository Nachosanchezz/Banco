# Quick GPU and configuration smoke test for PyTorch.
# Purpose:
# - Detects available GPUs and verifies that PyTorch can access them.
#
# Notes:
# - This script is meant as a lightweight environment sanity check.
# - Useful before running full training to confirm GPU.

import torch

print("\n=== PyTorch / CUDA Detection ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

cuda_runtime = getattr(torch.version, "cuda", None)
try:
    cudnn_version = torch.backends.cudnn.version()
except Exception:
    cudnn_version = None
print(f"CUDA runtime:    {cuda_runtime}")
print(f"cuDNN version:   {cudnn_version}")

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    print(f"[OK] {count} CUDA device(s) detected:")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        cc_major, cc_minor = torch.cuda.get_device_capability(i)
        print(f"  - GPU {i}: {name} (Compute Capability {cc_major}.{cc_minor})")
    print()

    # Simple GPU operation (matrix multiply) to validate compute path
    device = torch.device("cuda:0")
    a = torch.randn((512, 512), device=device)
    b = torch.randn((512, 512), device=device)
    c = a @ b  # GEMM on GPU
    torch.cuda.synchronize()
    print(f"[OK] Matmul on {device} completed (mean={c.mean().item():.6f})\n")
else:
    print("[WARN] No CUDA GPU detected. Running on CPU.\n")