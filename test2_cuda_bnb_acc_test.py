# test_bnb_accelerate.py
import torch

print("==== PyTorch CUDA Check ====")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA device detected.")

print("\n==== bitsandbytes Check ====")
try:
    import bitsandbytes as bnb
    print("bitsandbytes version:", bnb.__version__)
    if hasattr(bnb, "__cuda_version__"):
        print("bnb CUDA version:", bnb.__cuda_version__)
    else:
        print("bnb reports CPU-only build.")
except Exception as e:
    print("bitsandbytes import failed:", e)

print("\n==== Accelerate Check ====")
try:
    from accelerate import Accelerator
    accelerator = Accelerator()
    print("Accelerate device:", accelerator.device)
    print("Mixed precision:", accelerator.mixed_precision)
except Exception as e:
    print("Accelerate import failed:", e)
