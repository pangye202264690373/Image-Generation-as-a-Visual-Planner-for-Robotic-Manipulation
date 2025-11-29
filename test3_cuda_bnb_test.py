import bitsandbytes as bnb
from bitsandbytes.cuda_specs import CUDASpecs, get_cuda_specs
from bitsandbytes.cextension import get_cuda_bnb_library_path
cuda_specs = get_cuda_specs()
print("CUDA specs:", cuda_specs)
if cuda_specs:
    print("Expected CUDA .so path:", get_cuda_bnb_library_path(cuda_specs))
