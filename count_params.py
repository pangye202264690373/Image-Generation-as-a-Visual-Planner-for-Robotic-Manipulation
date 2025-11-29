# %% 参数统计脚本：统计整个 FluxPipeline 里所有参数（所有模块、所有权重）
import torch
from src.pipeline_pe_clone import FluxPipeline


def _get_dtype(name: str):
    name = (name or "").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


# ==== 修改成你自己的配置 ====
MODEL_PATH = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/root/PhotoDoodle/outputs/rt1_clean/checkpoint-58000/pytorch_lora_weights.safetensors"
DTYPE = "bfloat16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==== 收集模块 & 参数计数函数 ====
def get_unique_modules_from_pipeline(pipeline: FluxPipeline):
    """
    从 pipeline.components 收集所有唯一的 nn.Module，
    避免重复统计共享模块。
    """
    modules = []
    seen = set()

    # Diffusers 风格：pipeline.components 是一个 dict
    if hasattr(pipeline, "components"):
        for name, comp in pipeline.components.items():
            if isinstance(comp, torch.nn.Module):
                if id(comp) not in seen:
                    modules.append((name, comp))
                    seen.add(id(comp))
    else:
        # 兜底：如果没有 components，就尝试直接在属性里找 nn.Module
        for name in dir(pipeline):
            if name.startswith("_"):
                continue
            try:
                attr = getattr(pipeline, name)
            except Exception:
                continue
            if isinstance(attr, torch.nn.Module):
                if id(attr) not in seen:
                    modules.append((name, attr))
                    seen.add(id(attr))

    return modules


def count_params_in_modules(modules):
    total = 0
    trainable = 0
    for name, module in modules:
        for p in module.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
    return total, trainable


def count_params_by_module(modules):
    stats = {}
    for name, module in modules:
        t = 0
        tr = 0
        for p in module.parameters():
            n = p.numel()
            t += n
            if p.requires_grad:
                tr += n
        stats[name] = {"total": t, "trainable": tr}
    return stats


def count_lora_params_in_modules(modules):
    lora_total = 0
    for name, module in modules:
        for n, p in module.named_parameters():
            full_name = f"{name}.{n}"
            if "lora" in full_name.lower():
                lora_total += p.numel()
    return lora_total


# ==== pipeline 加载 ====
print("\n[INFO] Loading pipeline...")
pipe = FluxPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=_get_dtype(DTYPE),
    local_files_only=True
).to(DEVICE)

print(f"[INFO] Loading LoRA: {LORA_PATH}")
pipe.load_lora_weights(LORA_PATH)

# ==== 收集所有模块 ====
modules = get_unique_modules_from_pipeline(pipe)

print("\n[INFO] Found the following nn.Modules in pipeline:")
for name, _ in modules:
    print(f"  - {name}")

# ==== 全模型参数 ====
total, trainable = count_params_in_modules(modules)

print("\n==============================")
print("📦 Total Parameters in Pipeline")
print("==============================")
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}")
print(f"Non-trainable    : {total - trainable:,}")

# 根据 dtype 估算显存占用
bytes_per_param = 2 if DTYPE in ["bfloat16", "float16", "fp16"] else 4
gb = total * bytes_per_param / 1024**3
print(f"\nEstimated model size in {DTYPE}: {gb:.2f} GB\n")


# ==== 每个子模块参数 ====
print("===================================")
print("📌 Parameters by Submodule (Top-level components)")
print("===================================\n")

stats = count_params_by_module(modules)
for name, s in stats.items():
    t, tr = s["total"], s["trainable"]
    print(f"{name:30s} | total: {t:12,} | trainable: {tr:12,}")


# ==== LoRA 参数单独统计 ====
lora_params = count_lora_params_in_modules(modules)
print("\n==============================")
print("🎯 LoRA Parameters Count")
print("==============================")
print(f"LoRA params: {lora_params:,}")
if total > 0:
    print(f"LoRA ratio: {lora_params/total*100:.4f}% of total params\n")
else:
    print()
