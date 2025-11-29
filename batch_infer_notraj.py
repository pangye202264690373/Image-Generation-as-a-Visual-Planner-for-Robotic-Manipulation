# %%
import json
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from tqdm import tqdm
import torch

# %% [markdown]
# ## jocoplay

# %%
# # ==== 直接在这里指定配置 ====
# CONFIG = {
#     "model_path": "black-forest-labs/FLUX.1-dev",         # 本地 FLUX 基础模型目录
#     "lora_path": "/root/PhotoDoodle/outputs/jocoplay/checkpoint-50000/pytorch_lora_weights.safetensors", 
#     # ↑ 直接换成你要使用的本地 LoRA 权重，例如:
#     # "/home/u12027/PhotoDoodle_LoRA/sksmagiceffects.safetensors"
    
#     "jsonl_path": "/root/PhotoDoodle/data/joco_merge_clean_fulltext_test/text_aug_rand.jsonl",                # JSONL 输入文件（每行: {"text": "...", "source": "/path/to/image"}）
#     "output_dir": "inference/outputs_joco_text_2",                    # 输出目录
#     "height": 672,
#     "width": 672,
#     "guidance_scale": 3.5,
#     "num_steps": 20,
#     "max_sequence_length": 512,
#     "dtype": "bfloat16",                              # "bfloat16" / "float16" / "float32"
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "seed": None                                      # 固定随机种子(如 1234)，None 则随机
# }

# %%
# # ==== 直接在这里指定配置 ====
# CONFIG = {
#     "model_path": "black-forest-labs/FLUX.1-dev",         # 本地 FLUX 基础模型目录
#     "lora_path": "/root/PhotoDoodle/outputs/jocoplay_traj/checkpoint-50000/pytorch_lora_weights.safetensors", 
#     # ↑ 直接换成你要使用的本地 LoRA 权重，例如:
#     # "/home/u12027/PhotoDoodle_LoRA/sksmagiceffects.safetensors"
    
#     "jsonl_path": "/root/PhotoDoodle/data/joco_merge_traj_test/pairs_text_inpaint.jsonl",                # JSONL 输入文件（每行: {"text": "...", "source": "/path/to/image"}）
#     "output_dir": "inference/outputs_joco_traj_2",                    # 输出目录
#     "height": 672,
#     "width": 672,
#     "guidance_scale": 3.5,
#     "num_steps": 20,
#     "max_sequence_length": 512,
#     "dtype": "bfloat16",                              # "bfloat16" / "float16" / "float32"
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "seed": None                                      # 固定随机种子(如 1234)，None 则随机
# }

# %% [markdown]
# ## bridge V2

# %%
# ==== 直接在这里指定配置 ====
CONFIG = {
    "model_path": "black-forest-labs/FLUX.1-dev",         # 本地 FLUX 基础模型目录
    "lora_path": "/root/PhotoDoodle/outputs/bridge_traj/checkpoint-104000/pytorch_lora_weights.safetensors", 
    # ↑ 直接换成你要使用的本地 LoRA 权重，例如:
    # "/home/u12027/PhotoDoodle_LoRA/sksmagiceffects.safetensors"
    
    "jsonl_path": "/root/PhotoDoodle/data/bridge_test/traj_test_index_noop.jsonl",                # JSONL 输入文件（每行: {"text": "...", "source": "/path/to/image"}）
    "output_dir": "inference/outputs_bridgeV2_traj_notTrajectory",                    # 输出目录
    "height": 720,
    "width":960,
    "guidance_scale": 3.5,
    "num_steps": 20,
    "max_sequence_length": 512,
    "dtype": "bfloat16",                              # "bfloat16" / "float16" / "float32"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": None                                      # 固定随机种子(如 1234)，None 则随机
}

# %% [markdown]
# ## rt1

# %%
# # ==== 直接在这里指定配置 ====
# CONFIG = {
#     "model_path": "black-forest-labs/FLUX.1-dev",         # 本地 FLUX 基础模型目录
#     "lora_path": "/root/PhotoDoodle/outputs/rt1_clean/checkpoint-58000/pytorch_lora_weights.safetensors", 
    
#     "jsonl_path": "/root/PhotoDoodle/data/rt1_merge_clean/pairs_test_single_augmented_index.jsonl",                # JSONL 输入文件（每行: {"text": "...", "source": "/path/to/image"}）
#     "output_dir": "inference/outputs_rt1_text",                    # 输出目录
#     "height": 768,
#     "width":960,
#     "guidance_scale": 3.5,
#     "num_steps": 20,
#     "max_sequence_length": 512,
#     "dtype": "bfloat16",                              # "bfloat16" / "float16" / "float32"
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "seed": None                                      # 固定随机种子(如 1234)，None 则随机
# }

# %%
# # ==== 直接在这里指定配置 ====
# CONFIG = {
#     "model_path": "black-forest-labs/FLUX.1-dev",         # 本地 FLUX 基础模型目录
#     "lora_path": "/root/PhotoDoodle/outputs/rt1_traj/checkpoint-58000/pytorch_lora_weights.safetensors", 

#     "jsonl_path": "/root/PhotoDoodle/data/rt1_merge_traj/pairs_test_traj_replaced.jsonl",                # JSONL 输入文件（每行: {"text": "...", "source": "/path/to/image"}）
#     "output_dir": "inference/outputs_rt1_traj",                    # 输出目录
#     "height": 768,
#     "width":960,
#     "guidance_scale": 3.5,
#     "num_steps": 20,
#     "max_sequence_length": 512,
#     "dtype": "bfloat16",                              # "bfloat16" / "float16" / "float32"
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "seed": None                                      # 固定随机种子(如 1234)，None 则随机
# }

# %%
# ==== 导入 Pipeline ====
from src.pipeline_pe_clone import FluxPipeline


def _get_dtype(name: str):
    n = (name or "").lower()
    if n in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if n in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


def load_pipeline(model_path: str, lora_path: str, dtype: str, device: str) -> FluxPipeline:
    """只加载本地模型和本地 LoRA"""
    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=_get_dtype(dtype),
        local_files_only=True     # 不联网
    ).to(device)

    # === 第 ① 步：加载并 fuse 预训练 LoRA ===
    print(f"[INFO] Loading pretrain LoRA")
    pipe.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors", local_files_only=True)
    pipe.fuse_lora()                 # 合并到 base
    pipe.unload_lora_weights()       # 卸载 adapter 节省显存

    # 直接加载指定 LoRA
    if lora_path:
        print(f"[INFO] Loading LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)

    return pipe


def run_batch_inference(
    pipeline: FluxPipeline,
    jsonl_path: str,
    output_dir: str,
    height: int,
    width: int,
    guidance_scale: float,
    num_steps: int,
    max_sequence_length: int,
    seed: Optional[int],
    device: str
):
    jsonl_path = Path(jsonl_path)
    base_dir = jsonl_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    with open(results_path, "w", encoding="utf-8") as fw:
        for idx, line in enumerate(tqdm(lines, desc="Batch Inference")):
            # 解析 JSON
            try:
                rec: Dict[str, Any] = json.loads(line)
            except Exception as e:
                fw.write(json.dumps({"idx": idx, "ok": False, "error": f"JSON error: {e}", "raw": line}, ensure_ascii=False) + "\n")
                continue

            text = (rec.get("text") or "").strip()
            src = rec.get("source")

            if not text or not src:
                fw.write(json.dumps({"idx": idx, "ok": False, "error": "Missing text/source", "record": rec}, ensure_ascii=False) + "\n")
                continue

            # === 关键：相对路径按 JSONL 目录解析，并保持使用解析后的 src_path ===
            raw_src_path = Path(src)
            src_path = raw_src_path if raw_src_path.is_absolute() else (base_dir / raw_src_path)
            src_path = src_path.resolve()

            if not src_path.exists():
                fw.write(json.dumps({
                    "idx": idx, "ok": False,
                    "error": f"Image not found",
                    "source_in_json": str(raw_src_path),
                    "resolved_source": str(src_path)
                }, ensure_ascii=False) + "\n")
                continue

            try:
                with Image.open(src_path) as im:
                    cond_img = im.convert("RGB").resize((width, height))

                out = pipeline(
                    prompt=text,
                    condition_image=cond_img,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    max_sequence_length=max_sequence_length,
                )
                img = out.images[0]

                sample_id = str(rec.get("id", src_path.stem))
                save_path = output_dir / f"{sample_id}_generated.png"

                img.save(save_path)

                relative_output = save_path.relative_to(output_dir)
                result_rec = {
                    **rec,
                    "idx": idx,
                    "ok": True,
                    "resolved_source": str(src_path),
                    "output": str(relative_output)
                }
                fw.write(json.dumps(result_rec, ensure_ascii=False) + "\n")

            except Exception as e:
                result_rec = {
                    **rec,
                    "idx": idx,
                    "ok": False,
                    "resolved_source": str(src_path),
                    "error": f"{type(e).__name__}: {str(e)}"
                }
                fw.write(json.dumps(result_rec, ensure_ascii=False) + "\n")



# %%
cfg = CONFIG
pipe = load_pipeline(cfg["model_path"], cfg["lora_path"], cfg["dtype"], cfg["device"])
run_batch_inference(
    pipeline=pipe,
    jsonl_path=cfg["jsonl_path"],
    output_dir=cfg["output_dir"],
    height=cfg["height"],
    width=cfg["width"],
    guidance_scale=cfg["guidance_scale"],
    num_steps=cfg["num_steps"],
    max_sequence_length=cfg["max_sequence_length"],
    seed=cfg["seed"],
    device=cfg["device"]
)


