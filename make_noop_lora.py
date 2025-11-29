# make_noop_lora.py
import os
import argparse
import torch
from transformers import CLIPTokenizer, T5TokenizerFast, PretrainedConfig
from diffusers import AutoencoderKL, FluxTransformer2DModel
from src.pipeline_pe_clone import FluxPipeline   # 你现有脚本里的
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

def add_noop_lora(transformer, target_modules, r=128, alpha=128):
    transformer.add_adapter(LoraConfig(r=r, lora_alpha=alpha, init_lora_weights="gaussian",
                                       target_modules=target_modules))
    # 置零 + 冻结
    for n, p in transformer.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n) or ("lora_down" in n) or ("lora_up" in n):
            with torch.no_grad():
                p.zero_()
            p.requires_grad_(False)
    if hasattr(transformer, "set_adapter_scale"):
        try: transformer.set_adapter_scale(0.0)
        except Exception: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rank", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )
    # 和你训练脚本里一致的默认 LoRA 目标层
    target_modules = [
        "attn.to_k","attn.to_q","attn.to_v","attn.to_out.0",
        "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
        "ff.net.0.proj","ff.net.2","ff_context.net.0.proj","ff_context.net.2",
    ]
    add_noop_lora(transformer, target_modules, r=args.rank, alpha=args.rank)

    # 保存（与现有加载流程最兼容的通用 safetensors 形式）
    sd = get_peft_model_state_dict(transformer)  # 全 0
    save_file(sd, os.path.join(args.out_dir, "pytorch_lora_weights.safetensors"))
    print("✅ No-op LoRA 已生成：", os.path.join(args.out_dir, "pytorch_lora_weights.safetensors"))

if __name__ == "__main__":
    main()

# python make_noop_lora.py \
#   --pretrained_model_name_or_path ./PhotoDoodle_Pretrain \
#   --out_dir outputs/noop_lora