# calculate_mse.py
import torch
import torch.nn.functional as F

@torch.inference_mode()
def calculate_mse(
    videos1: torch.Tensor,
    videos2: torch.Tensor,
    only_final: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.float32,
    show_progress: bool = True,
):
    """
    逐帧/常数显存 的 MSE 计算（与 common_metrics_on_video_quality 风格对齐）
    输入:
        videos1, videos2: [B, T, C, H, W], 像素建议 ∈ [0,1]
        only_final:
          - True  -> 返回单个整体值（长度1的 list）+ batch 内 std
          - False -> 返回前缀曲线（长度 T 的 list）+ std
        device:  计算时使用的设备；None 表示沿用输入所在设备
        dtype:   计算 dtype；None 则沿用输入 dtype（推荐 float32）
        show_progress: 是否显示 tqdm 进度
    返回:
        {"mse": {"value": [...], "value_std": [...]}}

    设计要点：
      - 逐视频逐帧算：每次只搬一个 [C,H,W] 到 device，显存/内存峰值极低
      - 不构建计算图（inference_mode），运算后立刻转标量存放到 CPU
      - per-frame 的 MSE 定义为 mean((x-y)^2) over C,H,W
      - 整段整体 MSE = per-frame MSE 的时间平均（与全像素平均一致）
    """
    if videos1.ndim != 5 or videos2.ndim != 5:
        raise ValueError(f"Expected 5D tensors [B,T,C,H,W], got {videos1.shape=} and {videos2.shape=}")
    if videos1.shape != videos2.shape:
        raise ValueError(f"Shape mismatch: {videos1.shape=} vs {videos2.shape=}")

    B, T, C, H, W = videos1.shape
    dev1, dev2 = videos1.device, videos2.device
    if dev1 != dev2:
        raise ValueError(f"videos1 and videos2 must be on the same device, got {dev1=} vs {dev2=}")

    # 计算设备与 dtype 统一
    if device is None:
        device = dev1
    else:
        device = torch.device(device)
    use_dtype = videos1.dtype if dtype is None else dtype

    print("calculate_mse...")

    # 准备存放每帧 MSE 的容器（放 CPU，只有 B×T 个 float，极小）
    per_frame_mse_cpu = torch.empty((B, T), dtype=torch.float32, device="cpu")

    # 进度条
    rng_B = range(B)
    if show_progress:
        try:
            from tqdm import tqdm
            rng_B = tqdm(rng_B, desc="MSE (per video)")
        except Exception:
            pass

    # 逐视频 × 逐帧
    for b in rng_B:
        # 为了减少跨设备传输次数，仅搬当前帧到目标 device
        # 每帧计算：
        #   mse_t = mean((x - y)^2) over (C,H,W)
        for t in range(T):
            x = videos1[b, t].to(device=device, dtype=use_dtype, non_blocking=True)  # [C,H,W]
            y = videos2[b, t].to(device=device, dtype=use_dtype, non_blocking=True)  # [C,H,W]
            # F.mse_loss 默认 reduction='mean'
            mse_val = F.mse_loss(x, y, reduction='mean').item()
            per_frame_mse_cpu[b, t] = mse_val
            # 立刻释放当帧张量引用（让 CUDA allocator 能复用小缓存）
            del x, y

    # —— 统计与返回 —— #
    # only_final=True: 整段整体 MSE（每个样本对 T 做均值，再对 batch 求均值与 std）
    if only_final:
        per_sample = per_frame_mse_cpu.mean(dim=1)  # [B]
        value = [float(per_sample.mean().item())]
        std   = [float(per_sample.std(unbiased=False).item())]
        return {"mse": {"value": value, "value_std": std}}

    # only_final=False: 前缀曲线（n=1..T）— 先做时间前缀均值，再对 batch 求均值/方差
    # [B,T] -> cumsum -> prefix_mean: [B,T]
    cumsum = per_frame_mse_cpu.cumsum(dim=1)
    denom  = torch.arange(1, T+1, dtype=per_frame_mse_cpu.dtype).unsqueeze(0)  # [1,T]
    prefix_mean = cumsum / denom

    value_seq = prefix_mean.mean(dim=0).tolist()                   # 长度 T
    std_seq   = prefix_mean.std (dim=0, unbiased=False).tolist()   # 长度 T
    return {"value": value_seq, "value_std": std_seq}
