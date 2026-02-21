# ファイル名: memory_predictor_prev.py (★ is_pareprop 分岐を追加)

# ファイル名: memory_predictor_prev.py

from __future__ import annotations
from typing import List, Tuple, Dict
from .events import EventSpan
from src.config import GlobalConfig, ParMode
from src.env import EnvContext
from src.dataset_registry import DatasetSpec
import torch

_GiB = 1024 ** 3

def build_pred_timeline(
    spans: List[EventSpan],
    cfg: GlobalConfig,
    env: EnvContext,
    d_spec: DatasetSpec,
    model: torch.nn.Module,
) -> Tuple[List[float], Dict[str, List[float]]]:
    
    hidden_size = cfg.hidden_size
    num_blocks = cfg.num_hidden_layers
    num_classes = d_spec.num_classes
    dbyte = torch.finfo(cfg.autocast_dtype).bits // 8

    # --- ViT固有の計算 ---
    # シーケンス長 = (画像サイズ // パッチサイズ)^2 + 1(CLSトークン)
    num_patches = (cfg.img_size // cfg.patch_size) ** 2
    seq_len = num_patches + 1 
    
    # 活性化サイズ (Full Batch)
    activation_bytes_per_layer = cfg.batch_size * seq_len * hidden_size * 2 * dbyte

    stage_id = env.rank
    num_stages = env.world_size
    if cfg.par_mode == ParMode.PP:
        num_blocks_per_stage = num_blocks // num_stages
    else:
        num_blocks_per_stage = num_blocks

    is_first = (stage_id == 0)
    is_last = (stage_id == num_stages - 1)
    is_middle = not (is_first or is_last)

    is_ckpt = cfg.checkpointing
    is_rev = cfg.reversible
    is_pareprop = cfg.is_pareprop if is_rev else False
    is_std = not (is_ckpt or is_rev)

    cats = [
        "PARAMETER", "OPTIMIZER_STATE", "INPUT", "ACTIVATION", 
        "GRADIENT", "AUTOGRAD_DETAIL", "TEMPORARY", "UNKNOWN",
    ]
    cur: Dict[str, float] = {c: 0.0 for c in cats}

    param_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters() if p.requires_grad)
    optim_bytes = 2 * param_bytes
    gradient_bytes = param_bytes

    # --- 入力データサイズ (画像) ---
    # (Batch, Channels, Height, Width) * float32(4byte) or autocast_dtype
    # 通常 DataLoader は float32 で画像を返すことが多いが、autocast有効ならモデル入力直前でキャストされるかも
    # ここでは安全側に倒して float32 (4byte) で計算するか、dbyteを使うか。
    # 画像データは通常 float32 なので 4byte 固定にしておきます。
    input_image_bytes = cfg.batch_size * cfg.in_chans * cfg.img_size * cfg.img_size * 4 
    input_data_bytes = input_image_bytes

    # 活性化サイズ (Micro Batch)
    activation_bytes_per_block_mb = cfg.microbatch_size * seq_len * hidden_size * 2 * dbyte
    
    if is_std:
        chunk_actv_bytes = activation_bytes_per_block_mb * num_blocks_per_stage
    elif is_ckpt:
        chunk_actv_bytes = activation_bytes_per_block_mb
    elif is_rev:
        chunk_actv_bytes = 0
    else:
        chunk_actv_bytes = 0

    # バッファサイズ (中間層の入出力 = (Batch, SeqLen, Hidden))
    buffer_bytes_per_mb = cfg.microbatch_size * seq_len * hidden_size * 2 * dbyte
    static_buffer_bytes = 0
    if not is_first:
        if is_rev:
             static_buffer_bytes += buffer_bytes_per_mb
        else: 
             static_buffer_bytes += buffer_bytes_per_mb * (num_stages - stage_id)
    if not is_last:
        static_buffer_bytes += buffer_bytes_per_mb * (2 if is_last else 1) 
    if is_middle and is_rev:
        static_buffer_bytes += buffer_bytes_per_mb
        
    # 初期状態セット
    cur["PARAMETER"]       = param_bytes 
    cur["OPTIMIZER_STATE"] = optim_bytes
    cur["GRADIENT"]        = 0.0 
    cur["AUTOGRAD_DETAIL"] = 0.0 
    cur["ACTIVATION"]      = 0.0 
    cur["INPUT"]           = 0.0 
    if is_first: 
        cur["INPUT"] += input_data_bytes
    cur["INPUT"] += static_buffer_bytes
    cur["TEMPORARY"]       = 0.0 
    cur["UNKNOWN"]         = 0.0

    ts:      List[float]            = []
    traces:  Dict[str, List[float]] = {c: [] for c in cats}

    def _push(t):
        ts.append(t)
        for k in cats:
            traces[k].append(cur[k] / _GiB)

    _push(spans[0].t0_ms)

    is_gradient_assigned = False
    forward_chunk_id = 0
    backward_chunk_id = 0
    is_input_cache_allocated = False

    for ev in spans:

        if ev.label == "forward_one_chunk":
            _push(ev.t0_ms) 
            if is_std or is_ckpt:
                 cur["ACTIVATION"] += chunk_actv_bytes 
            if is_rev and is_first and not is_input_cache_allocated:
                 # ReversibleかつFirstステージの場合、入力画像を入力キャッシュとして保持
                 cur["INPUT"] += input_data_bytes 
                 is_input_cache_allocated = True
            _push(ev.t1_ms) 
            forward_chunk_id += 1

        elif ev.label == "backward_one_chunk":
            _push(ev.t0_ms) 

            # (1) Gradient buffer
            if not is_gradient_assigned:
                cur["GRADIENT"] += gradient_bytes
                is_gradient_assigned = True

            # (2) Activation recomputation
            temp_activation_for_recompute = 0
            if is_ckpt and not is_first: 
                temp_activation_for_recompute = activation_bytes_per_block_mb * 2
                cur["ACTIVATION"] += temp_activation_for_recompute
                
            elif is_rev: 
                if is_pareprop:
                     temp_activation_for_recompute = activation_bytes_per_block_mb * 1.5 
                else: 
                     temp_activation_for_recompute = activation_bytes_per_block_mb 
                
                if is_first or is_middle or is_last:
                    cur["ACTIVATION"] += temp_activation_for_recompute

            # (3) Input Cache release
            if is_rev and is_first:
                 if backward_chunk_id == cfg.num_microbatches - 1 and is_input_cache_allocated:
                      cur["INPUT"] -= input_data_bytes
                      is_input_cache_allocated = False 

            _push(ev.t1_ms) 

            # (4) Release recomputation memory
            if temp_activation_for_recompute > 0:
                 cur["ACTIVATION"] -= temp_activation_for_recompute

            # (5) Release saved activations (std)
            if is_std:
                 cur["ACTIVATION"] -= chunk_actv_bytes

            backward_chunk_id += 1

        elif ev.label == "optimize":
            _push(ev.t0_ms)
            _push(ev.t1_ms) 
            if is_gradient_assigned:
                cur["GRADIENT"] -= gradient_bytes
                is_gradient_assigned = False
            forward_chunk_id = 0
            backward_chunk_id = 0
            is_input_cache_allocated = False

    return ts, traces