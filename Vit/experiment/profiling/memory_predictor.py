# ファイル名: memory_predictor.py

from __future__ import annotations
from typing import List, Tuple, Dict

# ディレクトリ構造に合わせてインポートパスを修正 (../../models/vit_stages)
# ただし、実行ディレクトリからの相対パスになるため、システムによっては調整が必要
import sys
import torch
import torch.nn as nn

_GiB = 1024 ** 3

def build_pred_max(
    model: nn.Module | None = None,
    # --- ViT 固有の引数 ---
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    num_classes: int = 1000,
    # --- 共通の引数 ---
    num_blocks_per_stage: int = 3,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    stage_id: int = 0,
    num_stages: int = 8,
    global_batch_size: int = 256,
    microbatch_size: int = 64,
    is_rev: bool = False,
    is_pareprop: bool = False,
    is_ckpt: bool = False,
    dtype: torch.dtype = torch.float32,
) -> float:
    """ViT 用: ステージの最大メモリ使用量を予測"""
    
    dbyte = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    input_cache_bytes = 0
    
    # シーケンス長計算
    num_patches = (img_size // patch_size) ** 2
    sequence_length = num_patches + 1

    is_std = not (is_ckpt or is_rev)

    is_first     = (stage_id == 0)
    is_last     = (stage_id == num_stages - 1)
    is_middle     = not (is_first or is_last)
    
    # --- パラメータ計算 (ViT用に修正) ---
    param_bytes = 0
    if model is not None:
         param_bytes = sum(p.numel() * p.element_size()
                           for p in model.parameters() if p.requires_grad)
    else:
         # 概算モード
         # ViT Block (SelfAttention + MLP)
         # LayerNorm x 2 (4*hidden) + Attn (4*hidden^2 + 4*hidden) + MLP (8*hidden^2 + 8*hidden)
         # ざっくり hidden^2 * 12 + hidden * 16 くらい
         params_per_block_approx = (hidden_size * hidden_size * 12 + hidden_size * 16) * dbyte 
         param_bytes = params_per_block_approx * num_blocks_per_stage
         
         if is_first:
             # Patch Embedding (Conv2d: out*in*k*k + out)
             patch_embed_params = (hidden_size * in_chans * patch_size * patch_size + hidden_size)
             # Positional Embedding (1 * (patches+1) * hidden)
             pos_embed_params = sequence_length * hidden_size
             # CLS Token (1 * 1 * hidden)
             cls_token_params = hidden_size
             
             embed_bytes = (patch_embed_params + pos_embed_params + cls_token_params) * dbyte
             param_bytes += embed_bytes
             
         if is_last:
             # Head (LayerNorm + Linear)
             norm_params = hidden_size * 2
             head_params = hidden_size * num_classes + num_classes
             head_bytes = (norm_params + head_params) * dbyte
             param_bytes += head_bytes
    
    # (1) 静的メモリ
    gradient_bytes = param_bytes
    optim_bytes = 2 * param_bytes 

    # (2) 入力データ (グローバルバッチサイズ) - First ステージのみ
    # 画像データ (float32固定と仮定、またはdbyte)
    # ここでは入力画像もモデルと同じ精度にキャストされると仮定して dbyte を使用
    input_image_bytes = global_batch_size * in_chans * img_size * img_size * dbyte
    input_data_bytes = input_image_bytes if is_first else 0

    # (3) 活性化 (1マイクロバッチ分)
    # (microbatch_size, seq_len, hidden_size) * 2 tensors * bytes_per_element
    activation_bytes_per_block_mb = microbatch_size * sequence_length * hidden_size * 2 * dbyte
    
    num_concurrent_microbatches_peak = num_stages - stage_id

    if is_std:
        total_activation_bytes = activation_bytes_per_block_mb * num_blocks_per_stage * num_concurrent_microbatches_peak
    elif is_ckpt:
        kept_activation = activation_bytes_per_block_mb * num_concurrent_microbatches_peak
        recompute_temp = activation_bytes_per_block_mb * 2 
        total_activation_bytes = kept_activation + recompute_temp
    elif is_rev:
        if is_pareprop:
            recompute_temp = activation_bytes_per_block_mb * 2.0 
        else: 
            recompute_temp = activation_bytes_per_block_mb 
        total_activation_bytes = recompute_temp
        
        if is_first:
             input_cache_bytes = input_data_bytes 
        else:
             input_cache_bytes = 0
    else:
        total_activation_bytes = 0
        input_cache_bytes = 0


    # (4) 送受信バッファ (ピーク時)
    buffer_bytes_per_mb = microbatch_size * sequence_length * hidden_size * 2 * dbyte
    
    fwd_recv_buffers = 0
    if not is_first:
        if is_rev:
             fwd_recv_buffers = buffer_bytes_per_mb * (2 if is_last else 1)
        else: 
             fwd_recv_buffers = buffer_bytes_per_mb * num_concurrent_microbatches_peak

    bwd_grad_recv_buffers = 0
    if not is_last:
        bwd_grad_recv_buffers = buffer_bytes_per_mb * (2 if is_last else 1)

    bwd_ract_recv_buffers = 0
    if is_middle and is_rev:
        bwd_ract_recv_buffers = buffer_bytes_per_mb

    total_buffer_bytes = fwd_recv_buffers + bwd_grad_recv_buffers + bwd_ract_recv_buffers

    # --- 最大メモリ使用量の合計 ---
    res = 0
    res += param_bytes      
    res += optim_bytes      
    res += gradient_bytes   
    res += input_data_bytes 
    res += total_buffer_bytes 
    res += total_activation_bytes 
    if is_rev and is_first:
        res += input_cache_bytes 

    return res / _GiB