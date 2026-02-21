from __future__ import annotations
from typing import List, Tuple, Dict
# from .events import EventSpan # このファイルでは不要

from models.bert_stages import (
    BertFirstStage,
    BertMidStage,
    BertLastStage,
)
import torch # ★ torch をインポート
import torch.nn as nn # ★ nn をインポート

_GiB = 1024 ** 3

def build_pred_max(
    model: nn.Module | None = None, # ★ 型ヒントを nn.Module に変更
    # --- BERT 固有の引数 ---
    num_blocks_per_stage: int = 3,
    hidden_size: int = 768,
    sequence_length: int = 512,
    num_attention_heads: int = 12, # (現状の予測ロジックでは未使用だが将来用に残す)
    vocab_size: int = 30522,      # (FirstStage の Embedding 計算用)
    # --- 共通の引数 ---
    stage_id: int = 0,
    num_stages: int = 8,
    global_batch_size: int = 256, # ★ 引数名変更
    microbatch_size: int = 64,
    is_rev: bool = False,
    is_pareprop: bool = False, # ★ 追加
    is_ckpt: bool = False,
    # --- 不要になった CNN 引数を削除 ---
    # num_classes: int = 1000,
    # num_channel: int = 3,
    # image_height: int = 224,
    # image_width: int = 224,
    # --- ★ データ型用の引数を追加 (任意だが精度向上) ---
    dtype: torch.dtype = torch.float32,
) -> float:
    """★ BERT 用: ステージの最大メモリ使用量を予測"""
    
    # データ型に応じたバイト数を計算
    dbyte = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    input_cache_bytes = 0
    # --- ★ モデルブロック数 (引数 model が渡されなかった場合のフォールバック) ---
    # num_blocks_per_stage = 3 # CNN用
    # これは cfg.num_hidden_layers / world_size で決まるので引数で渡すべきだが、
    # フォールバックとして適当な値 (e.g., 3) を設定しておく
    
    is_std = not (is_ckpt or is_rev)

    is_first     = (stage_id == 0)
    is_last     = (stage_id == num_stages - 1)
    is_middle     = not (is_first or is_last)
    
    # --- ★ モデルパラメータ計算 (BERT用に修正) ---
    param_bytes = 0
    if model is not None:
         # 渡されたモデルオブジェクトから計算
         param_bytes = sum(p.numel() * p.element_size()
                           for p in model.parameters() if p.requires_grad)
    else:
         # モデルがない場合は概算 (非常に不正確になる可能性あり)
         # print("Warning: model object not provided to build_pred_max, estimating parameter size.")
         # ここで First/Mid/LastStage を仮構築して計算もできるが、引数が足りない
         # 簡易的に、1ブロックあたりのパラメータ数を概算し、ステージのブロック数をかける
         # (BertBlock のパラメータ数を別途計算しておく必要がある)
         params_per_block_approx = (hidden_size * hidden_size * 12 + hidden_size * hidden_size * 4 * 4) * dbyte # 仮 (SelfAttention + MLP)
         param_bytes = params_per_block_approx * num_blocks_per_stage
         if is_first:
             # Embedding レイヤーのパラメータを追加
             embed_bytes = (vocab_size * hidden_size + sequence_length * hidden_size + 2 * hidden_size) * dbyte # word + pos + type
             param_bytes += embed_bytes
         if is_last:
             # Head レイヤーのパラメータを追加
             head_bytes = (hidden_size * 2 * d_spec.num_classes) * dbyte # num_classes が必要
             param_bytes += head_bytes
    
    # (1) 静的メモリ: パラメータ、勾配、オプティマイザ状態
    gradient_bytes = param_bytes
    optim_bytes = 2 * param_bytes # Adam 仮定

    # (2) 入力データ (グローバルバッチサイズ) - First ステージのみ
    input_ids_bytes = global_batch_size * sequence_length * 8 # int64
    attn_mask_bytes = global_batch_size * sequence_length * 8 # int64
    input_data_bytes = input_ids_bytes + attn_mask_bytes if is_first else 0

    # (3) 活性化 (1マイクロバッチ分)
    # (microbatch_size, seq_len, hidden_size) * 2 tensors * bytes_per_element
    activation_bytes_per_block_mb = microbatch_size * sequence_length * hidden_size * 2 * dbyte
    
    # 保存される活性化の総量 (ピーク時)
    # std: パイプラインバブル数分のマイクロバッチ * ステージ内全ブロック
    # ckpt: パイプラインバブル数分のマイクロバッチ * 1ブロック分 + 再計算用の一時メモリ
    # rev: 再計算用の一時メモリのみ
    # pareprop: 再計算用の一時メモリのみ (sequential より少し多いかも？)
    
    # ピーク時のマイクロバッチ数を計算
    # 1F1B スケジュールでは、最大で (num_stages - stage_id) 個の Forward と
    # (stage_id + 1) 個の Backward が同時に存在する可能性がある
    # メモリピークは Forward が最も進んだとき or Backward が始まったとき？
    # Forward が最も進んだタイミングでは (num_stages - stage_id) 個の MB の活性化が溜まる
    num_concurrent_microbatches_peak = num_stages - stage_id

    if is_std:
        total_activation_bytes = activation_bytes_per_block_mb * num_blocks_per_stage * num_concurrent_microbatches_peak
    elif is_ckpt:
        # 保持する活性化 (1ブロック分 * バブル数) + 再計算中の最大メモリ (1ブロック分 * 2?)
        kept_activation = activation_bytes_per_block_mb * num_concurrent_microbatches_peak
        recompute_temp = activation_bytes_per_block_mb * 2 # 仮
        total_activation_bytes = kept_activation + recompute_temp
    elif is_rev:
        # 再計算中の一時メモリのみ
        if is_pareprop:
            recompute_temp = activation_bytes_per_block_mb * 2.0 # 仮
        else: # Sequential
            recompute_temp = activation_bytes_per_block_mb # 仮
        total_activation_bytes = recompute_temp
        # First Stage の input_cache も考慮に入れるべき
        if is_first:
             input_cache_bytes = input_data_bytes # グローバルバッチサイズ分
             # total_activation_bytes += input_cache_bytes # カテゴリが違うので別に加算
        else:
             input_cache_bytes = 0
    else:
        total_activation_bytes = 0
        input_cache_bytes = 0


    # (4) 送受信バッファ (ピーク時)
    # Forward 受信バッファ + Backward 勾配受信バッファ + Backward Racts 受信バッファ
    buffer_bytes_per_mb = microbatch_size * sequence_length * hidden_size * 2 * dbyte
    
    fwd_recv_buffers = 0
    if not is_first:
        if is_rev:
             # RingBuffer なのでピーク時も 1 (or is_last なら 2)
             fwd_recv_buffers = buffer_bytes_per_mb * (2 if is_last else 1)
        else: # std or ckpt
             fwd_recv_buffers = buffer_bytes_per_mb * num_concurrent_microbatches_peak

    bwd_grad_recv_buffers = 0
    if not is_last:
        # RingBuffer なのでピーク時も 1 (or is_last なら 2?) -> 2と仮定
        bwd_grad_recv_buffers = buffer_bytes_per_mb * (2 if is_last else 1) # 仮

    bwd_ract_recv_buffers = 0
    if is_middle and is_rev:
        # RingBuffer なのでピーク時も 1
        bwd_ract_recv_buffers = buffer_bytes_per_mb

    total_buffer_bytes = fwd_recv_buffers + bwd_grad_recv_buffers + bwd_ract_recv_buffers


    # --- ★ 最大メモリ使用量の合計 ---
    res = 0
    res += param_bytes      # パラメータ
    res += optim_bytes      # オプティマイザ状態
    res += gradient_bytes   # 勾配 (backward 中に確保される)
    res += input_data_bytes # 入力データ (is_first のみ)
    res += total_buffer_bytes # 送受信バッファ (ピーク時)
    res += total_activation_bytes # 活性化 (ピーク時)
    if is_rev and is_first:
        res += input_cache_bytes # Input Cache (Rev & First)

    # Autograd Detail や Temporary は無視

    return res / _GiB # convert to GiB