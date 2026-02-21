# ファイル名: memory_predictor_prev.py (★ is_pareprop 分岐を追加)

from __future__ import annotations
from typing import List, Tuple, Dict
from .events import EventSpan
# ★ config と env の型ヒントを追加 (cfg を使うため)
from src.config import GlobalConfig, ParMode
from src.env import EnvContext
from src.dataset_registry import DatasetSpecRevised
import torch # ★ torch をインポート

_GiB = 1024 ** 3

def build_pred_timeline(
    spans: List[EventSpan],
    cfg: GlobalConfig, # ★ 型ヒントを追加
    env: EnvContext,   # ★ 型ヒントを追加
    d_spec: DatasetSpecRevised, # ★ 型ヒントを修正
    model: torch.nn.Module, # ★ 型ヒントを追加
) -> Tuple[List[float], Dict[str, List[float]]]:
    
    hidden_size = cfg.hidden_size
    seq_len = cfg.max_position_embeddings
    num_blocks = cfg.num_hidden_layers
    num_classes = d_spec.num_classes
    dbyte = torch.finfo(cfg.autocast_dtype).bits // 8 #テンソル1要素あたりのバイト数
    activation_bytes_per_layer = cfg.batch_size * seq_len * hidden_size * 2 * dbyte # フルバッチサイズでのバイト単位のメモリ消費

    stage_id = env.rank
    num_stages = env.world_size
    if cfg.par_mode == ParMode.PP:
        num_blocks_per_stage = num_blocks // num_stages #パイプライン並列の層の数を計算
    else:
        num_blocks_per_stage = num_blocks

    is_first = (stage_id == 0)
    is_last = (stage_id == num_stages - 1)
    is_middle = not (is_first or is_last)

    is_ckpt = cfg.checkpointing
    is_rev = cfg.reversible
    # ★ is_pareprop フラグを取得
    is_pareprop = cfg.is_pareprop if is_rev else False # is_rev が False なら is_pareprop も False
    is_std = not (is_ckpt or is_rev)

    cats = [ # (変更なし)
        "PARAMETER", "OPTIMIZER_STATE", "INPUT", "ACTIVATION", 
        "GRADIENT", "AUTOGRAD_DETAIL", "TEMPORARY", "UNKNOWN",
    ]
    cur: Dict[str, float] = {c: 0.0 for c in cats} #各イベントのメモリ使用量を保持する辞書を0で初期化

    param_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters() if p.requires_grad) #全パラメータのうち勾配計算が必要なものの合計バイト数を計算
    optim_bytes = 2 * param_bytes # オプティマイザが保持するメモリはパラメータの２倍
    gradient_bytes = param_bytes # 勾配が保持するメモリサイズはパラメータと同等

    input_ids_bytes = cfg.batch_size * seq_len * 8 # int64(8バイト)でのinput_idsのメモリ消費
    attn_mask_bytes = cfg.batch_size * seq_len * 8 # int64(8バイト)でのattention_maskのメモリ消費
    input_data_bytes = input_ids_bytes + attn_mask_bytes

    activation_bytes_per_block_mb = cfg.microbatch_size * seq_len * hidden_size * 2 * dbyte # 2をかけるのはtransformerのattention層とMLP層の両方を保持するから
    if is_std:
        chunk_actv_bytes = activation_bytes_per_block_mb * num_blocks_per_stage # 中間活性化を保持する通常の順伝播のメモリ消費
    elif is_ckpt:
        chunk_actv_bytes = activation_bytes_per_block_mb
    elif is_rev:
        chunk_actv_bytes = 0 #リバーシブルなら活性化保存しないからメモリ消費0のはず
    else:
        chunk_actv_bytes = 0

    # パイプ欄並列のために確保される送受信バッファの通信量の合計サイズを計算(ステージの位置や実行方法により異なる)
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
        
    autograd_bytes = 0
    
    #学習開始前のメモリ状態をセット
    cur["PARAMETER"]       = param_bytes 
    cur["OPTIMIZER_STATE"] = optim_bytes
    cur["GRADIENT"]      = 0.0 
    cur["AUTOGRAD_DETAIL"] = 0.0 
    cur["ACTIVATION"] = 0.0 
    cur["INPUT"]           = 0.0 
    if is_first: 
        cur["INPUT"] += input_data_bytes
    cur["INPUT"] += static_buffer_bytes
    cur["TEMPORARY"]     = 0.0 
    cur["UNKNOWN"]       = 0.0

    ts:      List[float]            = []
    traces:  Dict[str, List[float]] = {c: [] for c in cats}

    # 時刻tでの層メモリ消費量を計算して返す
    def _push(t):
        ts.append(t)
        for k in cats:
            traces[k].append(cur[k] / _GiB)

    # 最初のイベントが始まる時刻でのメモリ状態を記録
    _push(spans[0].t0_ms)

    is_gradient_assigned = False
    forward_chunk_id = 0
    backward_chunk_id = 0
    is_input_cache_allocated = False

    # events.pyで記録された全てのイベント(spans)を一つずつ処理
    for ev in spans:

        if ev.label == "forward_one_chunk":
            _push(ev.t0_ms) 
            if is_std or is_ckpt:
                 cur["ACTIVATION"] += chunk_actv_bytes 
            if is_rev and is_first and not is_input_cache_allocated:
                 cur["INPUT"] += input_data_bytes 
                 is_input_cache_allocated = True
            _push(ev.t1_ms) 
            forward_chunk_id += 1

        elif ev.label == "backward_one_chunk":
            _push(ev.t0_ms) 

            # (1) Gradient buffer (変更なし)
            if not is_gradient_assigned:
                cur["GRADIENT"] += gradient_bytes
                is_gradient_assigned = True

            # (2) Activation recomputation - ★ is_pareprop で分岐 ★
            temp_activation_for_recompute = 0
            if is_ckpt and not is_first: 
                temp_activation_for_recompute = activation_bytes_per_block_mb * 2 # 仮
                cur["ACTIVATION"] += temp_activation_for_recompute
                
            # --- ★ is_rev の場合の分岐 ---
            elif is_rev: 
                # sequential_backward と pareprop_backward で必要な
                # 一時メモリを見積もる。
                # 現状は仮にどちらも同じサイズとするが、必要なら is_pareprop で分岐させる。
                if is_pareprop:
                     # Pareprop は並列実行するため、少し多めに見積もる？ (仮)
                     temp_activation_for_recompute = activation_bytes_per_block_mb * 1.5 # 仮
                else: # Sequential
                     temp_activation_for_recompute = activation_bytes_per_block_mb # 仮
                
                # 全ステージで再計算が発生
                if is_first or is_middle or is_last:
                    cur["ACTIVATION"] += temp_activation_for_recompute
            # --- ★ is_rev 分岐終了 ---

            # (3) Input Cache release (Rev & First) (変更なし)
            if is_rev and is_first:
                 if backward_chunk_id == cfg.num_microbatches - 1 and is_input_cache_allocated:
                      cur["INPUT"] -= input_data_bytes
                      is_input_cache_allocated = False 

            _push(ev.t1_ms) # イベント終了時点

            # --- Memory release after backward ---
            # (4) Release recomputation memory (変更なし)
            if temp_activation_for_recompute > 0:
                 cur["ACTIVATION"] -= temp_activation_for_recompute

            # (5) Release saved activations (std) (変更なし)
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

# tsはイベントが起こりメモリ消費量が変化する時刻を格納したリスト(ex:[0.0, 15.2, 18.5, 30.7, 33.1, ...])
# ts[10]が18.5のとき、traces["parameter"][10]には18.5ミリ秒時点でのparameterの予測使用量が入っている