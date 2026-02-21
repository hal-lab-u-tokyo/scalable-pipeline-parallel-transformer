# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Generic, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.fx.node import map_aggregate
from torch.nn.parallel import DistributedDataParallel
from torch.utils._pytree import tree_map_only

from torch.profiler import record_function
import torch.nn.functional as F

from ._backward import stage_backward, stage_backward_input, stage_backward_weight
from ._debug import map_debug_info
from ._utils import flatten_args, PipeInfo, validate_tensors_metadata


__all__ = [
    "RevPipelineStage",
]

logger = logging.getLogger(__name__)

# GPU間のデータ送受信に関しては必ずタプル形式で行うという規格
# tensor_A → (tensor_A,), [tensor_A, tensor_B] → (tensor_A, tensor_B)
def _normalize_model_output_as_tuple(output: Any) -> Tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.

    Currently, we let user provided models return either a Tensor or a tuple of Tensors from each stage. Due to
    torch.export tracing, compiled models may also return a list instead of a Tuple, which we will normalize back to a
    tuple for consistency.

    TODO: should we be stricter about asserting that stage modules (intermediate and output) all return only Tensor
    values?
    """
    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondance with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """

    def __init__(self, tensor):
        self.meta = tensor.to("meta")


class _RecvInfo:
    """
    Represents a stage input.
    """

    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input
        self.source = source
        # Buffer to receive the input into.
        self.buffer = buffer

    def __repr__(self):
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={self.buffer.size()})"


# An input can be either a received activation or a model input
InputInfo = Union[_RecvInfo, _RootArgPlaceholder]


def _make_tensor_from_meta(
    example: Union[torch.Tensor, FakeTensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example.size(),
        dtype=example.dtype,
        layout=example.layout,
        device=device,
    )

# 20250625 Ryota Miyagi
T = TypeVar("T")
class RingBuffer(Generic[T]):
    """dict compatible ring buffer."""
    def __init__(self, size: int):
        self.size = size
        self._data: List[Optional[T]] = [None] * size

    def __setitem__(self, idx: int, value: T) -> None:
        self._data[idx % self.size] = value

    def __getitem__(self, idx: int) -> T:
        item = self._data[idx % self.size]
        if item is None:
            raise KeyError(f"{idx} is not populated yet")
        return item

    def values(self):
        return [v for v in self._data if v is not None]

    def clear(self):
        for i in range(self.size):
            self._data[i] = None

#学習時はこのクラスがGPUの数だけ作られ、stage_indexを割り当ててそれぞれがどの処理を担当するのか(first, middle, last)とかを判定していく
# ★★★★★ _RevPipelineStageBase (最終整理版) ★★★★★
class _RevPipelineStageBaseWithPareprop(ABC):
    """
    パイプラインステージの基底クラス。
    リバーシブル実行 (カスタム逆伝播) と標準 Autograd 実行の両方をサポート。
    """
    def __init__(
        self,
        submodule: nn.Module, # ★ 型ヒントを nn.Module に
        stage_index: int,
        num_stages: int,
        device: torch.device,
        eval_actv_err: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        # dw_builder はカスタム逆伝播戦略では不要
    ):
        super().__init__()
        if stage_index >= num_stages: raise ValueError(...)

        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages: raise RuntimeError(...)

        # --- 実行モードフラグ ---
        # submodule が is_pareprop 属性を持つかチェック (bert_stages.py が設定)
        self.is_pareprop = getattr(submodule, "is_pareprop", False)
        # submodule がカスタム逆伝播メソッドを持つかチェック
        self.is_reversible = hasattr(submodule, 'sequential_backward') or hasattr(submodule, 'pareprop_backward')

        # --- ランタイム状態 ---
        self._outputs_meta: Optional[Tuple[Tensor, ...]] = None
        # fwd_cache: forward の出力を一時保存 (get_fwd_send_ops で pop / backward で参照)
        self.fwd_cache: Dict[int, Tuple[Tuple[Any, ...], List[Tensor]]] = {} # (output_tuple, input_tensors_list)
        self.reversiblelayer_cache = {} if self.is_last else None
        # bwd_cache: backward の結果を一時保存 (get_bwd_send_ops で pop)
        self.bwd_cache: Dict[int, Tuple[Optional[Tuple[Optional[Tensor], ...]], Optional[Tuple[Any, ...]]]] = {} # (grads_input, input_values)
        self.output_chunks: List[Any] = [] # is_last の最終出力
        self.has_backward = False
        self.log_prefix = f"[Stage {self.stage_index}]"
        self.eval_actv_err = eval_actv_err; self.actv_cache_for_eval = {}; self.mse_stats = []; self.cos_stats = []
        # input_cache: is_first の元の入力を保持 (リバーシブルパスの backward で使用)
        self.input_cache: Dict[int, Tuple[Tuple[Any, ...], Dict[str, Any]]] = {}

        # --- 通信バッファ ---
        # is_last は forward 受信不要、backward 送信不要のため buf_size=1 でも良いが、
        # 損失計算などで余分に保持する可能性を考慮し、元の実装通り 2 にしておく
        self.buf_size = 2 if self.is_last else 1
        self.args_recv_info: RingBuffer[Tuple[InputInfo, ...]] = RingBuffer(self.buf_size)
        self.act_send_info: Dict[int, List[Optional[int]]] = {} # ★ Optional[int] に変更
        self.grad_recv_info: RingBuffer[Tuple[_RecvInfo, ...]] = RingBuffer(self.buf_size)
        self.grad_send_info: Optional[List[Optional[int]]] = None
        # Racts 受信バッファは is_middle かつ is_reversible の場合のみ RingBuffer(1)
        self.ract_recv_info: RingBuffer[Tuple[_RecvInfo, ...]] = RingBuffer(1) if (self.is_middle or self.is_first) and self.is_reversible else RingBuffer(0) # size 0 も可
        self.ract_send_info: Optional[List[Optional[int]]] = None

        # --- その他 ---
        self.chunks: Optional[int] = None
        self.stage_index_to_group_rank: Dict[int, int] = {i: i % self.group_size for i in range(self.num_stages)}

    # --- Properties (変更なし) ---
    @property
    def has_backward(self) -> bool: return self._has_backward
    @has_backward.setter
    def has_backward(self, has_backward: bool): self._has_backward = has_backward
    @property
    def is_first(self): return self.stage_index == 0
    @property
    def is_last(self): return self.stage_index == self.num_stages - 1
    @property
    def is_middle(self) -> bool: return not (self.is_first or self.is_last)

    # --- Metadata and Infra Setup (racts 関連以外は変更なし) ---
    def _check_chunk_id(self, chunk_id: int): # ...
        if self.chunks is None: raise RuntimeError("Chunks not configured.")
        if not 0 <= chunk_id < self.chunks: raise RuntimeError(f"Chunk id {chunk_id} out of range [0, {self.chunks})")
    def _configure_outputs_meta(self, outputs_meta: Tuple[Tensor, ...]): # ...
        assert self._outputs_meta is None, "Output meta reconfig not supported"
        self._outputs_meta = tuple(outputs_meta)
    def get_outputs_meta(self) -> Tuple[Tensor, ...]: # ...
        assert self._outputs_meta is not None, "Output meta not configured"
        return self._outputs_meta
    def _create_grad_send_info( self, args_recv_info: Tuple, ) -> List[Optional[int]]: # ...
        info = []; map_aggregate(args_recv_info, lambda a: info.append(a.source if isinstance(a, _RecvInfo) else None)); return info
    def _create_ract_send_info( self, args_recv_info: Tuple, ) -> List[Optional[int]]: # ...
        # is_first は Racts を送らない想定
        if self.is_first: return [None] * len(flatten_args(args_recv_info)) # 仮
        info = []; map_aggregate(args_recv_info, lambda a: info.append(a.source if isinstance(a, _RecvInfo) else None)); return info

    @abstractmethod
    def _prepare_forward_infra( self, num_microbatches: int, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None, ) -> Tuple[Any, ...]: raise NotImplementedError

    def _prepare_backward_infra(self, num_microbatches: int):
        """逆伝播に必要な通信バッファ情報 (受信側) を準備する。"""
        if not self.has_backward: return
        self.chunks = num_microbatches
        for i in range(num_microbatches):
            # Grad 受信情報は常に準備 (is_last 以外)
            if not self.is_last:
                self.grad_recv_info[i] = self._create_grad_recv_info(self.act_send_info)
            # Racts 受信情報は is_middle or is_first かつ is_reversible の場合のみ準備
            if (self.is_middle or self.is_first) and self.is_reversible:
                self.ract_recv_info[i] = self._create_ract_recv_info(self.act_send_info)

    @abstractmethod
    def _create_grad_recv_info( self, act_send_info: Dict, ) -> Tuple[_RecvInfo, ...]: raise NotImplementedError
    @abstractmethod
    def _create_ract_recv_info( self, act_send_info: Dict, ) -> Tuple[_RecvInfo, ...]: raise NotImplementedError

    # --- Communication Ops Helpers (get_bwd_recv_ops 以外は変更なし) ---
    def _get_recv_ops(self, recv_infos: Tuple[InputInfo, ...], ) -> List[dist.P2POp]: # ...
        ops = []; rank_map = self.stage_index_to_group_rank
        for info in recv_infos:
            if isinstance(info, _RecvInfo):
                peer = rank_map[info.source]; global_peer = peer if self.group is None else dist.get_global_rank(self.group, peer)
                ops.append(dist.P2POp(dist.irecv, info.buffer, global_peer, self.group))
        return ops
    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]: # ...
        return self._get_recv_ops(self.args_recv_info[fwd_chunk_id])

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """逆伝播に必要な勾配と (必要な場合) 復元活性化を受信する操作リストを返す。"""
        if not self.has_backward: return []
        ops = []
        # Grad 受信 (is_last 以外)
        if not self.is_last:
            ops.extend(self._get_recv_ops(self.grad_recv_info[bwd_chunk_id]))
        # Racts 受信 (is_middle かつ is_reversible のみ)
        if (self.is_middle or self.is_first) and self.is_reversible:
             ops.extend(self._get_recv_ops(self.ract_recv_info[bwd_chunk_id]))
        return ops

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> List[dist.P2POp]:
        """順伝播の出力を次のステージに送信し、不要ならキャッシュから削除する。"""
        output_tuple = ()
        # is_last は送信しないので何もしない
        if self.is_last:
            pass
        
        # is_first または is_middle の場合
        else:
            if fwd_chunk_id in self.fwd_cache:
                # output_tuple = self.fwd_cache.pop(fwd_chunk_id)[0] # fwd_cacheにはforward_inputしか保存しないことにした
                output_tuple = self.fwd_cache.pop(fwd_chunk_id)                 
            else:
                 logger.warning(f"Chunk {fwd_chunk_id} not found in fwd_cache for sending.")
                 # return [] # 空のopsを返す

        ops: List[dist.P2POp] = []
        # ... (送信ロジックは変更なし) ...
        rank_map = self.stage_index_to_group_rank
        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info.get(idx, []) # Use .get for safety
            for dst in dst_stages:
                if dst is not None:
                    # logger.debug(...)
                    peer = rank_map[dst]; global_peer = peer if self.group is None else dist.get_global_rank(self.group, peer)
                    ops.append(dist.P2POp(dist.isend, out, global_peer, self.group))
        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> List[dist.P2POp]:
        """逆伝播の結果 (勾配 grads_input と復元活性化 input_values) を前のステージに送信する。"""
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first: return [] # First は送信不要

        # 送信先情報を準備 (初回のみ)
        if self.grad_send_info is None: self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])
        if self.ract_send_info is None and self.is_reversible: self.ract_send_info = self._create_ract_send_info(self.args_recv_info[0])

        ops: List[dist.P2POp] = []
        # bwd_cache から結果を取得し、キャッシュから削除
        if bwd_chunk_id not in self.bwd_cache:
             logger.warning(f"Chunk {bwd_chunk_id} not found in bwd_cache for sending.")
             return ops # or raise error?
        grads_input, input_values = self.bwd_cache.pop(bwd_chunk_id)

        rank_map = self.stage_index_to_group_rank
        # Send Gradients (grads_input)
        if grads_input is not None and self.grad_send_info is not None:
            for grad, dest in zip(grads_input, self.grad_send_info):
                if isinstance(grad, Tensor) and dest is not None:
                    # logger.debug(...)
                    peer = rank_map[dest]; global_peer = peer if self.group is None else dist.get_global_rank(self.group, peer)
                    ops.append(dist.P2POp(dist.isend, grad, global_peer, self.group))
        # Send Racts (input_values) - is_reversible の場合のみ
        if input_values is not None and self.is_reversible and self.ract_send_info is not None:
             # Ensure input_values is a tuple for zip
             input_values_tuple = input_values if isinstance(input_values, tuple) else (input_values,)
             for ract, dest in zip(input_values_tuple, self.ract_send_info):
                 # is_first (dest=0) には Racts を送らない
                 if isinstance(ract, Tensor) and dest is not None:
                     # logger.debug(...)
                     peer = rank_map[dest]; global_peer = peer if self.group is None else dist.get_global_rank(self.group, peer)
                     ops.append(dist.P2POp(dist.isend, ract, global_peer, self.group))
        return ops

    # --- Runtime State Management (clear_runtime_states 以外変更なし) ---
    def clear_runtime_states(self) -> None:
        """ステージのランタイム状態をクリアする。"""
        self.fwd_cache.clear()
        self.output_chunks.clear()
        self.bwd_cache.clear() # ★ bwd_cache もクリア
        self.input_cache.clear() # ★ input_cache もクリア
        self.actv_cache_for_eval.clear() # ★ eval cache もクリア

        # 受信バッファの勾配をクリア
        for rt in self.args_recv_info.values():
            if rt: # Check if buffer is populated
                 for a in rt:
                     if isinstance(a, _RecvInfo) and isinstance(a.buffer, Tensor): a.buffer.grad = None

    def _map_tensor_from_recv_info( self, recv_infos: Tuple[InputInfo, ...], ): # ...
        return map_aggregate(recv_infos, lambda info: info.buffer if isinstance(info, _RecvInfo) else ...) # Simplified
    def _retrieve_recv_activations(self, fwd_chunk_id: int): # ...
        return self._map_tensor_from_recv_info(self.args_recv_info[fwd_chunk_id])
    def _retrieve_recv_grads( self, bwd_chunk_id: int, ): # ...
        return self._map_tensor_from_recv_info(self.grad_recv_info[bwd_chunk_id])
    def _retrieve_recv_racts( self, bwd_chunk_id: int, ): # ...
        if not ((self.is_middle or self.is_first) and self.is_reversible): # Racts は Middle/Rev のみ受信
            raise RuntimeError("Racts should only be received by middle reversible stages.")
        return self._map_tensor_from_recv_info(self.ract_recv_info[bwd_chunk_id])

    # --- Core Execution Logic ---
    def forward_maybe_with_nosync(self, *args, **kwargs): # (変更なし)
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync(): out_val = self.submod(*args, **kwargs)
        else: out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync( self, backward_type, bwd_kwargs: Dict, last_backward=False ) -> Tuple[Tuple[Optional[Tensor], ...], Optional[List[Dict[str, Any]]]]:
        """標準 Autograd を実行する場合のラッパー (DP 同期制御付き)。"""
        # (カスタム逆伝播パスでは呼び出されない想定)
        def perform_backward() -> Tuple[Tuple[Optional[Tensor], ...], Optional[List[Dict[str, Any]]]]:
             # stage_backward_input/weight は使わないので "full" のみ実装
             if backward_type == "full":
                 return stage_backward(
                     bwd_kwargs["stage_output"],
                     bwd_kwargs["output_grads"],
                     bwd_kwargs["input_values"],
                 ), None
             else:
                 raise RuntimeError(f"Unsupported backward type '{backward_type}' for non-reversible path.")

        result: Tuple[Tuple[Optional[Tensor], ...], Optional[List[Dict[str, Any]]]]
        # --- DP 同期制御 ---
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward: # ... reducer prepare ...
                self.submod.reducer.prepare_for_backward(list(torch.nn.parallel.distributed._find_tensors(bwd_kwargs["stage_output"]))) # Simplified
                result = perform_backward()
            else: # ... no_sync ...
                with self.submod.no_sync(): result = perform_backward()
        elif isinstance(self.submod, FSDPModule):
            # ... set flags false ...
            self.submod.set_is_last_backward(False); self.submod.set_reshard_after_backward(False); self.submod.set_requires_gradient_sync(False);
            result = perform_backward()
            if last_backward: # ... run post backward ...
                 def run_post_backward(fsdp_module: FSDPModule):
                      fsdp_module.set_is_last_backward(True); fsdp_module.set_reshard_after_backward(True); fsdp_module.set_requires_gradient_sync(True);
                      fsdp_state = fully_shard.state(fsdp_module)
                      for state in fsdp_state._state_ctx.all_states:
                           if state._fsdp_param_group: state._fsdp_param_group.post_backward()
                 run_post_backward(self.submod)
        else: # Non-DP
            result = perform_backward()
        return result

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """1マイクロバッチ分の順伝播を実行し、結果をキャッシュする。"""
        with record_function("## forward_one_chunk ##"):
            # 1. 入力準備 (受信 or 外部引数)
            if self.is_first:
                composite_args = args
            else:
                composite_args = self._retrieve_recv_activations(fwd_chunk_id)
            composite_kwargs = kwargs or {}
            self._validate_fwd_input(args, kwargs)

            # 2. is_first の場合、元の入力をキャッシュ (Rev backward 用)
            if self.is_first:
                self.input_cache[fwd_chunk_id] = (composite_args, composite_kwargs)

            output = None
            if self.is_last:
                try:
                    x_body_out = self.submod.forward_features(*composite_args, **composite_kwargs)
                    x_body_out_detached = tuple(t.detach().requires_grad_(True) for t in x_body_out)
                    self.reversiblelayer_cache[fwd_chunk_id] = x_body_out_detached
                    output = self.submod.forward_head(x_body_out_detached)
                    output_detached = tree_map_only(torch.Tensor, lambda t: t.detach(), output)
                    self.output_chunks.append(output_detached)
                    output_tuple = _normalize_model_output_as_tuple(output)
                except Exception as e: raise RuntimeError(...) from e
            
            else:
                try:
                    output = self.submod(*composite_args, **composite_kwargs)
                    output_tuple = _normalize_model_output_as_tuple(output)
                    self.fwd_cache[fwd_chunk_id] = output_tuple
                except Exception as e: raise RuntimeError(...) from e
                
            # 5. fwd_cache に結果を保存
            #    (get_fwd_send_ops で pop / is_last の backward で参照)
            # flat_args = flatten_args(composite_args)
            # flat_kwargs = flatten_args(composite_kwargs)          
            # self.fwd_cache[fwd_chunk_id] = (output_tuple, flatten_input_tensors)
            # self.fwd_cache[fwd_chunk_id] = output_tuple

            # 6. エラー評価用キャッシュ (is_middle and eval_actv_err)
            if self.is_middle and self.eval_actv_err:
                flat_args = flatten_args(composite_args)
                flat_kwargs = flatten_args(composite_kwargs)
                flatten_input_tensors = flat_args + flat_kwargs
                self.actv_cache_for_eval[fwd_chunk_id] = tuple(t.detach().clone() for t in flatten_input_tensors)

            # logger.debug(...)
            self._validate_fwd_outputs(output_tuple)
            return output

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ `backward_one_chunk` (最終版)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss: Optional[Tensor] = None, # 損失値 (is_last)
        full_backward: bool = True, # I/W 分離用 (カスタム逆伝播では常に True 扱い)
        last_backward: bool = False, # DP 同期用
    ):
        """1マイクロバッチ分の逆伝播を実行し、結果をキャッシュする。"""
        with record_function("## backward_one_chunk ##"):
            self._check_chunk_id(bwd_chunk_id)

            grads_input: Optional[Tuple[Optional[Tensor], ...]] = None # -> Previous Stage
            input_values: Optional[Tuple[Any, ...]] = None          # -> Previous Stage (as Racts)

            # --- 1. 実行パスの選択 (リバーシブルか標準Autogradか) ---
            if self.is_reversible:
                # --- A. リバーシブル実行 (カスタム逆伝播) ---
                logger.debug(f"{self.log_prefix} Running custom backward for chunk {bwd_chunk_id}")

                # backward 関数を選択 (Pareprop or Sequential)
                backward_fn = self.submod.pareprop_backward if self.is_pareprop else self.submod.sequential_backward

                # --- 1a. ステージに応じた引数を準備 ---
                args_for_backward = []
                if self.is_first:

                    # 入力: 次からの dY, 元の入力 (input_cache)
                    grads_output = self._retrieve_recv_grads(bwd_chunk_id) # dY を取得
                    r_stage_output = self._retrieve_recv_racts(bwd_chunk_id)
                    original_args, original_kwargs = self.input_cache[bwd_chunk_id] # X を取得
                    
                    # ★ Y, dY, X の順で引数を構成
                    args_for_backward = list(r_stage_output) + list(grads_output) + list(original_args)              
                # if self.is_first:
                #     # 入力: 次からの dY, 元の入力 (input_cache)
                #     grads_output = self._retrieve_recv_grads(bwd_chunk_id)
                #     original_args, original_kwargs = self.input_cache[bwd_chunk_id] # pop しない
                #     # backward_fn は Y を要求しないはず (FirstStage.backward が Y をキャッシュ)
                #     args_for_backward = list(grads_output) + list(original_args)
                #     # TODO: original_kwargs も渡す必要があれば I/F 変更

                elif self.is_middle:
                    # 入力: 次からの dY, 次からの Y (racts)
                    grads_output = self._retrieve_recv_grads(bwd_chunk_id)
                    r_stage_output = self._retrieve_recv_racts(bwd_chunk_id)
                    args_for_backward = list(r_stage_output) + list(grads_output)

                elif self.is_last:
                    # 入力: 損失勾配 (loss), fwd_cache から取得した元の入力 (値のみ)
                    loss = torch.tensor(1.0, device=self.device) if loss is None else loss # スカラ損失なら勾配1.0
                    if bwd_chunk_id not in self.reversiblelayer_cache: raise RuntimeError(...)
                    cached_y_body = self.reversiblelayer_cache.pop(bwd_chunk_id)
                    # backward_fn は reconstructed_input_values を要求する I/F になっている
                    if isinstance(cached_y_body, torch.Tensor):
                        cached_y_body = (cached_y_body,)
                    args_for_backward = [loss, cached_y_body]

                # --- 1b. カスタム逆伝播を実行 ---
                try:
                    grads_input, input_values = backward_fn(*args_for_backward)
                except Exception as e:
                     logger.error(f"{self.log_prefix} Custom backward failed for chunk {bwd_chunk_id}")
                     raise e

                if self.eval_actv_err and bwd_chunk_id in self.actv_cache_for_eval:
                    # オリジナルの入力を取得（メモリ節約のため pop するのが先輩流）
                    orig_inputs = self.actv_cache_for_eval.pop(bwd_chunk_id) # ★ popに変更
                    
                    # 復元された入力 (input_values) をタプル化
                    recon_inputs = input_values if isinstance(input_values, tuple) else (input_values,)
                    
                    # 各テンソルごとに比較
                    for orig, recon in zip(orig_inputs, recon_inputs):
                        if isinstance(orig, torch.Tensor) and isinstance(recon, torch.Tensor):
                            # 先輩のコード準拠: バッチサイズ B を取得し、(B, -1) に変形して計算
                            B = orig.size(0)
                            
                            # 精度確保のため double にキャストして計算
                            orig_flat  = orig.double().view(B, -1)
                            recon_flat = recon.double().view(B, -1)
                            
                            # サンプルごとの MSE を計算
                            mse_per = (recon_flat - orig_flat).pow(2).mean(dim=1)
                            
                            # サンプルごとの Cosine Similarity を計算
                            cos_per = F.cosine_similarity(recon_flat, orig_flat, dim=1)
                            
                            # リストに追加 (extend を使用して全サンプルのデータを保存)
                            self.mse_stats.extend(mse_per.tolist())
                            self.cos_stats.extend(cos_per.tolist())
                # --- 1c. DP 同期 (手動) ---
                if last_backward and isinstance(self.submod, (DistributedDataParallel, FSDPModule)):
                    # カスタム逆伝播はパラメータの .grad 属性に直接書き込むため、
                    # backward_maybe_with_nosync のようなラッパーは使えない。
                    # ここで手動で勾配同期を行う必要がある。
                    if isinstance(self.submod, DistributedDataParallel):
                         # DDP の reducer を直接呼び出す (非推奨だが他に方法がない場合)
                         # self.submod.reducer.reduce() # 動作するか要検証
                         logger.warning("Manual DDP sync for custom backward needs implementation/verification.")
                         pass # TODO
                    elif isinstance(self.submod, FSDPModule):
                         # FSDP の post_backward 相当を手動で実行
                         logger.warning("Manual FSDP sync for custom backward needs implementation/verification.")
                         # run_post_backward(self.submod) # backward_maybe_with_nosync から拝借？
                         pass # TODO

            else:
                # --- B. 標準 Autograd 実行 (非リバーシブル) ---
                logger.debug(f"{self.log_prefix} Running standard autograd backward for chunk {bwd_chunk_id}")

                if bwd_chunk_id not in self.fwd_cache: raise RuntimeError(...)
                stage_output_tuple, input_values_list = self.fwd_cache.pop(bwd_chunk_id) # ★ 非Revは fwd_cache をここで pop
                input_values_tuple = tuple(input_values_list)

                # autograd.backward 用の引数を準備
                if self.is_last:
                    bwd_kwargs = { "stage_output": loss, "output_grads": None, "input_values": input_values_tuple, }
                else:
                    grads_output = self._retrieve_recv_grads(bwd_chunk_id)
                    bwd_kwargs = { "stage_output": stage_output_tuple, "output_grads": grads_output, "input_values": input_values_tuple, }

                # backward_maybe_with_nosync を呼び出して autograd 実行 ("full" のみサポート)
                if not full_backward: logger.warning("full_backward=False ignored.")
                grads_input_tuple, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
                grads_input = grads_input_tuple
                input_values = input_values_tuple # Autograd は入力を復元しない


            # --- 2. キャッシュとクリーンアップ (共通) ---
            self.bwd_cache[bwd_chunk_id] = (grads_input, input_values)

            # メモリ解放 (is_last の output を detach) - Autograd パスでは fwd_cache を pop 済み
            # if self.is_last and not self.is_first and not self.is_reversible:
            #      # (stage_output_to_detach の取得と detach 処理) ...
            #      pass # fwd_cache を pop したので不要？ 要確認

            # is_first の input_cache エントリを削除 (backward 完了後)
            if self.is_first and bwd_chunk_id in self.input_cache:
                 del self.input_cache[bwd_chunk_id]

            # 入力バッファ勾配クリア
            for info in self.args_recv_info[bwd_chunk_id]:
                if isinstance(info, _RecvInfo) and isinstance(info.buffer, Tensor):
                    info.buffer.grad = None

            logger.debug(f"{self.log_prefix} Backwarded chunk {bwd_chunk_id}")


    # --- (_validate_fwd_input, _validate_fwd_outputs 変更なし) ---
    def _validate_fwd_input(self, args, kwargs): # ...
        if self.is_first: expected_args = self.args_recv_info[0]
        else: return
        if len(kwargs): return
        expected_tensors_meta = [e.meta if isinstance(e, _RootArgPlaceholder) else e.buffer for e in expected_args]
        validate_tensors_metadata( f"Stage {self.stage_index} forward inputs", expected_tensors_meta, args )
    def _validate_fwd_outputs(self, outputs: Tuple[Tensor, ...]): # ...
        validate_tensors_metadata( f"Stage {self.stage_index} forward outputs", self.get_outputs_meta(), outputs )


class RevPipelineStageWithPareprop(_RevPipelineStageBaseWithPareprop):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    RevPipelineStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk, with no skip connections.

    RevPipelineStage performs runtime shape/dtype inference automatically by propagating the outputs from stage0 to
    stage1 and so forth, in linear order.  To bypass shape inference, pass the `input_args` and `output_args` to each
    RevPipelineStage instance.

    Args:
        submodule (nn.Module): The PyTorch module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        device (torch.device): The device where this stage is located.
        input_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The input arguments for the submodule.
        output_args (Union[torch.Tensor, Tuple[torch.tensor]], optional): The output arguments for the submodule.
        group (dist.ProcessGroup, optional): The process group for distributed training. If None, default group.
        dw_builder: TODO clean up comments
    """

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        eval_actv_err: bool = False,
        input_args: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, eval_actv_err, group)
        self.inputs: Optional[List[torch.Tensor]] = None
        self.inputs_meta: Optional[Tuple[torch.Tensor, ...]] = None
        # Note: inputs and submod should ideally be on meta device. We decided not to assert this (yet) becuase it
        # might be breaking for existing users.
        if input_args is None:
            assert output_args is None, (
                "If specifying output_args, input_args must also be specified. "
                "Otherwise, shape inference will be performed at runtime"
            )
        else:
            self.inputs_meta = (
                (input_args,) if isinstance(input_args, torch.Tensor) else input_args
            )
            if output_args is None:
                logger.warning(
                    "Deprecation warning: passing input_args and performing init-time shape inference is deprecated. "
                    "RevPipelineStage now supports runtime shape inference using the real inputs provided to schedule step(). "
                    "Either delete `input_args` arg to `RevPipelineStage` to opt-into runtime shape inference, "
                    "or additionally pass `output_args` to `RevPipelineStage` to fully override shape inference. "
                )
                try:
                    with torch.no_grad():
                        output_args = submodule(*self.inputs_meta)
                    output_args = tree_map_only(
                        torch.Tensor, lambda x: x.to("meta"), output_args
                    )
                except Exception as e:
                    raise RuntimeError(
                        "Failed to perform pipeline shape inference- are your inputs on the same device as your module?"
                    ) from e
            assert (
                output_args is not None
            ), "If passing input_args, also pass output_args to override shape inference"
            self._configure_outputs_meta(
                (output_args,) if isinstance(output_args, torch.Tensor) else output_args
            )

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: List[torch.Tensor] = []

        def stage_global_rank(peer_rank):
            return (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )

        self.prev_rank = stage_global_rank((self.group_rank - 1) % self.group_size)
        self.next_rank = stage_global_rank((self.group_rank + 1) % self.group_size)

        dbg_str = (
            f"Finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "  # noqa: G004
            f"{self.is_last=}, {self.num_stages=}, "
        )
        if self.inputs_meta is not None:
            dbg_str += (
                f"inputs: {[inp.shape for inp in self.inputs_meta]}, "
                f"output: {[output.shape for output in self.get_outputs_meta()]}"
            )
        else:
            dbg_str += " running shape-inference at runtime"

        logger.debug(dbg_str)

    def _shape_inference(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if kwargs is None:
            kwargs = {}
        assert args is not None, "Args may be an empty tuple but not None"

        # We skip recv communication if we're the first stage, but also if the previous stage is on the same rank
        # and can pass its output shapes in as args instead of using send/recv.
        if (
            self.is_first
            # if not first stage, then check if prev stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index - 1] == self.group_rank
        ):
            logger.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `args`",
                self.stage_index,
            )
            args = tree_map_only(torch.Tensor, lambda x: x.to("meta"), args)
        else:
            assert (
                len(args) == 0
            ), "Can't supply input args for shape inference on non-first stage"
            objects = [None]
            logger.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.recv_object_list(
                objects, src=self.prev_rank, group=self.group, device=self.device
            )
            recv_args = objects[0]
            assert isinstance(recv_args, tuple), type(recv_args)
            args = recv_args

        # cache input shapes for use during recv buffer allocation
        self.inputs_meta = args
        args = tree_map_only(
            torch.Tensor, lambda x: torch.zeros_like(x, device=self.device), args
        )

        # set attributes needed for forward
        with torch.no_grad():
            logger.debug("Shape inference: stage %s running forward", self.stage_index)
            outputs = self.submod(*args, **kwargs)

        # if single tensor, convert so it is always a list
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        # communicate meta outputs not real outputs for two reasons
        # 1 - its faster (esp. since obj coll pickles tensor data!)
        # 2 - avoid activating a cuda context for the src rank when unpickling on the recv end!
        outputs_meta = tuple(
            tree_map_only(torch.Tensor, lambda x: x.to("meta"), outputs)
        )
        self._configure_outputs_meta(outputs_meta)
        del outputs
        import gc
        torch.cuda.empty_cache()
        # Passing outputs to the next stage:
        # two cases-
        # 1. Usually: use send/recv communication to pass the output
        # 2. Special case: for V-schedules, 2 'adjacent' stages (e.g. stage 3, 4 in an 8-stage 4-rank V)
        #    pass their shape info via return value and function args rather than send/recv.
        if (
            self.is_last
            # if not last stage, then check if next stage is on the same rank
            or self.stage_index_to_group_rank[self.stage_index + 1] == self.group_rank
        ):
            # Case (2) above: pass shape info via return value and caller passes it as args to next stage's
            # _shape_inference call
            logger.debug(
                "Shape inference: stage %s skipping send to next stage",
                self.stage_index,
            )

        else:
            # Case (1): send shapes via send operation, and ensure not to return it to the caller
            logger.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.send_object_list(
                [outputs_meta],
                dst=self.next_rank,
                group=self.group,
                device=self.device,
            )
            outputs_meta = tuple()

        return outputs_meta

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        # TODO move self.device to an argument from step API (from its input tensors)?
        assert num_microbatches is not None, "TODO fix num_microbatches"

        outputs: Tuple[Any, ...] = tuple()
        if self.inputs_meta is None:
            outputs = self._shape_inference(args, kwargs)

        assert self.inputs_meta is not None
        # Receive info during forward
        # TODO: create args_recv_info lazily? (same needed for RevPipelineStage)
        for chunk_id in range(num_microbatches):
            if not self.is_first:
                # We assume that we always receive from stage - 1
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp, self.device),
                        )
                        for inp in self.inputs_meta
                    ]
                )
                # In case there is backward pass, set requires_grad for receive buffers
                if self.has_backward:
                    for r in recv_infos:
                        r.buffer.requires_grad_(True)

                self.args_recv_info[chunk_id] = recv_infos
            else:
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs_meta]
                )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: Dict[int, List] = {}

        for idx in range(len(self.get_outputs_meta())):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []

        return outputs

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        grad_recv_info: Tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(
                            self.get_outputs_meta()[idx], self.device
                        ),
                    )
                    for idx, dst_list in act_send_info.items()
                ]
            )
        return grad_recv_info

    def _create_ract_recv_info(
            self, 
            act_send_info: Dict,
    ) -> Tuple[_RecvInfo, ...]:
        ract_recv_info: Tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving reconstructed activations from multiple sources is not supported
            # hence we only take the first destination
            ract_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_ract_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(
                            self.get_outputs_meta()[idx], self.device
                        ),
                    )
                    for idx, dst_list in act_send_info.items()
                ]
            )
        return ract_recv_info

    def _init_p2p_neighbors(self):
        """
        Set up p2p communitors between previous and next stages
        by sending a dummy tensor.

        If this is used, must be called for all pipeline stages.
        """
        ops = []
        recv_tensor = torch.zeros(1, device="cuda")
        send_tensor = torch.ones(1, device="cuda")
        # forward
        if not self.is_first:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.prev_rank, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.next_rank, self.group))

        # backward
        if not self.is_first:
            ops.append(dist.P2POp(dist.isend, send_tensor, self.prev_rank, self.group))
        if not self.is_last:
            ops.append(dist.P2POp(dist.irecv, recv_tensor, self.next_rank, self.group))

        return True