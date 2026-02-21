# ファイル名: bert_stages.py (修正版)

import torch
import torch.nn as nn
from .bert_blocks import BertEmbeddings, ReversibleBertBlock
from typing import Tuple, Optional, Any # ★ 追加
from torch import Tensor

from .reversible import (
    Coupling as BaseCoupling,
    Decoupling as BaseDecoupling,
    ParePropReversibleBlockWrapper,
    ParePropReversibleSequential
)


# --- BertCoupling / Decouplingをフレームワーク基底クラスから継承 ---
class BertCoupling(BaseCoupling):
    """入力を2つに分割してリバーシブル形式に"""
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x  # 自動でタプル(x, x)が返される

class BertDecoupling(BaseDecoupling):
    """リバーシブル形式から通常のテンソルに戻す"""
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x2, x1 = inputs
        # 2つを結合
        return torch.cat([x1, x2], dim=-1)

# --- BertFullStage (完全版) ---
class BertFullStage(nn.Module): # ★ RevModuleBase 継承を削除
    """パイプライン並列なしの完全なBERTモデル (効率化版)"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 12, # Full model uses all blocks
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        num_classes: int = 2, # Classification head
        enable_amp: bool = False,
        is_pareprop: bool = False, # Controls backward mode
        act_buf: dict | None = None, # Not used in this version
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs, # Allows extra arguments
    ):
        super().__init__()
        
        self.is_pareprop = is_pareprop # Selects backward method later
        self.autocast_dtype = autocast_dtype # For automatic mixed precision
        
        # 1. Embeddings (Input processing)
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size
        )
        
        # 2. Coupling (Convert to reversible format)
        self.coupling = BertCoupling()
        
        # 3. Reversible Body (Main transformer blocks)
        # Both is_pareprop=True/False now use the same structure
        
        # Initialize CUDA streams needed for Pareprop
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.s1 = torch.cuda.Stream(device=device)
        self.s2 = torch.cuda.Stream(device=device)
        
        # Wrap each ReversibleBertBlock with ParePropReversibleBlockWrapper
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
        
        # Use ParePropReversibleSequential for the body
        self.body = ParePropReversibleSequential(
            modules=wrapped_blocks,
            s1=self.s1, s2=self.s2, # Pass streams
            autocast_dtype=autocast_dtype,
            # debug_context is omitted here but could be added
        )
        
        # 4. Decoupling (Convert back from reversible format)
        self.decoupling = BertDecoupling()
        
        # 5. Final Layers (Normalization and Classification Head)
        # LayerNorm input size is 2 * hidden_size because decoupling concatenates
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.head = nn.Linear(2 * hidden_size, num_classes)
    
    def forward(self, input_ids, token_type_ids=None):
        """Standard BERT forward pass"""
        
        # 1. Apply embeddings
        x = self.embeddings(input_ids, token_type_ids)
        
        # 2. Convert to reversible format
        x = self.coupling(x) # Output: (x, x)
        
        # 3. Pass through reversible blocks
        # This calls ParePropReversibleSequential.forward, which runs in no_grad
        x = self.body(x)
        
        # 4. Convert back to standard tensor
        x = self.decoupling(x) # Output: concatenated tensor
        
        # 5. Extract CLS token output and apply final layers
        cls_output = x[:, 0] # Get the first token's output
        x = self.norm(cls_output)
        x = self.head(x)
        
        return x

class BertFirstStage(nn.Module):
    """パイプラインの最初のステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        vocab_size: int = 30522,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        enable_amp: bool = False,
        # --- 2. フレームワーク用の引数を追加 ---
        is_pareprop: bool = False,
        act_buf: dict | None = None,
        autocast_dtype: torch.dtype = torch.float32,
        debug_context: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size
        )
        self.is_pareprop = is_pareprop
        self.autocast_dtype = autocast_dtype
        self.coupling = BertCoupling()

        self.s1 = None
        self.s2 = None
            
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
            
        self.rev = ParePropReversibleSequential(
            modules=wrapped_blocks,
            s1=self.s1, s2=self.s2,
            autocast_dtype=autocast_dtype,
        )
        self._streams_initialized = False  # ★ フラグを追加
        # self.cached_rev_output: Optional[Tuple[Tensor, Tensor]] = None # ★ Body(Rev)の出力Yをキャッシュ
            
    def _init_streams_if_needed(self):
        """Streamを遅延初期化"""
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.rev.s1 = torch.cuda.Stream(device=device)
                self.rev.s2 = torch.cuda.Stream(device=device)
                self._streams_initialized = True
                
    def forward(self, input_ids, token_type_ids=None):
        # Pareprop の場合のみストリーム初期化
        if self.is_pareprop:
            self._init_streams_if_needed()

        # --- Non-Reversible Part (enable_grad) ---
        with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
            x = self.embeddings(input_ids, token_type_ids)
            # 常にグラフ付きで保存
            x_rev_in = tuple(t.detach() for t in self.coupling(x))
        
        # self.rev.forward は内部で no_grad を使う
        x_rev_out = self.rev(x_rev_in)
        
        return x_rev_out
    
    # 直列的なsequential_backward を実装
    def sequential_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor, # Body(Rev)の出力勾配 dY (from Stage 1)
        original_input_ids: Tensor, # backward_one_chunk から受け取る
        original_token_type_ids: Optional[Tensor] = None # backward_one_chunk から受け取る
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]: # (入力勾配 dX_input), (復元入力 X_input)
        
        # 1. Body(Rev) の逐次逆伝播を実行 -> dX_rev (Coupling出力勾配), X_rev (復元されたCoupling出力)
        dX_rev, X_rev = self.rev.sequential_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )

        # グラフを保持して非可逆部分(Embedding+Coupling)を再計算
        with torch.enable_grad(): # backward中は no_grad コンテキストの場合があるので, enable_grad() で囲む
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                
                # original_input_ids は requires_grad=False だが、
                # 途中の self.embeddings.parameters() が requires_grad=True
                x = self.embeddings(original_input_ids, original_token_type_ids)
                coupled_out = self.coupling(x)

            # dX_rev (可逆ブロックの入力勾配) を使って、
            # coupled_out (非可逆ブロックの出力) から autograd を実行
            # これにより self.embeddings の .grad が計算される
            torch.autograd.backward(coupled_out, grad_tensors=dX_rev)

        # input_ids など requires_grad=False の入力の勾配は None になる
        grads_input = (
             original_input_ids.grad if original_input_ids.grad is not None else None,
             original_token_type_ids.grad if original_token_type_ids is not None and original_token_type_ids.grad is not None else None
        )
            
        # 3. 結果を返す
        # grads_input: Embedding 層の入力に対する勾配
        # input_values: 復元された入力 = 元の入力
        return grads_input, (original_input_ids, original_token_type_ids)

    # pareprop_backward を実装
    def pareprop_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor, # Body(Rev)の出力勾配 dY (from Stage 1)
        original_input_ids: Tensor,
        original_token_type_ids: Optional[Tensor] = None
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]: # (入力勾配 dX_input), (復元入力 X_input)
        self._init_streams_if_needed()
        
        # 1. Body(Rev) の Pareprop 逆伝播を実行 -> dX_rev, X_rev
        # 1. Body(Rev) の Pareprop 逆伝播を実行 -> dX_rev, X_rev
        dX_rev, X_rev = self.rev.pareprop_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )
        
        # グラフを保持して非可逆部分(Embedding+Coupling)を再計算
        with torch.enable_grad(): # backward中は no_grad コンテキストの場合があるので, enable_grad() で囲む
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                
                # original_input_ids は requires_grad=False だが、
                # 途中の self.embeddings.parameters() が requires_grad=True
                x = self.embeddings(original_input_ids, original_token_type_ids)
                coupled_out = self.coupling(x)

            # dX_rev (可逆ブロックの入力勾配) を使って、
            # coupled_out (非可逆ブロックの出力) から autograd を実行
            # これにより self.embeddings の .grad が計算される
            torch.autograd.backward(coupled_out, grad_tensors=dX_rev)

        # input_ids など requires_grad=False の入力の勾配は None になる
        grads_input = (
             original_input_ids.grad if original_input_ids.grad is not None else None,
             original_token_type_ids.grad if original_token_type_ids is not None and original_token_type_ids.grad is not None else None
        )      

        # 3. 結果を返す
        return grads_input, (original_input_ids, original_token_type_ids)

class BertMidStage(nn.Module):
    """パイプラインの中間ステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        enable_amp: bool = False,
        # --- 2. フレームワーク用の引数を追加 ---
        is_pareprop: bool = False,
        act_buf: dict | None = None,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__() # 3. 親クラス
        self.is_pareprop = is_pareprop
        self.autocast_dtype = autocast_dtype
        
        self.s1 = None
        self.s2 = None
            
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
        self.mod = ParePropReversibleSequential(
            modules=wrapped_blocks,
            s1=self.s1, s2=self.s2,
            autocast_dtype=autocast_dtype,
        )
        self._streams_initialized = False
            
    def _init_streams_if_needed(self):
        """Streamを遅延初期化"""
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.mod.s1 = torch.cuda.Stream(device=device)
                self.mod.s2 = torch.cuda.Stream(device=device)
                self._streams_initialized = True
                 
    def forward(self, actv2: torch.Tensor, actv1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_pareprop:
            self._init_streams_if_needed() 
        
        output = self.mod((actv2, actv1))
        return output
    
    def pareprop_backward(
            self, 
            Y_2: torch.Tensor, 
            Y_1: torch.Tensor, 
            dY_2: torch.Tensor, 
            dY_1: torch.Tensor
        ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        _RevPipelineStageBase から呼び出される Pareprop 逆伝播
        """
        self._init_streams_if_needed() # 逆伝播の前にストリームを初期化
        return self.mod.pareprop_backward(Y_2, Y_1, dY_2, dY_1)

    # sequential_backward (逐次)
    def sequential_backward(
        self, 
        Y_2: torch.Tensor, 
        Y_1: torch.Tensor, 
        dY_2: torch.Tensor, 
        dY_1: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # こちらはストリームを使わないので _init_streams_if_needed は不要
        return self.mod.sequential_backward(Y_2, Y_1, dY_2, dY_1)   
    
    
class BertLastStage(nn.Module):
    """パイプラインの最終ステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        num_classes: int = 2,
        enable_amp: bool = False,
        is_pareprop: bool = False,
        act_buf: dict | None = None,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.is_pareprop = is_pareprop
        self.autocast_dtype = autocast_dtype
        self.s1 = None
        self.s2 = None        
        # Streamは遅延初期化するので、ここでは作らない
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
            
        self.mod = ParePropReversibleSequential(
            modules=wrapped_blocks, 
            s1=self.s1,
            s2=self.s2,
            autocast_dtype=autocast_dtype,
        )
        self._streams_initialized = False 
        
        self.decoupling = BertDecoupling()
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.head = nn.Linear(2 * hidden_size, num_classes)
        # 逆伝播用に Body の出力をキャッシュする変数
        self.cached_y_body: Optional[Tuple[Tensor, Tensor]] = None
        self.cached_y_body_detached: Optional[Tuple[Tensor, Tensor]] = None
        # 逆伝播用にグラフを持つ非リバーシブル部分の出力を保持する変数
        self.non_rev_out_with_grad: Optional[Tensor] = None
    
    def _init_streams_if_needed(self):
        """Streamを遅延初期化"""
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.mod.s1 = torch.cuda.Stream(device=device)  
                self.mod.s2 = torch.cuda.Stream(device=device) 
                self._streams_initialized = True
    
    def forward(self, actv2: Tensor, actv1: Tensor) -> Tensor:
        if self.is_pareprop:
            self._init_streams_if_needed()

        x_body_in = (actv2, actv1)
        x_body_out = self.mod(x_body_in) # Y_body を計算
        self.cached_y_body = tuple(t.detach() for t in x_body_out)

        # --- Non-Reversible Part (enable_grad) ---
        # Y_body をグラフに接続し直し、その参照もキャッシュ
        self.cached_y_body_detached = tuple(t.detach().requires_grad_(True) for t in self.cached_y_body)
        with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
            x = self.decoupling(self.cached_y_body_detached) # キャッシュした参照を使う
            cls_output = x[:, 0]
            x = self.norm(cls_output)
            final_output = self.head(x)
        # グラフ付きの最終出力をキャッシュ
        self.non_rev_out_with_grad = final_output

        return final_output

    def sequential_backward(
        self,
        loss_tensor: Optional[Tensor],
        # reconstructed_input_values: Tuple[Tensor, ...] # LastStageでは不要
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:

        # 1. Non-Reversible Part の勾配計算 (autograd)
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        if self.non_rev_out_with_grad is not None and self.cached_y_body_detached is not None and loss_tensor is not None:
            # backward を実行
            loss_tensor.backward()
            # キャッシュした参照から .grad を取得
            dY_body = tuple(y.grad for y in self.cached_y_body_detached)

            # 使用済みキャッシュをクリア
            self.non_rev_out_with_grad = None
            self.cached_y_body_detached = None
        else:
             raise RuntimeError("Required tensors for backward were not cached during forward.")

        # フォールバック (もし .grad が None だった場合 - 通常は起こらないはず)
        if dY_body is None or any(g is None for g in dY_body):
             print("Warning: Failed to get gradients for Y_body via autograd. Using zeros.")
             if self.cached_y_body is None: raise RuntimeError("Y_body value cache missing.")
             dY_body = tuple(torch.zeros_like(t) for t in self.cached_y_body) # 仮

        # 2. Body(Rev) の逐次逆伝播を実行
        if self.cached_y_body is None:
             raise RuntimeError("Y_body value cache missing.")
        grads_input, input_values = self.mod.sequential_backward(
            *self.cached_y_body, *dY_body
        )
        self.cached_y_body = None 

        return grads_input, input_values

    def pareprop_backward(
        self,
        loss_tensor: Optional[Tensor],
        # reconstructed_input_values: Tuple[Tensor, ...] # LastStageでは不要
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        self._init_streams_if_needed()

        # 1. Non-Reversible Part の勾配計算 (autograd)
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        if self.non_rev_out_with_grad is not None and self.cached_y_body_detached is not None and loss_tensor is not None:
            # backward を実行
            loss_tensor.backward()
            # キャッシュした参照から .grad を取得
            dY_body = tuple(y.grad for y in self.cached_y_body_detached)

            # 使用済みキャッシュをクリア
            self.non_rev_out_with_grad = None
            self.cached_y_body_detached = None
        else:
             raise RuntimeError("Required tensors for backward were not cached during forward.")

        # フォールバック
        if dY_body is None or any(g is None for g in dY_body):
             print("Warning: Failed to get gradients for Y_body via autograd. Using zeros.")
             if self.cached_y_body is None: raise RuntimeError("Y_body value cache missing.")
             dY_body = tuple(torch.zeros_like(t) for t in self.cached_y_body) # 仮

        # 2. Body(Rev) の Pareprop 逆伝播を実行
        if self.cached_y_body is None:
             raise RuntimeError("Y_body value cache missing.")
        grads_input, input_values = self.mod.pareprop_backward(
            *self.cached_y_body, *dY_body
        )
        self.cached_y_body = None 

        return grads_input, input_values