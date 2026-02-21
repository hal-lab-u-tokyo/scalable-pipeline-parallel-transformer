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
        return x2 + x1
    
# --- 共通の重み初期化用Mixin ---
class WeightInitMixin:
    def _init_weights(self, module):
        """lrdebug.text に準拠した重み初期化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

# --- BertFullStage (完全版) ---
class BertFullStage(nn.Module, WeightInitMixin): # ★ RevModuleBase 継承を削除
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
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)
    
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
        pooled_output = x.mean(dim=1) # Get the first token's output
        x = self.norm(pooled_output)
        x = self.head(x)
        
        return x

class BertFirstStage(nn.Module, WeightInitMixin):
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
        self.apply(self._init_weights)
            
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
            
        # ★★★ Input同期: s1/s2 は Main Stream の完了を待つ ★★★
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.rev.s1 is not None: self.rev.s1.wait_stream(current_stream)
            if self.rev.s2 is not None: self.rev.s2.wait_stream(current_stream)
            
        # self.rev.forward は内部で no_grad を使う
        x_rev_out = self.rev(x_rev_in)
        # ★★★ 追加: メインストリームで計算完了を待つ ★★★
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.rev.s1 is not None:
                current_stream.wait_stream(self.rev.s1)
            if self.rev.s2 is not None:
                current_stream.wait_stream(self.rev.s2)
        return x_rev_out
    
    # --- 共通のEmbedding逆伝播ロジック ---
    def _backward_embedding_recompute(self, dX_rev, original_input_ids, original_token_type_ids):
        """Embeddingを再計算して逆伝播を行う"""
        with torch.enable_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                # 1. Embedding再計算
                x = self.embeddings(original_input_ids, original_token_type_ids)
                # 2. Coupling再計算
                coupled_out = self.coupling(x)

            # 3. Autograd実行: Couplingの出力に対して dX_rev の勾配を流す
            torch.autograd.backward(coupled_out, grad_tensors=dX_rev)

    def sequential_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor,
        original_input_ids: Tensor,
        original_token_type_ids: Optional[Tensor] = None
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        
        dX_rev, X_rev = self.rev.sequential_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )

        # 【修正】再計算ロジックを使用
        self._backward_embedding_recompute(dX_rev, original_input_ids, original_token_type_ids)

        return (None, None), (original_input_ids, original_token_type_ids)

    def pareprop_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor,
        original_input_ids: Tensor,
        original_token_type_ids: Optional[Tensor] = None
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        self._init_streams_if_needed()
        
        dX_rev, X_rev = self.rev.pareprop_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )
        
        # ★★★ Backward Output同期: Embedding計算の前にストリーム完了を待つ ★★★
        current_stream = torch.cuda.current_stream()
        if self.rev.s1 is not None: current_stream.wait_stream(self.rev.s1)
        if self.rev.s2 is not None: current_stream.wait_stream(self.rev.s2)
        
        # 【修正】再計算ロジックを使用
        self._backward_embedding_recompute(dX_rev, original_input_ids, original_token_type_ids)

        return (None, None), (original_input_ids, original_token_type_ids)

class BertMidStage(nn.Module, WeightInitMixin):
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
        self.apply(self._init_weights)
        
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

        # ★★★ Input同期: s1/s2 は Main Stream (前段からの受信) の完了を待つ ★★★
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.mod.s1 is not None: self.mod.s1.wait_stream(current_stream)
            if self.mod.s2 is not None: self.mod.s2.wait_stream(current_stream)

            
        output = self.mod((actv2, actv1))
        # ★★★ 追加: メインストリームで計算完了を待つ ★★★
        if self.is_pareprop:
            # 現在のストリーム (Main Stream) を取得
            current_stream = torch.cuda.current_stream()
            # s1, s2 の処理が終わるまで Main Stream を待機させる
            if self.mod.s1 is not None:
                current_stream.wait_stream(self.mod.s1)
            if self.mod.s2 is not None:
                current_stream.wait_stream(self.mod.s2)
        
        
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
        out = self.mod.pareprop_backward(Y_2, Y_1, dY_2, dY_1)
        # ★★★ Backward Output同期: 計算結果をCPU/通信に渡す前に完了を待つ ★★★
        current_stream = torch.cuda.current_stream()
        if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
        if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)   
        return out

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
    
    
class BertLastStage(nn.Module, WeightInitMixin):
    """パイプラインの最終ステージ (修正版)"""
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
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)
    
    def _init_streams_if_needed(self):
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.mod.s1 = torch.cuda.Stream(device=device)  
                self.mod.s2 = torch.cuda.Stream(device=device) 
                self._streams_initialized = True
                
    def forward_features(self, actv2: Tensor, actv1: Tensor):
        if self.is_pareprop:
            self._init_streams_if_needed()

        x_body_in = (actv2, actv1)

        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.mod.s1 is not None: self.mod.s1.wait_stream(current_stream)
            if self.mod.s2 is not None: self.mod.s2.wait_stream(current_stream)
            
        x_body_out = self.mod(x_body_in) 
        
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
            if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)
            
        return x_body_out
    
    def forward_head(self, x_body_out):
        # --- Non-Reversible Part (enable_grad) ---
        with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
            x = self.decoupling(x_body_out) 
            pooled_output = x.mean(dim=1)
            x = self.norm(pooled_output)
            final_output = self.head(x)       
        
        return final_output
    
    def forward(self, actv2: Tensor, actv1: Tensor) -> Tensor:
        x_body_out = self.forward_features(actv2=actv2, actv1=actv1)
        final_output = self.forward_head(x_body_out)
        return final_output

    def sequential_backward(
        self,
        loss_tensor: Optional[Tensor],
        cached_y_body,
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        """
        Sequential Backward (Stateless版)
        Args:
            loss_tensor: Headの出力から計算されたLoss (Backward起点)。
            cached_y_body: forward_featuresの出力であり、forward_headの入力として使われたTensor群。
                           Headの逆伝播を受け取るために、forward_head実行時にrequires_grad=Trueであったもの。
        """
        # 1. Non-Reversible Part (Head) の勾配計算
        # Headへの入力(cached_y_body)に対する勾配 dY_body を取得する
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        
        if loss_tensor is not None:
            # Headの逆伝播を実行し、cached_y_body.grad に勾配を蓄積させる
            loss_tensor.backward()
            
            # 蓄積された勾配を取り出す
            dY_body = tuple(y.grad for y in cached_y_body)
            
            # Stateless性を保つため、使用した勾配はクリアしておく（推奨）
            for y in cached_y_body:
                y.grad = None
        else:
             raise RuntimeError("loss_tensor is None. Cannot perform backward.")

        # フォールバック: 万が一勾配がなければゼロ埋め
        if dY_body is None or any(g is None for g in dY_body):
             # logger.warning("dY_body contains None, falling back to zeros.")
             dY_body = tuple(torch.zeros_like(t) if g is None else g for t, g in zip(cached_y_body, dY_body))

        # 2. Body (Reversible Part) の逐次逆伝播を実行
        # cached_y_bodyは値としてはBackboneの出力と同じなので、入力再構成(Inverse)に利用できる
        grads_input, input_values = self.mod.sequential_backward(
            *cached_y_body, *dY_body
        )

        return grads_input, input_values

    def pareprop_backward(
        self,
        loss_tensor: Optional[Tensor],
        cached_y_body,
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        """
        PareProp Backward (Stateless版)
        """
        self._init_streams_if_needed()
        
        # 1. Non-Reversible Part (Head) の勾配計算
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        
        if loss_tensor is not None:
            loss_tensor.backward()
            dY_body = tuple(y.grad for y in cached_y_body)
            
            for y in cached_y_body:
                y.grad = None
        else:
             raise RuntimeError("loss_tensor is None. Cannot perform backward.")

        if dY_body is None or any(g is None for g in dY_body):
             dY_body = tuple(torch.zeros_like(t) if g is None else g for t, g in zip(cached_y_body, dY_body))

        # 2. Body (Reversible Part) の Pareprop 逆伝播を実行
        grads_input, input_values = self.mod.pareprop_backward(
            *cached_y_body, *dY_body
        )
        
        # Stream同期
        current_stream = torch.cuda.current_stream()
        if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
        if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)

        return grads_input, input_values