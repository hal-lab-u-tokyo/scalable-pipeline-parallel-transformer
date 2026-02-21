# ファイル名: vit_stages.py

import torch
import torch.nn as nn
from .vit_blocks import ViTEmbeddings, ReversibleViTBlock
from typing import Tuple, Optional, Any
from torch import Tensor

from .reversible import (
    Coupling as BaseCoupling,
    Decoupling as BaseDecoupling,
    ParePropReversibleBlockWrapper,
    ParePropReversibleSequential
)

# --- ViTCoupling / Decoupling ---
# ロジックはBERT版と同じですが、名前をViTに合わせています
class ViTCoupling(BaseCoupling):
    """入力を2つに複製してリバーシブル形式に (X -> X, X)"""
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x 

class ViTDecoupling(BaseDecoupling):
    """リバーシブル形式から通常のテンソルに戻す (X1, X2 -> X1 + X2)"""
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x2, x1 = inputs
        return x2 + x1
    
# --- 共通の重み初期化用Mixin (Conv2d対応を追加) ---
class WeightInitMixin:
    def _init_weights(self, module):
        """lrdebug.text に準拠した重み初期化 + Conv2d(ViT用)"""
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
        # ViTのPatch Embedding用
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# --- ViTFullStage (完全版: 単一GPU用) ---
class ViTFullStage(nn.Module, WeightInitMixin):
    """パイプライン並列なしの完全なReversible ViTモデル"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 12, 
        num_classes: int = 1000, # ImageNet default
        enable_amp: bool = False,
        is_pareprop: bool = False,
        act_buf: dict | None = None,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        
        self.is_pareprop = is_pareprop
        self.autocast_dtype = autocast_dtype
        
        # 1. Embeddings (Image processing)
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size
        )
        
        # 2. Coupling
        self.coupling = ViTCoupling()
        
        # 3. Reversible Body
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.s1 = torch.cuda.Stream(device=device)
        self.s2 = torch.cuda.Stream(device=device)
        
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleViTBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
        
        self.body = ParePropReversibleSequential(
            modules=wrapped_blocks,
            s1=self.s1, s2=self.s2,
            autocast_dtype=autocast_dtype,
        )
        
        # 4. Decoupling
        self.decoupling = ViTDecoupling()
        
        # 5. Final Layers
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)
    
    def forward(self, x):
        """Standard ViT forward pass"""
        # x: (B, C, H, W)
        
        # 1. Embeddings
        x = self.embeddings(x)
        
        # 2. Reversible Format
        x = self.coupling(x)
        
        # 3. Body
        x = self.body(x)
        
        # 4. Decoupling
        x = self.decoupling(x)
        
        # 5. Head (CLS Token)
        cls_output = x[:, 0] # ViT uses CLS token at index 0
        x = self.norm(cls_output)
        x = self.head(x)
        
        return x

# --- Pipeline Stages ---

class ViTFirstStage(nn.Module, WeightInitMixin):
    """パイプラインの最初のステージ (画像入力)"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        enable_amp: bool = False,
        is_pareprop: bool = False,
        act_buf: dict | None = None,
        autocast_dtype: torch.dtype = torch.float32,
        debug_context: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size
        )
        self.is_pareprop = is_pareprop
        self.autocast_dtype = autocast_dtype
        self.coupling = ViTCoupling()

        self.s1 = None
        self.s2 = None
            
        wrapped_blocks = [
            ParePropReversibleBlockWrapper(
                ReversibleViTBlock(hidden_size, num_attention_heads, enable_amp)
            )
            for _ in range(num_blocks)
        ]
            
        self.rev = ParePropReversibleSequential(
            modules=wrapped_blocks,
            s1=self.s1, s2=self.s2,
            autocast_dtype=autocast_dtype,
        )
        self._streams_initialized = False
        self.apply(self._init_weights)
            
    def _init_streams_if_needed(self):
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.rev.s1 = torch.cuda.Stream(device=device)
                self.rev.s2 = torch.cuda.Stream(device=device)
                self._streams_initialized = True
                
    def forward(self, x):
        # x: (B, C, H, W)
        if self.is_pareprop:
            self._init_streams_if_needed()

        # Non-Reversible Part (Embedding)
        with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
            x = self.embeddings(x)
            x_rev_in = tuple(t.detach() for t in self.coupling(x))
            
        # Input同期
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.rev.s1 is not None: self.rev.s1.wait_stream(current_stream)
            if self.rev.s2 is not None: self.rev.s2.wait_stream(current_stream)
            
        x_rev_out = self.rev(x_rev_in)

        # Output同期
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.rev.s1 is not None: current_stream.wait_stream(self.rev.s1)
            if self.rev.s2 is not None: current_stream.wait_stream(self.rev.s2)
            
        return x_rev_out
    
    # --- Embedding逆伝播ロジック (画像版) ---
    def _backward_embedding_recompute(self, dX_rev, original_images):
        """Embeddingを再計算して逆伝播を行う"""
        with torch.enable_grad():
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                # 1. Embedding再計算
                x = self.embeddings(original_images)
                # 2. Coupling再計算
                coupled_out = self.coupling(x)

            # 3. Autograd実行
            torch.autograd.backward(coupled_out, grad_tensors=dX_rev)

    def sequential_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor,
        original_images: Tensor, # Input IDs -> Images
        **kwargs # token_type_idsなどは不要なので吸収
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        
        dX_rev, X_rev = self.rev.sequential_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )

        self._backward_embedding_recompute(dX_rev, original_images)

        #return (None, None), (original_images,)
        return (None, None), (None,)


    def pareprop_backward(
        self,
        Y_rev_out_2: Tensor, Y_rev_out_1: Tensor,
        dY_rev_out_2: Tensor, dY_rev_out_1: Tensor,
        original_images: Tensor,
        **kwargs
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        self._init_streams_if_needed()
        
        dX_rev, X_rev = self.rev.pareprop_backward(
            Y_rev_out_2, Y_rev_out_1, dY_rev_out_2, dY_rev_out_1
        )
        
        # Stream同期
        current_stream = torch.cuda.current_stream()
        if self.rev.s1 is not None: current_stream.wait_stream(self.rev.s1)
        if self.rev.s2 is not None: current_stream.wait_stream(self.rev.s2)
        
        self._backward_embedding_recompute(dX_rev, original_images)

        #return (None, None), (original_images,)
        return (None, None), (None,)

class ViTMidStage(nn.Module, WeightInitMixin):
    """パイプラインの中間ステージ (BertMidStageとほぼ同じロジック)"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
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
                ReversibleViTBlock(hidden_size, num_attention_heads, enable_amp)
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
        if self.is_pareprop and not self._streams_initialized:
            device = next(self.parameters()).device
            if device.type == 'cuda':
                self.mod.s1 = torch.cuda.Stream(device=device)
                self.mod.s2 = torch.cuda.Stream(device=device)
                self._streams_initialized = True
                 
    def forward(self, actv2: torch.Tensor, actv1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_pareprop:
            self._init_streams_if_needed()

        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.mod.s1 is not None: self.mod.s1.wait_stream(current_stream)
            if self.mod.s2 is not None: self.mod.s2.wait_stream(current_stream)

        output = self.mod((actv2, actv1))
        
        if self.is_pareprop:
            current_stream = torch.cuda.current_stream()
            if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
            if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)
        
        return output
    
    def pareprop_backward(self, Y_2, Y_1, dY_2, dY_1):
        self._init_streams_if_needed()
        out = self.mod.pareprop_backward(Y_2, Y_1, dY_2, dY_1)
        current_stream = torch.cuda.current_stream()
        if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
        if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)   
        return out

    def sequential_backward(self, Y_2, Y_1, dY_2, dY_1):
        return self.mod.sequential_backward(Y_2, Y_1, dY_2, dY_1)   
    
    
class ViTLastStage(nn.Module, WeightInitMixin):
    """パイプラインの最終ステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        num_classes: int = 1000,
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
                ReversibleViTBlock(hidden_size, num_attention_heads, enable_amp)
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
        
        self.decoupling = ViTDecoupling()
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
        # --- Non-Reversible Part (Head) ---
        with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
            x = self.decoupling(x_body_out) 
            # CLS token Extraction
            cls_output = x[:, 0]
            x = self.norm(cls_output)
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
        
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        
        if loss_tensor is not None:
            loss_tensor.backward()
            dY_body = tuple(y.grad for y in cached_y_body)
            for y in cached_y_body:
                y.grad = None
        else:
             raise RuntimeError("loss_tensor is None")

        if dY_body is None or any(g is None for g in dY_body):
             dY_body = tuple(torch.zeros_like(t) if g is None else g for t, g in zip(cached_y_body, dY_body))

        grads_input, input_values = self.mod.sequential_backward(
            *cached_y_body, *dY_body
        )

        return grads_input, input_values

    def pareprop_backward(
        self,
        loss_tensor: Optional[Tensor],
        cached_y_body,
    ) -> tuple[tuple[Optional[Tensor], ...], tuple[Tensor, ...]]:
        
        self._init_streams_if_needed()
        
        dY_body: Optional[Tuple[Optional[Tensor], ...]] = None
        
        if loss_tensor is not None:
            loss_tensor.backward()
            dY_body = tuple(y.grad for y in cached_y_body)
            for y in cached_y_body:
                y.grad = None
        else:
             raise RuntimeError("loss_tensor is None")

        if dY_body is None or any(g is None for g in dY_body):
             dY_body = tuple(torch.zeros_like(t) if g is None else g for t, g in zip(cached_y_body, dY_body))

        grads_input, input_values = self.mod.pareprop_backward(
            *cached_y_body, *dY_body
        )
        
        current_stream = torch.cuda.current_stream()
        if self.mod.s1 is not None: current_stream.wait_stream(self.mod.s1)
        if self.mod.s2 is not None: current_stream.wait_stream(self.mod.s2)

        return grads_input, input_values