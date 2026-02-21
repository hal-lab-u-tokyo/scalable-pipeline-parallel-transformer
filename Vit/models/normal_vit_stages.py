import torch
import torch.nn as nn
from .vit_blocks import ViTEmbeddings, NormalViTBlock  # 先ほどのファイルをimport
from typing import Tuple, Optional, Any
from torch import Tensor

class WeightInitMixin:
    def _init_weights(self, module):
        """lrdebug.text に準拠した重み初期化（ViT/BERT共通）"""
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
        # Conv2d (ViTのパッチ埋め込み) の初期化を追加
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class NormalViTFullStage(nn.Module, WeightInitMixin):
    """単一GPU等で動作する通常のFull ViT"""
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
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.autocast_dtype = autocast_dtype
        
        # 1. Embeddings (Image -> Patches + PosEmbed)
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size
        )
        
        # 2. Body (Normal ViT Blocks)
        self.body = nn.Sequential(
            *[NormalViTBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
        )
        
        # 3. Final Layers (Normalization and Classification Head)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        
        self.apply(self._init_weights)
    
    def forward(self, x):
        """Standard ViT forward pass"""
        
        # 1. Apply embeddings (B, C, H, W) -> (B, N, D)
        x = self.embeddings(x)
        
        # 2. Pass through blocks
        x = self.body(x)
        
        # 3. Extract CLS token output (index 0) and classify
        cls_output = x[:, 0]
        x = self.norm(cls_output)
        x = self.head(x)
        
        return x

class NormalViTFirstStage(nn.Module, WeightInitMixin):
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
        autocast_dtype: torch.dtype = torch.float32,
        debug_context: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.autocast_dtype = autocast_dtype
        
        self.embeddings = ViTEmbeddings(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=hidden_size
        )

        self.body = nn.Sequential(
            *[NormalViTBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
        )
        self.apply(self._init_weights)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.embeddings(x)
        x = self.body(x)
        return x

class NormalViTMidStage(nn.Module, WeightInitMixin):
    """パイプラインの中間ステージ (入力も出力も隠れ層の状態)"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        enable_amp: bool = False,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__() 
        self.autocast_dtype = autocast_dtype
        
        self.mod = nn.Sequential(
            *[NormalViTBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
        )
        self.apply(self._init_weights)
                 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return self.mod(x)
    
class NormalViTLastStage(nn.Module, WeightInitMixin):
    """パイプラインの最終ステージ (分類ヘッドあり)"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        num_classes: int = 1000,
        enable_amp: bool = False,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.autocast_dtype = autocast_dtype    
            
        self.mod = nn.Sequential(
            *[NormalViTBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> Tensor:
        # x: (B, N, D)
        x = self.mod(x)
        
        # Extract CLS token
        cls_output = x[:, 0]
        
        x = self.norm(cls_output)
        x = self.head(x)
        return x