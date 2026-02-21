import torch
import torch.nn as nn
from .bert_blocks import BertEmbeddings, ReversibleBertBlock
from typing import Tuple, Optional, Any # ★ 追加
from torch import Tensor # ★ 追加
from torch.cuda.amp import autocast # ★ 追加

from .reversible import (
    Coupling as BaseCoupling,
    Decoupling as BaseDecoupling,
)

class BertCoupling(BaseCoupling):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x
    
class BertDecoupling(BaseDecoupling):
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x2, x1 = inputs
        return torch.cat([x1, x2], dim=-1)
    
class BertFullStageWithInverse(nn.Module):
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
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs, # Allows extra arguments
    ):
        super().__init__()
        self.autocast_dtype = autocast_dtype # For automatic mixed precision
        
        # 1. Embeddings (Input processing)
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size
        )
        
        # 2. Coupling (Convert to reversible format)
        self.coupling = BertCoupling()
        
        # 3. Reversible Body (Main transformer blocks)
        # Both is_pareprop=True/False now use the same structure

        # Wrap each ReversibleBertBlock with ParePropReversibleBlockWrapper
        self.body = (
            nn.Sequential(
                *[ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
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
    
    
    
class BertFirstStageWithInverse(nn.Module):
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
        autocast_dtype: torch.dtype = torch.float32,
        debug_context: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size
        )
        self.autocast_dtype = autocast_dtype
        self.coupling = BertCoupling()

        self.rev = (
            nn.Sequential(
                *[ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
            
    # ★★★ forward を変更: is_pareprop の分岐を削除 ★★★
    def forward(self, input_ids, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        x = self.coupling(x)
        x = self.rev(x)
        
        return x

class BertMidStageWithInverse(nn.Module): # 1. 継承
    """パイプラインの中間ステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        enable_amp: bool = False,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__() # 3. 親クラス
        self.autocast_dtype = autocast_dtype
        
        self.mod = (
            nn.Sequential(
                *[ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
                 
    def forward(self, actv2: torch.Tensor, actv1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mod((actv2, actv1))
    
    def inverse(self, actv3: torch.Tensor, actv2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (actv3, actv2)
        for block in reversed(self.mod):
            x = block.inverse(x)
        return x 

class BertLastStageWithInverse(nn.Module):
    """パイプラインの最終ステージ"""
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_blocks: int = 3,
        num_classes: int = 2,
        enable_amp: bool = False,
        autocast_dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.autocast_dtype = autocast_dtype    
        # ★ Streamは遅延初期化するので、ここでは作らない
            
        self.mod = (
            nn.Sequential(
                *[ReversibleBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
        self.decoupling = BertDecoupling()
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.head = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, actv2: Tensor, actv1: Tensor) -> Tensor:
        x = (actv2, actv1)
        x = self.mod(x)
        x = self.decoupling(x)
        cls_output = x[:, 0]
        x = self.norm(cls_output)
        x = self.head(x)
        return x