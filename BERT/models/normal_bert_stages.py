import torch
import torch.nn as nn
from .bert_blocks import BertEmbeddings, NormalBertBlock
from typing import Tuple, Optional, Any # ★ 追加
from torch import Tensor # ★ 追加
from torch.cuda.amp import autocast # ★ 追加

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
            
class NormalBertFullStage(nn.Module, WeightInitMixin):
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
        
        # 3. Reversible Body (Main transformer blocks)
        # Both is_pareprop=True/False now use the same structure

        # Wrap each ReversibleBertBlock with ParePropReversibleBlockWrapper
        self.body = (
            nn.Sequential(
                *[NormalBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
        
        # 5. Final Layers (Normalization and Classification Head)
        # LayerNorm input size is 2 * hidden_size because decoupling concatenates
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)
    
    def forward(self, input_ids, token_type_ids=None):
        """Standard BERT forward pass"""
        
        # 1. Apply embeddings
        x = self.embeddings(input_ids, token_type_ids)
        
        # 3. Pass through reversible blocks
        # This calls ParePropReversibleSequential.forward, which runs in no_grad
        x = self.body(x)
        
        # 5. Extract CLS token output and apply final layers
        cls_output = x[:, 0] # Get the first token's output
        x = self.norm(cls_output)
        x = self.head(x)
        
        return x
    
class NormalBertFirstStage(nn.Module, WeightInitMixin):
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


        self.rev = (
            nn.Sequential(
                *[NormalBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
        self.apply(self._init_weights)
        
    def forward(self, input_ids, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        x = self.rev(x)
        
        return x

class NormalBertMidStage(nn.Module, WeightInitMixin):
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
        super().__init__() 
        self.autocast_dtype = autocast_dtype
        
        self.mod = (
            nn.Sequential(
                *[NormalBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )
        self.apply(self._init_weights)
                 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x)
    
class NormalBertLastStage(nn.Module, WeightInitMixin):
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
        # Streamは遅延初期化するので、ここでは作らない
            
        self.mod = (
            nn.Sequential(
                *[NormalBertBlock(hidden_size, num_attention_heads, enable_amp) for _ in range(num_blocks)]
            )
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.mod(x)
        cls_output = x[:, 0]
        x = self.norm(cls_output)
        x = self.head(x)
        return x