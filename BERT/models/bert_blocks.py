import torch
from torch import nn
from torch.nn import MultiheadAttention as MHA

#Bertブロックのそれぞれを実装
#backwardは書かずにtorchの自動微分機能に任せる
#中間結果を保存しないことなどはpipelining/reversible_stages.pyに任せる

class BertEmbeddings(nn.Module):
    """あなたのBertEmbeddingsをそのまま使用"""
    def __init__(self, vocab_size, embed_dim, max_position_embeddings, type_vocab_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embed_dim)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.norm(embeddings)
        return embeddings


class MLPSubblock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, enable_amp=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.enable_amp = enable_amp
        
    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=self.enable_amp):
            return self.mlp(self.norm(x))
        
        
class AttentionSubBlock(nn.Module):

    def __init__(self, dim, num_heads, enable_amp=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MHA(dim, num_heads, batch_first=True)
        self.enable_amp = enable_amp

    def forward(self, x):
        with torch.amp.autocast(device_type="cuda", enabled=self.enable_amp):
            x = self.norm(x)
            out, _ = self.attn(x, x, x)

        return out


# class AttentionSubBlock(nn.Module):
#     def __init__(self, dim, num_heads, enable_amp=False):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
#         # PyTorch 2.0+ の nn.MultiheadAttention を想定
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.enable_amp = enable_amp
        
#     def forward(self, x):
#         # AMP (自動混合精度) の適用
#         with torch.amp.autocast(device_type="cuda", enabled=self.enable_amp):
#             x_norm = self.norm(x)

#             with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
#                 out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
                
#             return out


# inverse()メソッドを持つReversibleBlock
class ReversibleBertBlock(nn.Module):
    """
    宮城さんプロジェクトのフレームワークに適合させたReversibleBlock
    
    順伝播: (X_1, X_2) → (Y_1 = X_1 + F(X_2), Y_2 = X_2 + G(Y_1))
    逆計算: (Y_1, Y_2) → (X_1, X_2)
    """
    def __init__(self, dim, num_heads, enable_amp):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)
        
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        宮城さんプロジェクトの形式に合わせる
        inputs: (X_2, X_1) の順で受け取る（AdditiveCouplingと同じ形式）
        """
        X_2, X_1 = inputs
        
        # Y_1 = X_1 + F(X_2)
        f_X_2 = self.F(X_2)
        Y_1 = X_1 + f_X_2
        # Y_2 = X_2 + G(Y_1)
        g_Y_1 = self.G(Y_1)
        Y_2 = X_2 + g_Y_1
        
        
        return Y_2, Y_1  # (Y_2, Y_1)の順で返す
    
    def inverse(self, outputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        逆計算: (Y_2, Y_1) → (X_2, X_1)
        活性化を復元するだけで、勾配は計算しない
        """
        Y_2, Y_1 = outputs
        
        with torch.no_grad():
            # X_2 = Y_2 - G(Y_1)
            g_Y_1 = self.G(Y_1)
            X_2 = Y_2 - g_Y_1
            
            # X_1 = Y_1 - F(X_2)
            f_X_2 = self.F(X_2)
            X_1 = Y_1 - f_X_2
        
        return X_2, X_1  # (X_2, X_1)の順で返す
    
class NormalBertBlock(nn.Module):

    def __init__(self, dim, num_heads, enable_amp):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.F(x) + x
        x = self.G(x) + x
        return x
        
        