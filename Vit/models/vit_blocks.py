import torch
from torch import nn
from torch.nn import MultiheadAttention as MHA

# --------------------------------------------------------
# 1. ViT Embeddings (ここが主要な変更点)
# --------------------------------------------------------

class ViTEmbeddings(nn.Module):
    """
    画像をパッチに分割し、位置埋め込みとCLSトークンを付加するクラス
    Input: (B, C, H, W) -> Output: (B, N_patches + 1, Embed_Dim)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, dropout=0.0):
        super().__init__()
        
        # 画像サイズとパッチサイズからパッチ数を計算
        # 例: 224x224, patch=16 -> 14x14 = 196 patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # パッチ分割と線形射影を同時に行うConv2d
        # カーネルサイズとストライドをpatch_sizeにすることで、重複なしのパッチ切り出しと同じになる
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLSトークン (学習可能なパラメータ)
        # バッチサイズに合わせてexpandして結合する
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置埋め込み (パッチ数 + CLSトークン分)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # 初期化（timmやJAXの実装に合わせるのが一般的ですが、ここではPyTorch標準で）
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 1. Patch Partition & Linear Projection
        # (B, Embed_Dim, Grid, Grid) -> (B, Embed_Dim, Num_Patches) -> (B, Num_Patches, Embed_Dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        
        # 2. Append CLS Token
        # (1, 1, Embed_Dim) -> (B, 1, Embed_Dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add Position Embedding
        x = x + self.pos_embed
        
        # 4. Norm & Dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x

# --------------------------------------------------------
# 2. Sub-blocks (Bertのものと構造は同じでOK)
# --------------------------------------------------------

class MLPSubblock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, enable_amp=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
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
            # ViTでもSelf-Attentionの計算は同じ
            out, _ = self.attn(x, x, x)
        return out

# --------------------------------------------------------
# 3. Reversible ViT Block
# --------------------------------------------------------

class ReversibleViTBlock(nn.Module):
    """
    ViT用のReversibleBlock。
    構造自体はBert版と同じですが、AttentionとMLPの呼び出し順序や構成を
    ViTの論文(Dosovitskiy et al.)やtimm実装に近づけることも可能です。
    今回はReversible構造(Additive Coupling)を維持することを最優先しています。
    """
    def __init__(self, dim, num_heads, enable_amp):
        super().__init__()
        # Reversibleの場合、入力チャネルは2分割されるため、
        # ここでの dim は「分割後の次元（全体の1/2）」を指すことが多いです。
        # 宮城さんの実装に合わせて調整してください。
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)
        
    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播: (X_2, X_1) -> (Y_2, Y_1)
        Additive Coupling:
        Y_1 = X_1 + F(X_2)
        Y_2 = X_2 + G(Y_1)
        """
        X_2, X_1 = inputs
        
        # Attention Path
        f_X_2 = self.F(X_2)
        Y_1 = X_1 + f_X_2
        
        # MLP Path
        g_Y_1 = self.G(Y_1)
        Y_2 = X_2 + g_Y_1
        
        return Y_2, Y_1
    
    def inverse(self, outputs: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        逆計算: (Y_2, Y_1) -> (X_2, X_1)
        """
        Y_2, Y_1 = outputs
        
        with torch.no_grad():
            # X_2 = Y_2 - G(Y_1)
            g_Y_1 = self.G(Y_1)
            X_2 = Y_2 - g_Y_1
            
            # X_1 = Y_1 - F(X_2)
            f_X_2 = self.F(X_2)
            X_1 = Y_1 - f_X_2
        
        return X_2, X_1

# --------------------------------------------------------
# 4. Normal ViT Block (Reference)
# --------------------------------------------------------

class NormalViTBlock(nn.Module):
    def __init__(self, dim, num_heads, enable_amp):
        super().__init__()
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads, enable_amp=enable_amp)
        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard Residual Connections
        x = x + self.F(x)
        x = x + self.G(x)
        return x