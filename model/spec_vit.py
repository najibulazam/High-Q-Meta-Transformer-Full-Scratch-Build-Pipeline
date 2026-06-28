# model/spec_vit.py
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + src2
        return src

class SpecViT(nn.Module):
    def __init__(self, num_parameters=5, spectrum_points=1000, embed_dim=64, num_heads=8, depth=6, dim_feedforward=128):
        super().__init__()
        self.input_projection = nn.Linear(num_parameters, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2, embed_dim))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=dim_feedforward)
            for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, spectrum_points),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        for layer in self.layers:
            x = layer(x)
            
        return self.mlp_head(x[:, 0])