# model/meta_vit.py
import torch
import torch.nn as nn
from model.spec_vit import TransformerEncoderLayer

class MetaViT(nn.Module):
    def __init__(self, spectrum_points=300, num_parameters=5, embed_dim=128, num_heads=4, depth=3):
        super().__init__()
        self.input_projection = nn.Linear(spectrum_points, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2, embed_dim))
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*2)
            for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_parameters)
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