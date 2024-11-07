import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseSelfAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelwiseSelfAttention, self).__init__()
        self.dim = dim
        self.query_conv = nn.Linear(dim, dim)
        self.key_conv = nn.Linear(dim, dim)
        self.value_conv = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, dim))

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x + self.pos_embedding  # Positional embedding
        x = x.view(B, H * W, C)     # Reshape to (B, N, C)

        # Linear projections
        q = self.query_conv(x)      # (B, N, C)
        k = self.key_conv(x)        # (B, N, C)
        v = self.value_conv(x)      # (B, N, C)

        # Compute attention over channels at each spatial location
        q = q.view(B, H * W, 1, C)  # (B, N, 1, C)
        k = k.view(B, H * W, C, 1)  # (B, N, C, 1)
        attn = torch.matmul(q, k).squeeze(2) * self.scale  # (B, N, C)
        attn = attn.softmax(dim=-1)  # Softmax over channels

        # Apply attention to values
        out = attn * v               # Element-wise multiplication
        out = out.view(B, H, W, C)   # Reshape back to (B, H, W, C)
        return out
