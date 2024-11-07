import torch
import torch.nn as nn
from .layers import LayerNorm2d, SimpleGate
from .attention import CrossAttentionMechanism, ChannelwiseSelfAttention
from .aspp import OptimizedASPP

class SBSRBlock(nn.Module):
    """SBSR Block combining attention and ASPP"""
    
    def __init__(self, 
                 channels: int,
                 DW_Expand: int = 2,
                 FFN_Expand: int = 2,
                 drop_out_rate: float = 0.):
        super().__init__()
        
        self.crossa = CrossAttentionMechanism(channels)
        self.channela = ChannelwiseSelfAttention(channels)
        self.aspp = OptimizedASPP(channels)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1, bias=True),
            nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2, bias=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1, bias=True),
            nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2, bias=True)
        )
        
        self.conv3 = nn.Conv2d(channels, channels*2, 1, bias=True)
        self.conv4 = nn.Conv2d(channels, channels, 1, bias=True)
        
        self.sg = SimpleGate()
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.norm1(x)
        
        x = self.channela(x)
        x = self.conv1(x)
        x = self.sg(x)
        
        x = self.crossa(x)
        x = self.conv2(x)
        x = self.sg(x)
        
        x = self.aspp(x)
        
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv3(x)
        x = self.sg(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma