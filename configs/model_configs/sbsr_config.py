from ..base_config import BaseConfig
from dataclasses import dataclass

@dataclass
class SBSRConfig(BaseConfig):
    """Configuration for SBSR model"""
    
    # Model-specific parameters
    width: int = 32
    middle_blk_num: int = 1
    enc_blk_nums: tuple = (1, 1, 1, 1)
    dec_blk_nums: tuple = (1, 1, 1, 1)
    
    # Architecture specific settings
    use_cross_attention: bool = True
    use_channel_attention: bool = True
    use_aspp: bool = True
    
    # Attention mechanisms settings
    attention_dim: int = 32
    attention_heads: int = 8
    attention_dropout: float = 0.0
    
    # ASPP settings
    aspp_rates: tuple = (6, 12, 18)
    aspp_dropout: float = 0.0
    
    # Layer norm settings
    layer_norm_eps: float = 1e-6
    
    # SimpleGate settings
    gate_channels: int = None  # If None, will use width * 2
    
    def __post_init__(self):
        super().__post_init__()
        if self.gate_channels is None:
            self.gate_channels = self.width * 2