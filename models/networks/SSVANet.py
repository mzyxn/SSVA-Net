import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from ..blocks.sbsr_block import SBSRBlock
from ..base_model import BaseModel

class SBSRNet(BaseModel):
    """Shadow Removal Network using SBSR blocks"""
    
    def __init__(self, config):
        super().__init__(config)
        
        self.intro = nn.Conv2d(config.img_channel, config.width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(config.width, config.img_channel, 3, padding=1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = config.width
        # Encoder blocks
        for num in config.enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[SBSRBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        
        # Middle blocks
        self.middle_blks = nn.Sequential(
            *[SBSRBlock(chan) for _ in range(config.middle_blk_num)]
        )
        
        # Decoder blocks
        for num in config.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[SBSRBlock(chan) for _ in range(num)])
            )
        
        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure the input size is compatible with the model architecture."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        
        # Initial convolution
        x = self.intro(x)
        
        # Encoder path
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        # Middle blocks
        x = self.middle_blks(x)
        
        # Decoder path with skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        # Final convolution and residual connection
        x = self.ending(x)
        x = x + self.check_image_size(x)
        
        return x[:, :, :H, :W]

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a training step."""
        lq_images, gt_images = batch['lq'], batch['gt']
        pred_images = self.forward(lq_images)
        
        return {
            'pred_images': pred_images,
            'gt_images': gt_images,
            'lq_images': lq_images
        }

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a validation step."""
        with torch.no_grad():
            return self.train_step(batch)

    @torch.no_grad()
    def inference(self, lq_image: torch.Tensor) -> torch.Tensor:
        """Run inference on a single image."""
        self.eval()
        pred_image = self.forward(lq_image)
        return pred_image