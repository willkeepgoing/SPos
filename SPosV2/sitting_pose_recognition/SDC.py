import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math


class SpatialDisplacementContact(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, is_pe=False):
        super().__init__()

        self.patch_shifting = DisplacementContact(merging_size)
        patch_dim = (in_dim*5) * (merging_size**2)

        self.is_pe = is_pe
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = merging_size, p2 = merging_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
        out = self.patch_shifting(out)
        out = self.merging(out)
        
        return out


class DisplacementContact(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.shift = int(patch_size * (1/2))
        
    def forward(self, x):
     
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))

        x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

        out = x_cat
        
        return out