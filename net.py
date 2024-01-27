# Ideas:
# Data augmentation by masking DINOv2 features

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce


# NestNets seem to be more sample efficient? https://arxiv.org/pdf/2205.09459.pdf
class HyperFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        #self.ffn = PreNormResidual(dim, FeedForward(dim, hidden_dim, dropout))

        # Use one scalar parameter for each unique operation (edge)
        self.params = nn.ParameterList([nn.Parameter(torch.empty(1)) for _ in range(8)])
        # Bias terms for each operation
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(8)])
        self._init_parameters()

    def _init_parameters(self):
        for param in self.params:
            nn.init.normal_(param)  # Initialize with a normal distribution
        for bias in self.biases:
            nn.init.zeros_(bias)  # Initialize biases to zero

    def forward(self, x):
        x = self.ffn(x)
        y = F.gelu(x * self.params[0] + self.biases[0])
        z = F.gelu(x * self.params[1] + self.biases[1])
        y, z = self.ffn(y), self.ffn(z)
        u = F.gelu((y * self.params[2] + self.biases[2]) + (z * self.params[3] + self.biases[3]))
        v = F.gelu((y * self.params[4] + self.biases[4]) + (z * self.params[5] + self.biases[5]))
        u, v = self.ffn(u), self.ffn(v)
        w = F.gelu((u * self.params[6] + self.biases[6]) + (v * self.params[7] + self.biases[7]))
        return self.ffn(w)


################################################################################
# Transformer-like MLP

from m2.blockdiag_linear import BlockdiagLinear

# "TransNormerLLM" (Qin et al 2024) https://arxiv.org/pdf/2307.14995.pdf
class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class NeckFormer(nn.Module):
    def __init__(self, d_in=1024, height=16, width=16, d_out=128, d_hidden=128, segments=8, depth=1, expansion_factor=4, dropout=0.1):
        super(NeckFormer, self).__init__()
        assert (d_hidden % segments) == 0, 'Dimension must be divisible by the number of segments'

        layers = [
            # Reshape and linear layer to transform the channel dimension
            Rearrange('b c h w -> b h w c'),
            nn.Linear(d_in, d_hidden)
        ]

        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    # Inter-token mixing (Attention)
                    PreNormResidual(d_hidden, nn.Sequential(
                        # Mix across columns
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b w c (h s)', s=segments),
                            BlockdiagLinear(height * segments, height * segments, bias=False, nblocks=4),
                            Rearrange('b w c (h s) -> b h w (c s)', s=segments),
                        ),
                        BlockdiagLinear(d_hidden, d_hidden, bias=False, nblocks=4),

                        nn.Dropout(dropout),

                        # Mix across rows
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b h c (w s)', s=segments),
                            BlockdiagLinear(width * segments, width * segments, bias=False, nblocks=4),
                            Rearrange('b h c (w s) -> b h w (c s)', s=segments),
                        ),
                        BlockdiagLinear(d_hidden, d_hidden, bias=False, nblocks=4),
                    )),

                    # Intra-token mixing (FFN)
                    PreNormResidual(d_hidden, nn.Sequential(
                        BlockdiagLinear(d_hidden, d_hidden * expansion_factor, bias=False, nblocks=4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        BlockdiagLinear(d_hidden * expansion_factor, d_hidden, bias=False, nblocks=4),
                    ))
                )
            )

        # Output pooling
        layers.extend([
            nn.LayerNorm(d_hidden),
            Reduce('b h w c -> b c', 'max'),
            nn.Linear(d_hidden, d_out)
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


################################################################################
# ImageEncoder

# Encode from a large latent space into a smaller one
class ImageEncoder(nn.Module):
    def __init__(self, d_in, d_mlp, d_out):
        super(ImageEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_mlp),
            nn.GELU(),

            nn.Linear(d_mlp, d_mlp),
            nn.GELU(),

            nn.Linear(d_mlp, d_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.encoder(x)


################################################################################
# DINOv2

class DINOv2Backbone(nn.Module):
    def __init__(self, encoder='vitl', height=16, width=16, d_hidden=128, d_out=128):
        super(DINOv2Backbone, self).__init__()

        assert encoder in ['vits', 'vitb', 'vitl', 'vitg']

        self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14_reg'.format(encoder))
        self.pretrained.eval()

    def forward(self, x):
        # Get the last layer
        features = self.pretrained.get_intermediate_layers(x, 1, reshape=True, return_class_token=False)

        batch = torch.stack(features, dim=0).squeeze(0)
        return batch


################################################################################
# Image Decoder

# Decode a 64x64 RGB image with outputs from -1..1, from an input feature space
class ImageDecoder(nn.Module):
    def __init__(self, d_in=128, d_chan=8, d_conv=4):
        super(ImageDecoder, self).__init__()

        assert d_in == d_chan * d_conv * d_conv, "Channel split must be even"

        self.decoder = nn.Sequential(
            nn.Linear(d_in, d_chan*d_conv*d_conv),
            nn.GELU(),

            nn.Unflatten(-1, (d_chan, d_conv, d_conv)),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(d_chan, d_chan, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(d_chan, d_chan, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(d_chan, d_chan, kernel_size=4, stride=2, padding=1),
            nn.GELU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(d_chan, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(x)
