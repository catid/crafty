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
# Linear Recurrent Units (LRUs)

# Introduced in:
# "Resurrecting Recurrent Neural Networks for Long Sequences" (Orvieto 2023)
# https://arxiv.org/pdf/2303.06349.pdf

# From https://github.com/adrian-valente/lru_experiments/
# Another good example here: https://github.com/bojone/rnn/blob/main/lru.py
# Some evaluation of LRU vs GRU, transformers and more: https://arxiv.org/pdf/2310.02367.pdf
# This project found LRU performs well for online learning: https://arxiv.org/pdf/2305.15947.pdf
class LRU(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 d_out: int,
                 r_min: float = 0.,
                 r_max: float = 1.,
                 max_phase: float = 6.28
                 ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        self.theta_log = nn.Parameter(torch.empty(d_hidden))
        self.nu_log = nn.Parameter(torch.empty(d_hidden))
        self.gamma_log = nn.Parameter(torch.empty(d_hidden))
        self.B_re = nn.Parameter(torch.empty(d_hidden, d_in))
        self.B_im = nn.Parameter(torch.empty(d_hidden, d_in))
        self.C_re = nn.Parameter(torch.empty(d_out, d_hidden))
        self.C_im = nn.Parameter(torch.empty(d_out, d_hidden))
        self.D = nn.Parameter(torch.empty(d_out, d_in))

        self._init_params()

    def diag_lambda(self) -> torch.Tensor:
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

    def _init_params(self):
        nn.init.uniform_(self.theta_log, a=0, b=self.max_phase)

        u = torch.rand((self.d_hidden,))
        nu_init = torch.log(-0.5 * torch.log(u * (self.r_max**2 - self.r_min**2) + self.r_min**2))
        with torch.no_grad():
            self.nu_log.copy_(nu_init)
            diag_lambda = self.diag_lambda()
            self.gamma_log.copy_(torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2)))

        nn.init.xavier_normal_(self.B_re)
        nn.init.xavier_normal_(self.B_im)
        nn.init.xavier_normal_(self.C_re)
        nn.init.xavier_normal_(self.C_im)
        nn.init.xavier_normal_(self.D)  # Set something like diagonal matrix eventually

    def forward(self, u: torch.Tensor, init_states: torch.Tensor = None) -> torch.Tensor:
        diag_lambda = self.diag_lambda()
        B_norm = torch.diag(self.gamma_log).to(torch.cfloat) @ (self.B_re + 1j * self.B_im)
        C = self.C_re + 1j * self.C_im

        # Initial states can be a vector of shape (d_hidden,) or a matrix of shape (batch_size, d_hidden)
        if init_states is not None and init_states.ndim == 1:
            init_states = init_states.unsqueeze(0)

        h = init_states.to(torch.cfloat) if init_states is not None \
                else torch.zeros((u.shape[0], self.d_hidden), dtype=torch.cfloat, device=self.theta_log.device)
        outputs = []
        # FIXME: Incorporate https://github.com/proger/accelerated-scan
        for t in range(u.shape[1]):
            h = h * diag_lambda + u[:, t].to(torch.cfloat) @ B_norm.T
            y = torch.real(h @ C.T) + u[:, t] @ self.D.T
            outputs.append(y)

        return torch.stack(outputs, dim=1)

    def __repr__(self):
        return f"LRU(d_in={self.d_in}, d_hidden={self.d_hidden}, d_out={self.d_out})"


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
