"""
CoordConv3D — drop-in replacement for nn.Conv3d that gives the convolution
direct access to absolute spatial position, per Liu et al. 2018
(arXiv:1807.03247), extended to 3D (D, H, W) volumes.

Usage: swap a single nn.Conv3d(in_ch, out_ch, ...) for
CoordConv3D(in_ch, out_ch, ...) — same call signature, everything else
about the surrounding model is unchanged.
"""
import torch
import torch.nn as nn


class AddCoords3D(nn.Module):
    """Concatenates normalised [-1, 1] D/H/W coordinate channels (and
    optionally a radial-distance channel) onto a 5D (B, C, D, H, W) tensor.
    """
    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        device, dtype = x.device, x.dtype

        d_coords = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        h_coords = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        w_coords = torch.linspace(-1, 1, W, device=device, dtype=dtype)

        dd, hh, ww = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")

        dd = dd.expand(B, 1, D, H, W)
        hh = hh.expand(B, 1, D, H, W)
        ww = ww.expand(B, 1, D, H, W)

        out = torch.cat([x, dd, hh, ww], dim=1)

        if self.with_r:
            rr = torch.sqrt(dd ** 2 + hh ** 2 + ww ** 2)
            out = torch.cat([out, rr], dim=1)

        return out


class CoordConv3D(nn.Module):
    """Drop-in replacement for nn.Conv3d. Adds 3 (or 4, with_r=True) extra
    input channels carrying absolute spatial position before a standard
    Conv3d. Same call signature as nn.Conv3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 with_r: bool = False):
        super().__init__()
        self.with_r = with_r
        self.addcoords = AddCoords3D(with_r=with_r)
        extra_ch = 4 if with_r else 3
        self.conv = nn.Conv3d(
            in_channels + extra_ch, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.addcoords(x)
        return self.conv(x)