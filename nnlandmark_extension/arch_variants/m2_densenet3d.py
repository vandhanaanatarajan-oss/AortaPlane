"""
M2_DenseNet3D
--------------
M2's landmark-then-plane approach, with a DenseNet3D encoder instead
of M2's own ResNet3D encoder, feeding a U-Net-style decoder (WITH skip
connections, matching M2's own decoder structure) -> 3 heatmap heads
-> soft-argmax -> 9D landmarks.
"""
import torch
import torch.nn as nn
from densenet3d_encoder import DenseNet3DEncoder
from shared_blocks import soft_argmax_3d


class UNetUpBlock3D_Generic(nn.Module):
    """Same as shared_blocks.UNetUpBlock3D, but channel counts are
    supplied at construction time to match DenseNet3D's (different from
    ResNet3D's) channel progression at each stage."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class M2_DenseNet3D(nn.Module):
    def __init__(self, dropout_p: float = 0.3, growth_rate: int = 16, block_layers=(4, 4, 4)):
        super().__init__()
        self.encoder = DenseNet3DEncoder(growth_rate=growth_rate, block_layers=block_layers)
        s0_ch, s1_ch, s2_ch, s3_ch = 32, *self.encoder.stage_out_channels  # e.g. 32, 48, 56, 60

        self.up1 = UNetUpBlock3D_Generic(in_ch=s3_ch, skip_ch=s2_ch, out_ch=s2_ch)
        self.up2 = UNetUpBlock3D_Generic(in_ch=s2_ch, skip_ch=s1_ch, out_ch=s1_ch)
        self.up3 = UNetUpBlock3D_Generic(in_ch=s1_ch, skip_ch=s0_ch, out_ch=s0_ch)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(s0_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.head_RC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_NC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_LC = nn.Conv3d(16, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        s0, s1, s2, s3 = self.encoder(x)

        d1 = self.up1(s3, s2)
        d2 = self.up2(d1, s1)
        d3 = self.up3(d2, s0)
        d4 = self.up4(d3)

        hm_rc = self.head_RC(d4)
        hm_nc = self.head_NC(d4)
        hm_lc = self.head_LC(d4)

        rc = soft_argmax_3d(hm_rc)
        nc = soft_argmax_3d(hm_nc)
        lc = soft_argmax_3d(hm_lc)

        landmarks = torch.cat([rc, nc, lc], dim=1)
        heatmaps = torch.cat([hm_rc, hm_nc, hm_lc], dim=1)
        return landmarks, heatmaps


if __name__ == "__main__":
    model = M2_DenseNet3D()
    x = torch.randn(2, 1, 64, 64, 64)
    landmarks, heatmaps = model(x)
    print(f"landmarks shape: {landmarks.shape} (expect [2, 9])")
    print(f"heatmaps shape:  {heatmaps.shape} (expect [2, 3, 64, 64, 64])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert landmarks.shape == (2, 9)
    assert heatmaps.shape == (2, 3, 64, 64, 64)
    print("PASS")
