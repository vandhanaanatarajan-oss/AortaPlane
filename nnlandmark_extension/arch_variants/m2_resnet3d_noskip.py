"""
M2_ResNet3D_NoSkip
-------------------
M2's landmark-then-plane approach, but with a PLAIN upsampling decoder
(no U-Net-style skip connections) -- isolates what the skip connections
themselves contribute, since M2's own encoder is already ResNet3D-based.

Encoder: identical stem + 3x ResBlock3D as M1/M2 (imported, not
redefined, for a genuinely fair comparison).
Decoder: same number of upsampling stages as M2's U-Net decoder, but
each stage is just Upsample -> Conv -> BN -> ReLU on the PREVIOUS
decoder output alone -- no concatenation with encoder skip features.
Head: identical 3-heatmap + soft-argmax head as M2.
"""
import torch
import torch.nn as nn
from shared_blocks import ResBlock3D, soft_argmax_3d


class PlainUpBlock3D(nn.Module):
    """Same idea as UNetUpBlock3D but WITHOUT the skip connection concat."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class M2_ResNet3D_NoSkip(nn.Module):
    def __init__(self, dropout_p: float = 0.3, use_cbam: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ResBlock3D(32, 64, stride=2, use_se=False, use_cbam=use_cbam)
        self.layer2 = ResBlock3D(64, 128, stride=2, use_se=False, use_cbam=use_cbam)
        self.layer3 = ResBlock3D(128, 256, stride=2, use_se=False, use_cbam=use_cbam)

        self.up1 = PlainUpBlock3D(256, 128)
        self.up2 = PlainUpBlock3D(128, 64)
        self.up3 = PlainUpBlock3D(64, 32)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.head_RC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_NC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_LC = nn.Conv3d(16, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)

        d1 = self.up1(s3)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
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
    model = M2_ResNet3D_NoSkip()
    x = torch.randn(2, 1, 64, 64, 64)
    landmarks, heatmaps = model(x)
    print(f"landmarks shape: {landmarks.shape} (expect [2, 9])")
    print(f"heatmaps shape:  {heatmaps.shape} (expect [2, 3, 64, 64, 64])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert landmarks.shape == (2, 9)
    assert heatmaps.shape == (2, 3, 64, 64, 64)
    print("PASS")
