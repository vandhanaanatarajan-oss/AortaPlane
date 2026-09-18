"""
M1_UNet
-------
M1's direct 6D-regression approach, but with a full U-Net
encoder-decoder (WITH skip connections, identical to M2's decoder)
as the backbone -- instead of M1's own encoder-only (no decoder) design.
GAP is applied to the decoder's final feature map, then fed into a
fresh FC head sized for that feature map's channel count.

Encoder + decoder: identical to M2 (imported, not redefined).
Head: GAP -> Dropout -> FC -> FC -> 6D, same structure as M1's own head,
but resized for the decoder's 16-channel output (M1's original head
was sized for its 256-channel encoder-only output).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from shared_blocks import ResBlock3D, UNetUpBlock3D


class M1_UNet(nn.Module):
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

        self.up1 = UNetUpBlock3D(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = UNetUpBlock3D(in_ch=128, skip_ch=64, out_ch=64)
        self.up3 = UNetUpBlock3D(in_ch=64, skip_ch=32, out_ch=32)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # Head -- resized for the decoder's 16-channel output (M1's
        # original head was sized for its 256-channel encoder output)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)

        d1 = self.up1(s3, s2)
        d2 = self.up2(d1, s1)
        d3 = self.up3(d2, s0)
        d4 = self.up4(d3)

        out = self.gap(d4).view(d4.size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = M1_UNet()
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"output shape: {out.shape} (expect [2, 6])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 6)
    print("PASS")
