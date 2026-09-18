mkdir -p /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants
cd /data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants

cat > shared_blocks.py << 'ARCHEOF'
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


class CBAMBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        avg = self.channel_fc(self.avg_pool(x).view(b, c))
        mx = self.channel_fc(self.max_pool(x).view(b, c))
        ch_w = self.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        x = x * ch_w
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_w = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sp_w


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, use_cbam=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        if use_cbam:
            self.se = CBAMBlock3D(out_ch)
        elif use_se:
            self.se = SEBlock3D(out_ch)
        else:
            self.se = nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return F.relu(out)


class UNetUpBlock3D(nn.Module):
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


def soft_argmax_3d(heatmap):
    B, _, D, H, W = heatmap.shape
    h = heatmap.view(B, -1)
    h = torch.softmax(h, dim=-1).view(B, 1, D, H, W)
    device = heatmap.device
    d_coords = torch.linspace(0, 1, D, device=device)
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)
    grid_d = d_coords.view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
    grid_h = h_coords.view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
    grid_w = w_coords.view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
    coord_d = (h * grid_d).sum(dim=[2, 3, 4])
    coord_h = (h * grid_h).sum(dim=[2, 3, 4])
    coord_w = (h * grid_w).sum(dim=[2, 3, 4])
    return torch.cat([coord_d, coord_h, coord_w], dim=1)
ARCHEOF

cat > m1_unet.py << 'ARCHEOF'
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
ARCHEOF

cat > m1_densenet3d.py << 'ARCHEOF'
"""
M1_DenseNet3D
--------------
M1's direct 6D-regression head, with a DenseNet3D encoder instead of
M1's own ResNet3D encoder. Encoder-only (no decoder), matching M1's own
"no decoder at all" design -- GAP straight off the final DenseNet3D
feature map.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from densenet3d_encoder import DenseNet3DEncoder


class M1_DenseNet3D(nn.Module):
    def __init__(self, dropout_p: float = 0.3, growth_rate: int = 16, block_layers=(4, 4, 4)):
        super().__init__()
        self.encoder = DenseNet3DEncoder(growth_rate=growth_rate, block_layers=block_layers)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(self.encoder.out_channels, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.gap(feats[-1]).view(feats[-1].size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = M1_DenseNet3D()
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"output shape: {out.shape} (expect [2, 6])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 6)
    print("PASS")
ARCHEOF

cat > densenet3d_encoder.py << 'ARCHEOF'
"""
DenseNet3D encoder
-------------------
A genuinely new component -- DenseNet3D doesn't exist anywhere in the
current codebase. Implements the standard DenseNet idea (each layer's
output is concatenated with ALL previous layers' outputs within a dense
block, encouraging feature reuse) in 3D, with a similar overall
downsampling schedule to M1/M2's ResNet3D encoder (4 downsampling
stages: stem + 3 more, matching their stem+layer1+layer2+layer3), so
parameter count and receptive field are broadly comparable -- a fairer
comparison than picking an arbitrarily deeper/shallower network.
"""
import torch
import torch.nn as nn


class DenseLayer3D(nn.Module):
    """One layer within a dense block: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3),
    output has `growth_rate` channels, concatenated onto the input."""
    def __init__(self, in_ch, growth_rate, bn_size=4):
        super().__init__()
        inter_ch = bn_size * growth_rate
        self.bn1 = nn.BatchNorm3d(in_ch)
        self.conv1 = nn.Conv3d(in_ch, inter_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_ch)
        self.conv2 = nn.Conv3d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)


class DenseBlock3D(nn.Module):
    def __init__(self, in_ch, num_layers, growth_rate, bn_size=4):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer3D(ch, growth_rate, bn_size))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class TransitionLayer3D(nn.Module):
    """BN-ReLU-Conv(1x1, halves channels)-AvgPool(halves spatial size)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_ch)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(torch.relu(self.bn(x)))
        return self.pool(x)


class DenseNet3DEncoder(nn.Module):
    """
    Stem (stride 4, matching ResBlock3D encoder's stem+maxpool) -> 3x
    (DenseBlock -> TransitionLayer), matching the 4-stage downsampling
    of M1/M2's ResNet3D encoder: 64^3 input -> 2^3 feature map, mirroring
    stem + layer1 + layer2 + layer3's spatial reduction.

    Returns a list of feature maps at each stage [s0, s1, s2, s3] for
    U-Net-style decoders that need skip connections, and the final
    feature map alone (s3) is what a GAP+FC head would use.
    """
    def __init__(self, growth_rate=16, block_layers=(4, 4, 4), bn_size=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )  # -> (32, 16,16,16), matches ResBlock3D encoder's stem output

        ch = 32
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.stage_out_channels = []
        for num_layers in block_layers:
            block = DenseBlock3D(ch, num_layers, growth_rate, bn_size)
            ch = block.out_channels
            trans_out = ch // 2  # halve channels at each transition, standard DenseNet practice
            trans = TransitionLayer3D(ch, trans_out)
            self.blocks.append(block)
            self.transitions.append(trans)
            ch = trans_out
            self.stage_out_channels.append(ch)

        self.out_channels = ch  # final stage's channel count, for building a matching head

    def forward(self, x):
        s0 = self.stem(x)  # (32, 16,16,16)
        feats = [s0]
        x = s0
        for block, trans in zip(self.blocks, self.transitions):
            x = block(x)
            x = trans(x)
            feats.append(x)
        return feats  # [s0, s1, s2, s3] -- s3 is the final, most-downsampled feature map


if __name__ == "__main__":
    enc = DenseNet3DEncoder()
    x = torch.randn(2, 1, 64, 64, 64)
    feats = enc(x)
    for i, f in enumerate(feats):
        print(f"stage {i}: {f.shape}")
    print(f"final channel count: {enc.out_channels}")
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"Encoder parameters: {n_params:,}")
    assert feats[-1].shape[2:] == (2, 2, 2), f"Expected final spatial size (2,2,2), got {feats[-1].shape[2:]}"
    print("PASS")
ARCHEOF

cat > m2_resnet3d_noskip.py << 'ARCHEOF'
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
ARCHEOF

cat > m2_densenet3d.py << 'ARCHEOF'
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
ARCHEOF


echo ""
echo "=== Running local sanity checks (no GPU needed) ==="
for f in m1_unet.py m1_densenet3d.py m2_resnet3d_noskip.py m2_densenet3d.py; do
  echo "--- $f ---"
  python3 $f
done