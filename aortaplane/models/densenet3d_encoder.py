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
