"""
M1_nnLandmark
--------------
M1's direct 6D-regression head, with nnLandmark's own real internal
network (PlainConvEncoder, from the dynamic_network_architectures
package nnU-Net/nnLandmark is actually built on) as the encoder,
reconfigured for M1/M2's fixed 64^3 input -- nnLandmark's own native
config is built for full-resolution variable-size CT scans, which
doesn't fit this pipeline, so this uses a downsampling schedule
matching M1/M2's other encoders (5 stride-2 stages, 64 -> 2), while
keeping nnLandmark's REAL, characteristic building-block choices:
InstanceNorm3d, LeakyReLU, conv_bias=True -- confirmed from
nnLandmark's actual plans.json (nnUNetPlansV1Match, seen directly on
Apocrita earlier this project) -- not guessed defaults.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

N_STAGES = 6
FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]  # exact match to nnLandmark's real plans.json channel progression
STRIDES = [1, 2, 2, 2, 2, 2]  # stage 0 stays at full input res (nnU-Net's real convention, confirmed in plans.json), enabling the decoder to reconstruct back to 64^3 via skip connections


class M1_nnLandmark(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.encoder = PlainConvEncoder(
            input_channels=1,
            n_stages=N_STAGES,
            features_per_stage=FEATURES_PER_STAGE,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=STRIDES,
            n_conv_per_stage=2,
            conv_bias=True,  # confirmed from nnLandmark's real plans.json
            norm_op=nn.InstanceNorm3d,  # confirmed real choice, not M1/M2's own BatchNorm3d
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,  # confirmed real choice, not M1/M2's own ReLU
            nonlin_kwargs={'inplace': True},
            return_skips=False,  # M1-style has no decoder, so no skip connections needed
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(FEATURES_PER_STAGE[-1], 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        feat = self.encoder(x)  # final feature map, (B, 320, 2, 2, 2)
        out = self.gap(feat).view(feat.size(0), -1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    model = M1_nnLandmark()
    x = torch.randn(2, 1, 64, 64, 64)
    out = model(x)
    print(f"output shape: {out.shape} (expect [2, 6])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert out.shape == (2, 6)
    print("PASS")
