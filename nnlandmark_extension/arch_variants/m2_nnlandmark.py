"""
M2_nnLandmark
--------------
M2's landmark-then-plane approach, using nnLandmark's own REAL, COMPLETE
network -- PlainConvUNet, encoder AND decoder together, not just a
backbone -- the most faithful of all 6 variants, since it reuses both
halves of nnLandmark's actual architecture, just reconfigured for
M1/M2's fixed 64^3 input and with num_classes=3 (matching M2's 3
heatmap channels: RC/NC/LC) instead of nnLandmark's own segmentation
output.

Same real, confirmed building-block choices as M1_nnLandmark:
InstanceNorm3d, LeakyReLU, conv_bias=True (from nnLandmark's actual
plans.json). deep_supervision=False, since none of the other 5
variants use multi-scale supervision either -- keeps this comparable.
"""
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from shared_blocks import soft_argmax_3d

N_STAGES = 6
FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]  # exact match to nnLandmark's real plans.json
STRIDES = [1, 2, 2, 2, 2, 2]  # stage 0 stays at full input res -- needed for the decoder to reconstruct back to 64^3


class M2_nnLandmark(nn.Module):
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        # num_classes=3 -> nnLandmark's own decoder outputs 3 channels
        # directly (one per landmark: RC/NC/LC) -- no separate head
        # module needed, unlike the other M2-style variants, since
        # PlainConvUNet's decoder already produces exactly this shape.
        self.net = PlainConvUNet(
            input_channels=1,
            n_stages=N_STAGES,
            features_per_stage=FEATURES_PER_STAGE,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=STRIDES,
            n_conv_per_stage=2,
            num_classes=3,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=False,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        heatmaps = self.net(x)  # (B, 3, 64, 64, 64) -- nnLandmark's decoder upsamples all the way back to input resolution natively

        rc = soft_argmax_3d(heatmaps[:, 0:1])
        nc = soft_argmax_3d(heatmaps[:, 1:2])
        lc = soft_argmax_3d(heatmaps[:, 2:3])

        landmarks = torch.cat([rc, nc, lc], dim=1)
        return landmarks, heatmaps


if __name__ == "__main__":
    model = M2_nnLandmark()
    x = torch.randn(2, 1, 64, 64, 64)
    landmarks, heatmaps = model(x)
    print(f"landmarks shape: {landmarks.shape} (expect [2, 9])")
    print(f"heatmaps shape:  {heatmaps.shape} (expect [2, 3, 64, 64, 64])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    assert landmarks.shape == (2, 9)
    assert heatmaps.shape == (2, 3, 64, 64, 64)
    print("PASS")
