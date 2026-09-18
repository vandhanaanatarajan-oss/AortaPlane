#!/usr/bin/env python3
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

try:
    from thop import profile
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False
    print("thop not installed -- run: pip install thop --break-system-packages")
    raise SystemExit(1)

N_STAGES = 6
FEATURES_PER_STAGE = [32, 64, 128, 256, 320, 320]
STRIDES = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
KERNEL_SIZES = [[3, 3, 3]] * 6
PATCH_SIZE = (160, 128, 112)

model = PlainConvUNet(
    input_channels=1,
    n_stages=N_STAGES,
    features_per_stage=FEATURES_PER_STAGE,
    conv_op=nn.Conv3d,
    kernel_sizes=KERNEL_SIZES,
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
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters (freshly built, should roughly match checkpoint's 88,211,599): {n_params:,}")

x = torch.randn(1, 1, *PATCH_SIZE)
print(f"Input shape: {tuple(x.shape)} (real patch size from plans.json)")

macs, _ = profile(model, inputs=(x,), verbose=False)
flops = macs * 2
print(f"\nFLOPs: {flops:,.0f}")
print(f"GFLOPs: {flops / 1e9:.2f}")
