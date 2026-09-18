#!/usr/bin/env python3
import sys
import torch

sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa')
sys.path.insert(0, '/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/arch_variants')

from src.model import AnnulusPlaneNet, AnnulusLandmarkNet
from m1_unet import M1_UNet
from m1_densenet3d import M1_DenseNet3D
from m1_nnlandmark import M1_nnLandmark
from m2_densenet3d import M2_DenseNet3D
from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_nnlandmark import M2_nnLandmark

try:
    from thop import profile
    HAVE_THOP = True
except ImportError:
    HAVE_THOP = False
    print("WARNING: thop not installed -- FLOPs will be skipped, only params computed.")
    print("Install with: pip install thop --break-system-packages\n")

MODELS = {
    "M1_native": lambda: AnnulusPlaneNet(use_cbam=True),
    "M2_native": lambda: AnnulusLandmarkNet(),
    "M1_unet": lambda: M1_UNet(),
    "M1_densenet3d": lambda: M1_DenseNet3D(),
    "M1_nnlandmark": lambda: M1_nnLandmark(),
    "M2_densenet3d": lambda: M2_DenseNet3D(),
    "M2_resnet3d_noskip": lambda: M2_ResNet3D_NoSkip(),
    "M2_nnlandmark": lambda: M2_nnLandmark(),
}


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cpu")
    x = torch.randn(1, 1, 64, 64, 64).to(device)

    print(f"{'Model':<22} {'Parameters':>15} {'FLOPs':>18} {'GFLOPs':>10}")
    print("-" * 68)

    results = []
    for name, model_fn in MODELS.items():
        model = model_fn().to(device).eval()
        n_params = count_params(model)

        flops_str = "N/A"
        gflops_str = "N/A"
        if HAVE_THOP:
            try:
                macs, _ = profile(model, inputs=(x,), verbose=False)
                flops = macs * 2
                flops_str = f"{flops:,.0f}"
                gflops_str = f"{flops / 1e9:.2f}"
            except Exception as e:
                flops_str = f"ERROR: {e}"

        print(f"{name:<22} {n_params:>15,} {flops_str:>18} {gflops_str:>10}")
        results.append({"model": name, "params": n_params, "flops": flops_str, "gflops": gflops_str})

    print("\nDone.")


if __name__ == "__main__":
    main()
