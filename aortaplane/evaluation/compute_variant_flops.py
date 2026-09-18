"""
compute_variant_flops.py -- FINAL VERSION

Computes FLOPs for all 8 rows of Table II (native AortaPlane, native
Baseline, and the 6 encoder/architecture variants), using thop.

Handles two different checkpoint formats found on disk:
  - native checkpoints (M1_fold0, M2_fold0): wrapped dict with a
    'model_state_dict' key
  - all 6 variant checkpoints: raw state_dict (torch.load returns the
    state_dict itself)

Run from /data/DERI-ecgai/__users/Vandhanaa (project root) inside the
project's own venv:
    source /data/DERI-ecgai/__users/Vandhanaa/venv/bin/activate
    pip install thop --user   # if not already installed
    python compute_variant_flops.py

Input resolution: confirmed via checkpoint['args']['target_size'] == 64
for both native models (M1_fold0, M2_fold0) -- NOT 128. M1_128_fold0
is a separate, apparently broken/stale checkpoint (worse angle error
than the 64^3 run, contradicting Table III's own ablation finding) --
excluded here, needs its own separate investigation before trusting
anything from it.
"""

import sys
import os
import torch
import torch.nn as nn
from thop import profile, clever_format

# arch_variants/ holds m1_densenet3d.py, m1_nnlandmark.py, m2_nnlandmark.py
# (train_m1_variants.py / train_m2_variants.py must sys.path-insert this
# directory somewhere not caught by the `^import|^from` grep -- confirmed
# these files only exist there, not at project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nnlandmark_comparison", "arch_variants"))

from src.model import AnnulusPlaneNet, AnnulusLandmarkNet     # native AortaPlane / Baseline
from m1_unet import M1_UNet
from m2_resnet3d_noskip import M2_ResNet3D_NoSkip
from m2_densenet3d import M2_DenseNet3D                        # root version, confirmed identical
                                                                  # encoder to arch_variants' copy
from m1_densenet3d import M1_DenseNet3D                         # from arch_variants/, via sys.path above
from m1_nnlandmark import M1_nnLandmark                         # from arch_variants/
from m2_nnlandmark import M2_nnLandmark                         # from arch_variants/

CHECKPOINT_ROOT = "checkpoints"
INPUT_SHAPE = (1, 1, 64, 64, 64)  # confirmed target_size=64 for native models;
                                   # all 6 variants trained on the same fixed
                                   # M1/M2 pipeline input, same resolution

# (display name, checkpoint dir, model class, constructor kwargs, checkpoint format)
VARIANTS = [
    ("AortaPlane (native)",        "M1_fold0",             AnnulusPlaneNet,
        dict(dropout_p=0.3, use_se=False, use_cbam=True), "wrapped"),
    ("Baseline (native)",          "M2_fold0",             AnnulusLandmarkNet,
        dict(dropout_p=0.3, use_cbam=True), "wrapped"),
    ("AortaPlane + U-Net",         "M1_unet_fold0",        M1_UNet,
        dict(dropout_p=0.3, use_cbam=True), "raw"),
    ("AortaPlane + DenseNet3D",    "M1_densenet3d_fold0",  M1_DenseNet3D,
        dict(dropout_p=0.3, growth_rate=16, block_layers=(4, 4, 4)), "raw"),
    ("AortaPlane + nnLandmark",    "M1_nnlandmark_fold0",  M1_nnLandmark,
        dict(dropout_p=0.3), "raw"),
    ("Baseline + ResNet3D(noskip)", "M2_resnet3d_noskip_fold0", M2_ResNet3D_NoSkip,
        dict(dropout_p=0.3, use_cbam=True), "raw"),
    ("Baseline + DenseNet3D",      "M2_densenet3d_fold0",  M2_DenseNet3D,
        dict(dropout_p=0.3, growth_rate=16, block_layers=(4, 4, 4)), "raw"),
    ("Baseline + nnLandmark",      "M2_nnlandmark_fold0",  M2_nnLandmark,
        dict(dropout_p=0.3), "raw"),
]


def load_state_dict(ckpt_path, fmt):
    sd = torch.load(ckpt_path, map_location="cpu")
    if fmt == "wrapped":
        return sd["model_state_dict"]
    return sd


def profile_variant(name, ckpt_dir, model_cls, kwargs, fmt):
    ckpt_path = f"{CHECKPOINT_ROOT}/{ckpt_dir}/{ckpt_dir}_best.pth"
    model = model_cls(**kwargs)
    state_dict = load_state_dict(ckpt_path, fmt)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(*INPUT_SHAPE)

    caveat = ""
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # FLOPs = 2x MACs by convention
        flops_str, params_str = clever_format([flops, params], "%.3f")
    except Exception as e:
        flops_str, params_str = "FAILED", "FAILED"
        caveat = f"  [thop error: {e}]"
        flops = None

    manual_params = sum(p.numel() for p in model.parameters())

    # trilinear-upsample decoders are known (from the 12 Aug run) to be
    # underestimated by thop even when profiling succeeds without error --
    # flag these three explicitly rather than silently trusting the number
    if name in ("Baseline (native)", "AortaPlane + U-Net", "Baseline + DenseNet3D"):
        caveat += "  [*underestimate: uses nn.Upsample(trilinear), thop under-profiles this op]"

    print(f"{name:32s}  FLOPs: {flops_str:>10s}  Params(thop): {params_str:>10s}"
          f"  Params(manual): {manual_params:,}{caveat}")

    return {"name": name, "flops": flops_str, "flops_raw": flops,
            "params_manual": manual_params, "caveat": caveat}


if __name__ == "__main__":
    results = []
    for name, ckpt_dir, model_cls, kwargs, fmt in VARIANTS:
        try:
            results.append(profile_variant(name, ckpt_dir, model_cls, kwargs, fmt))
        except Exception as e:
            print(f"{name:32s}  ERROR loading/profiling: {e}")

    print("\n--- Table II FLOPs column (paste in) ---")
    for r in results:
        star = "*" if r["caveat"] else ""
        print(f"{r['name']}: {r['flops']}{star}")

    print(
        "\nNOTE ON ACCURACY: three rows use nn.Upsample(mode='trilinear') "
        "decoders, which thop under-profiles (same known issue as native "
        "M2's earlier 9.93 GFLOPs* figure) -- keep the '*' caveat on those "
        "rows in the final table rather than presenting them as exact."
    )
    print(
        "\nNOTE: nnLandmark's OWN standalone native backbone (88.21M "
        "params, separate from these transplant rows) is NOT part of this "
        "grid. Its existing FLOPs script "
        "(nnlandmark_comparison/compute_nnlandmark_native_flops.py) "
        "produced a freshly-built model with only 30.8M params vs the "
        "checkpoint's real 88.2M -- its architecture config doesn't match "
        "the trained model, so its 7930.83 GFLOPs figure is NOT reliable "
        "and needs separate debugging if you want that number too."
    )