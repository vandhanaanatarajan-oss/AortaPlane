import torch

paths = {
    "AortaPlane + U-Net":            "checkpoints/M1_unet_fold0/M1_unet_fold0_best.pth",
    "AortaPlane + DenseNet3D":       "checkpoints/M1_densenet3d_fold0/M1_densenet3d_fold0_best.pth",
    "Baseline + ResNet3D (no skip)": "checkpoints/M2_resnet3d_noskip_fold0/M2_resnet3d_noskip_fold0_best.pth",
    "Baseline + DenseNet3D":         "checkpoints/M2_densenet3d_fold0/M2_densenet3d_fold0_best.pth",
}

for name, p in paths.items():
    try:
        sd = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"{name:<32} FAILED TO LOAD {p}: {e}")
        continue

    if not isinstance(sd, dict):
        print(f"{name:<32} UNEXPECTED FORMAT: {type(sd)}")
        continue

    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"{name:<32} {total:>12,} params ({total/1e6:.2f}M)   [{p}]")
