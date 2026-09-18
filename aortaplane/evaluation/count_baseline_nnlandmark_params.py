import torch

p = "checkpoints/M2_nnlandmark_fold0/M2_nnlandmark_fold0_best.pth"
sd = torch.load(p, map_location="cpu", weights_only=False)

if not isinstance(sd, dict):
    print(f"UNEXPECTED FORMAT: {type(sd)}")
else:
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"Baseline + nnLandmark    {total:>12,} params ({total/1e6:.2f}M)   [{p}]")
