#!/usr/bin/env python3
import torch

CKPT_PATH = ("nnLM_data/results/Dataset741_AnnulusCT/"
             "nnLandmark_PartialUnfreeze__nnUNetPlansV1Match__3d_fullres/"
             "fold_0/checkpoint_final.pth")

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

print("Checkpoint top-level keys:", list(ckpt.keys()))

state_dict = None
for key in ["network_weights", "state_dict", "model_state_dict"]:
    if key in ckpt:
        state_dict = ckpt[key]
        print(f"Found state dict under key: '{key}'")
        break

if state_dict is None:
    print("No known state-dict key found -- inspect ckpt.keys() above manually.")
else:
    n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
    print(f"\nnnLandmark V1 PartialUnfreeze (fold 0) parameter count: {n_params:,}")
