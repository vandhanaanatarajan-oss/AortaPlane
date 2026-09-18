#!/usr/bin/env python3
"""Extract the encoder weights from a trained nnU-Net segmentation checkpoint."""
import torch

CHECKPOINT_PATH = "/data/DERI-ecgai/001_CTA_Segmention/models/Aorta_v0/weights/Dataset050_SEGA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
OUT_PATH = "/data/DERI-ecgai/__users/Vandhanaa/nnlandmark_comparison/encoder_transplant/aorta_v0_encoder.pth"

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
print("Checkpoint top-level keys:", list(ckpt.keys()))

state_dict = ckpt.get("network_weights", ckpt.get("state_dict", ckpt))
print(f"\nTotal params in state_dict: {len(state_dict)}")
print("\nFirst 20 param names:")
for k in list(state_dict.keys())[:20]:
    print(" ", k)

# Filter to encoder-only keys (nnU-Net PlainConvUNet naming convention typically
# prefixes encoder params with 'encoder.' and decoder with 'decoder.')
encoder_keys = [k for k in state_dict if k.startswith("encoder.")]
decoder_keys = [k for k in state_dict if k.startswith("decoder.")]
other_keys = [k for k in state_dict if not k.startswith("encoder.") and not k.startswith("decoder.")]

print(f"\nEncoder params: {len(encoder_keys)}")
print(f"Decoder params: {len(decoder_keys)}")
print(f"Other params: {len(other_keys)}")
if other_keys:
    print("Other key names:", other_keys[:10])

encoder_state = {k: state_dict[k] for k in encoder_keys}
torch.save(encoder_state, OUT_PATH)
print(f"\nSaved encoder-only weights to {OUT_PATH}")
