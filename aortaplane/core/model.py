"""
model.py  — IMPROVED VERSION v2
---------------------------------
Key improvements over v1:
  1. Channel Attention (SE block)  — model learns WHICH feature channels matter most
  2. Architecture summary printer  — shows layer shapes step by step
  3. Parameter breakdown           — shows params per layer group
  4. Dropout rate as argument      — easy to experiment with

The SE (Squeeze-and-Excite) block is a simple attention mechanism
that is cited in papers and adds only ~1% extra parameters.
It is one of your dissertation's 'attention' contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. SQUEEZE-AND-EXCITATION BLOCK  (NEW in v2)
# ─────────────────────────────────────────────

class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.

    How it works:
    1. Squeeze:  global average pool  →  (B, C, 1, 1, 1)
    2. Excite:   two FC layers        →  channel weights in [0, 1]
    3. Scale:    multiply each channel by its learned weight

    This lets the model learn "which feature channels matter most"
    for predicting the annulus plane, acting as a lightweight attention mechanism.

    Reference: Hu et al. "Squeeze-and-Excitation Networks" CVPR 2018.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.gap(x).view(b, c)     # squeeze  → (B, C)
        w = self.fc(w).view(b, c, 1, 1, 1)  # excite  → (B, C, 1, 1, 1)
        return x * w                    # scale



# ─────────────────────────────────────────────
# 1b. CBAM BLOCK (Channel + Spatial Attention)
# ─────────────────────────────────────────────
class CBAMBlock3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D feature maps.
    Applies channel attention then spatial attention sequentially.
    Reference: Woo et al. "CBAM" ECCV 2018.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # Spatial attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7,
                                      padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        # Channel attention
        avg = self.channel_fc(self.avg_pool(x).view(b, c))
        mx  = self.channel_fc(self.max_pool(x).view(b, c))
        ch_w = self.sigmoid(avg + mx).view(b, c, 1, 1, 1)
        x = x * ch_w
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sp_w  = self.sigmoid(self.spatial_conv(
                    torch.cat([avg_s, max_s], dim=1)))
        return x * sp_w

# ─────────────────────────────────────────────
# 2. RESIDUAL BLOCK  (improved: SE attention added)
# ─────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """
    3D residual block with optional SE channel attention.

    Architecture:
        Input → Conv3D → BN → ReLU → Conv3D → BN → SE (optional) → + skip → ReLU
    """

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, use_se: bool = True, use_cbam: bool = False):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        if use_cbam:
            self.se = CBAMBlock3D(out_ch)
        elif use_se:
            self.se = SEBlock3D(out_ch)
        else:
            self.se = nn.Identity()

        # Projection shortcut when shape changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)              # channel attention
        out = out + self.shortcut(x)   # residual connection
        return F.relu(out)


# ─────────────────────────────────────────────
# 3. MAIN MODEL
# ─────────────────────────────────────────────

class AnnulusPlaneNet(nn.Module):
    """
    Lightweight single-encoder 3D CNN for aortic annulus plane regression.

    Architecture:
        Stem  →  4 × ResBlock3D (with SE attention)  →  GAP  →  2 × FC  →  6D

    Output: [nx, ny, nz, cx_norm, cy_norm, cz_norm]
    """

    def __init__(self, dropout_p: float = 0.3, use_se: bool = True, use_cbam: bool = False,
                 use_coordconv: bool = False, coordconv_r: bool = False):
        super().__init__()
        self.use_se   = use_se
        self.use_cbam = use_cbam
        self.use_coordconv = use_coordconv
        self.coordconv_r   = coordconv_r

        # ── Stem ──
        # Input  (1, S, S, S)  →  (32, S/4, S/4, S/4)
        if use_coordconv:
            from src.coordconv3d import CoordConv3D
            stem_conv = CoordConv3D(1, 32, kernel_size=7, stride=2, padding=3,
                                    bias=False, with_r=coordconv_r)
        else:
            stem_conv = nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem = nn.Sequential(
            stem_conv,
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # ── Encoder ──
        self.layer1 = ResBlock3D(32,  64,  stride=2, use_se=use_se, use_cbam=use_cbam)  # S/8
        self.layer2 = ResBlock3D(64,  128, stride=2, use_se=use_se, use_cbam=use_cbam)  # S/16
        self.layer3 = ResBlock3D(128, 256, stride=2, use_se=use_se, use_cbam=use_cbam)  # S/32

        # ── Head ──
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1     = nn.Linear(256, 128)
        self.fc2     = nn.Linear(128, 6)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_samples: int = 20) -> tuple:
        """
        Monte Carlo Dropout inference.
        Keep dropout ACTIVE during inference → 20 different predictions.
        Mean = best estimate.  Std = uncertainty (confidence score).
        """
        self.train()   # activates dropout
        with torch.no_grad():
            preds = torch.stack([self(x).squeeze(0)
                                 for _ in range(n_samples)])
        self.eval()
        return preds.mean(dim=0), preds.std(dim=0)

    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """
        Return intermediate SE channel weights for interpretability.
        These show which feature channels the model focuses on at each layer.
        """
        weights = {}
        self.eval()
        with torch.no_grad():
            x = self.stem(x)

            # Layer 1
            out = F.relu(self.layer1.bn1(self.layer1.conv1(x)))
            out = self.layer1.bn2(self.layer1.conv2(out))
            if self.use_se:
                b, c = out.shape[:2]
                w1 = self.layer1.se.gap(out).view(b, c)
                w1 = self.layer1.se.fc(w1)
                weights['layer1_se'] = w1.squeeze(0).cpu().numpy()
            x = self.layer1(x)

            # Layer 2
            out = F.relu(self.layer2.bn1(self.layer2.conv1(x)))
            out = self.layer2.bn2(self.layer2.conv2(out))
            if self.use_se:
                b, c = out.shape[:2]
                w2 = self.layer2.se.gap(out).view(b, c)
                w2 = self.layer2.se.fc(w2)
                weights['layer2_se'] = w2.squeeze(0).cpu().numpy()
            x = self.layer2(x)

            # Layer 3
            out = F.relu(self.layer3.bn1(self.layer3.conv1(x)))
            out = self.layer3.bn2(self.layer3.conv2(out))
            if self.use_se:
                b, c = out.shape[:2]
                w3 = self.layer3.se.gap(out).view(b, c)
                w3 = self.layer3.se.fc(w3)
                weights['layer3_se'] = w3.squeeze(0).cpu().numpy()

        return weights


# ─────────────────────────────────────────────
# 4. ARCHITECTURE SUMMARY  (NEW in v2)
# ─────────────────────────────────────────────

def print_architecture_summary(model: AnnulusPlaneNet,
                                input_size: int = 64):
    """
    Print layer-by-layer shape and parameter count.
    Very useful for supervisor meetings and dissertation.
    """
    print("\n" + "="*55)
    print(f"  AnnulusPlaneNet — Architecture Summary")
    print(f"  Input: (1, {input_size}, {input_size}, {input_size})")
    print("="*55)

    groups = {
        'Stem (Conv7 + BN + ReLU + MaxPool)': model.stem,
        'Layer 1  (ResBlock 32→64, stride=2)': model.layer1,
        'Layer 2  (ResBlock 64→128, stride=2)': model.layer2,
        'Layer 3  (ResBlock 128→256, stride=2)': model.layer3,
        'Head (GAP + Dropout + FC256→128→6)': nn.Sequential(
            model.gap, model.dropout, model.fc1, model.fc2
        )
    }

    total = 0
    S = input_size
    shapes = [
        f"({1}, {S}, {S}, {S})",
        f"(32, {S//4}, {S//4}, {S//4})",
        f"(64, {S//8}, {S//8}, {S//8})",
        f"(128, {S//16}, {S//16}, {S//16})",
        f"(256, {S//32}, {S//32}, {S//32})",
        f"(6,)"
    ]

    for i, (name, module) in enumerate(groups.items()):
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total += params
        print(f"\n  {name}")
        print(f"    Output shape : {shapes[i+1]}")
        print(f"    Parameters   : {params:,}")

    print(f"\n{'='*55}")
    print(f"  TOTAL trainable parameters: {total:,}")
    print(f"  SE attention blocks        : {'Yes' if model.use_se else 'No'}")
    print("="*55 + "\n")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 5. QUICK TEST — run: python model.py
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("Testing AnnulusPlaneNet v2 with SE attention...")

    # Test with SE
    model_se = AnnulusPlaneNet(dropout_p=0.3, use_se=True)
    model_se.eval()
    print_architecture_summary(model_se, input_size=64)

    # Forward pass
    dummy = torch.randn(2, 1, 64, 64, 64)
    out   = model_se(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}   values: {out[0].detach().numpy().round(3)}")

    # MC Dropout uncertainty test
    mean, std = model_se.predict_with_uncertainty(dummy[:1], n_samples=10)
    print(f"\nMC Dropout (10 samples):")
    print(f"  Mean : {mean.numpy().round(3)}")
    print(f"  Std  : {std.numpy().round(3)}")
    print(f"  (High std = uncertain, Low std = confident)")

    # SE attention weights
    attn = model_se.get_attention_weights(dummy[:1])
    for k, v in attn.items():
        print(f"\n  {k}: top-5 channels = {v.argsort()[-5:][::-1]}"
              f"  (weights range [{v.min():.3f}, {v.max():.3f}])")

    print("\n✓ All tests passed")


# ─────────────────────────────────────────────
# 6. UNET DECODER BLOCK
# ─────────────────────────────────────────────

class UNetUpBlock3D(nn.Module):
    """
    One U-Net decoder step:
      upsample x2  →  concat skip connection  →  2x Conv3D-BN-ReLU
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='trilinear',
                                align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# 7. SOFT-ARGMAX  (differentiable argmax)
# ─────────────────────────────────────────────

def soft_argmax_3d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3D heatmap (B, 1, D, H, W) to a (B, 3) coordinate
    via softmax-weighted spatial expectation.
    Returns normalised coordinates in [0, 1].
    """
    B, _, D, H, W = heatmap.shape
    h = heatmap.view(B, -1)
    h = torch.softmax(h, dim=-1).view(B, 1, D, H, W)

    device = heatmap.device
    d_coords = torch.linspace(0, 1, D, device=device)
    h_coords = torch.linspace(0, 1, H, device=device)
    w_coords = torch.linspace(0, 1, W, device=device)

    grid_d = d_coords.view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
    grid_h = h_coords.view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
    grid_w = w_coords.view(1, 1, 1, 1, W).expand(B, 1, D, H, W)

    coord_d = (h * grid_d).sum(dim=[2, 3, 4])   # (B, 1)
    coord_h = (h * grid_h).sum(dim=[2, 3, 4])
    coord_w = (h * grid_w).sum(dim=[2, 3, 4])

    return torch.cat([coord_d, coord_h, coord_w], dim=1)  # (B, 3)


# ─────────────────────────────────────────────
# 8. M2 — LANDMARK NET  (Type A approach)
# ─────────────────────────────────────────────

class AnnulusLandmarkNet(nn.Module):
    """
    Type A model: ResNet3D encoder + U-Net decoder → 3 landmark heatmaps
    → soft-argmax → 9D output (RC_D/H/W, NC_D/H/W, LC_D/H/W).

    The annulus plane is then derived geometrically from landmarks:
        centre = (RC + NC + LC) / 3
        normal = normalise(cross(NC-RC, LC-RC))

    Architecture:
        Encoder : stem → layer1 → layer2 → layer3
                  (B,1,64,64,64) → (B,256,2,2,2)
        Decoder : 3x UNetUpBlock with skip connections
                  → (B,32,16,16,16)
        Head    : 3 separate 1x1 Conv → 3 heatmaps (B,1,64,64,64)
                  → soft-argmax → 9D landmarks
    """

    def __init__(self, dropout_p: float = 0.3, use_cbam: bool = True):
        super().__init__()

        # ── Shared encoder (same as AnnulusPlaneNet) ──
        self.stem   = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )  # → (B, 32, 16, 16, 16)

        self.layer1 = ResBlock3D(32,  64,  stride=2,
                                 use_se=False, use_cbam=use_cbam)  # → (B,64,8,8,8)
        self.layer2 = ResBlock3D(64,  128, stride=2,
                                 use_se=False, use_cbam=use_cbam)  # → (B,128,4,4,4)
        self.layer3 = ResBlock3D(128, 256, stride=2,
                                 use_se=False, use_cbam=use_cbam)  # → (B,256,2,2,2)

        # ── U-Net decoder ──
        self.up1 = UNetUpBlock3D(in_ch=256, skip_ch=128, out_ch=128)  # → (B,128,4,4,4)
        self.up2 = UNetUpBlock3D(in_ch=128, skip_ch=64,  out_ch=64)   # → (B,64,8,8,8)
        self.up3 = UNetUpBlock3D(in_ch=64,  skip_ch=32,  out_ch=32)   # → (B,32,16,16,16)

        # Final upsample to full resolution (no skip — stem halved twice)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False),
            nn.Conv3d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )  # → (B,16,64,64,64)

        # ── 3 landmark heatmap heads (one per cusp) ──
        self.head_RC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_NC = nn.Conv3d(16, 1, kernel_size=1)
        self.head_LC = nn.Conv3d(16, 1, kernel_size=1)

        self.dropout = nn.Dropout(p=dropout_p)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            landmarks : (B, 9)  — RC(3) + NC(3) + LC(3) normalised [0,1]
            heatmaps  : (B, 3, D, H, W)  — raw heatmaps for visualisation
        """
        # Encoder
        s0 = self.stem(x)        # (B,32,16,16,16)
        s1 = self.layer1(s0)     # (B,64,8,8,8)
        s2 = self.layer2(s1)     # (B,128,4,4,4)
        s3 = self.layer3(s2)     # (B,256,2,2,2)

        # Decoder
        d1 = self.up1(s3, s2)   # (B,128,4,4,4)
        d2 = self.up2(d1, s1)   # (B,64,8,8,8)
        d3 = self.up3(d2, s0)   # (B,32,16,16,16)
        d4 = self.up4(d3)       # (B,16,64,64,64)

        # Heatmaps
        hm_rc = self.head_RC(d4)  # (B,1,64,64,64)
        hm_nc = self.head_NC(d4)
        hm_lc = self.head_LC(d4)

        # Soft-argmax → landmark coordinates
        rc = soft_argmax_3d(hm_rc)  # (B,3)
        nc = soft_argmax_3d(hm_nc)
        lc = soft_argmax_3d(hm_lc)

        landmarks = torch.cat([rc, nc, lc], dim=1)  # (B,9)
        heatmaps  = torch.cat([hm_rc, hm_nc, hm_lc], dim=1)  # (B,3,D,H,W)

        return landmarks, heatmaps


def landmarks_to_plane(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Derive 6D plane (normal + centre) from 9D landmark predictions.
    Used at inference to convert M2 output to same format as M1.

    Args:
        landmarks : (B, 9)  — RC(3) + NC(3) + LC(3)
    Returns:
        plane     : (B, 6)  — normal(3) + centre(3)
    """
    rc = landmarks[:, 0:3]
    nc = landmarks[:, 3:6]
    lc = landmarks[:, 6:9]

    centre = (rc + nc + lc) / 3.0

    v1 = nc - rc
    v2 = lc - rc
    normal = torch.linalg.cross(v1, v2)
    normal = F.normalize(normal, dim=-1)

    return torch.cat([normal, centre], dim=1)  # (B, 6)
