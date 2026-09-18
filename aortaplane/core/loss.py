"""
loss.py  — IMPROVED VERSION v2
---------------------------------
Key improvements over v1:
  1. Huber loss option for centre   — more robust to outlier patients
  2. Per-component logging          — logs angular and centre separately to WandB
  3. Detailed metric report         — prints mean/std/median for both metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PlaneLoss(nn.Module):
    """
    Combined loss for annulus plane regression.

    angular_loss  = 1 - |cos θ|       (normal vector direction)
    centre_loss   = MSE or Huber       (normalised centre point)
    total         = w_normal * angular + w_centre * centre
    """

    def __init__(self,
                 w_normal: float = 1.0,
                 w_centre: float = 1.0,
                 use_huber: bool = False,
                 huber_delta: float = 0.1,
                 eps: float = 1e-8):
        super().__init__()
        self.w_normal    = w_normal
        self.w_centre    = w_centre
        self.use_huber   = use_huber
        self.huber_delta = huber_delta
        self.eps         = eps

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple:
        pred_n,   pred_c   = pred[:, :3],   pred[:, 3:]
        target_n, target_c = target[:, :3], target[:, 3:]

        # Angular loss
        pred_n_unit  = F.normalize(pred_n, dim=1, eps=self.eps)
        cos_sim      = (pred_n_unit * target_n).sum(dim=1)
        angular_loss = (1.0 - cos_sim.abs()).mean()

        # Centre loss
        if self.use_huber:
            centre_loss = F.huber_loss(pred_c, target_c,
                                        delta=self.huber_delta)
        else:
            centre_loss = F.mse_loss(pred_c, target_c)

        total = self.w_normal * angular_loss + self.w_centre * centre_loss

        components = {
            'total'       : total.item(),
            'angular'     : angular_loss.item(),
            'centre'      : centre_loss.item(),
            'w_normal'    : self.w_normal,
            'w_centre'    : self.w_centre,
        }
        return total, components


# ─────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────

def angle_error_degrees(pred_n: torch.Tensor,
                         gt_n:   torch.Tensor,
                         eps:    float = 1e-8) -> torch.Tensor:
    """Angle between predicted and GT normal in degrees [0, 90]."""
    pred_u  = F.normalize(pred_n, dim=1, eps=eps)
    cos_sim = (pred_u * gt_n).sum(dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_sim.abs()))


def centre_distance_voxels(pred_c_norm: torch.Tensor,
                            gt_c_norm:   torch.Tensor,
                            image_size:  int) -> torch.Tensor:
    """Euclidean distance in voxels."""
    return ((pred_c_norm - gt_c_norm) * image_size).norm(dim=1)


def print_metrics_report(angle_errors: list,
                          centre_dists: list,
                          label: str = "Test set"):
    """Print a clean metrics report for supervisor meetings."""
    a = np.array(angle_errors)
    c = np.array(centre_dists)
    print(f"\n{'='*50}")
    print(f"  METRICS REPORT — {label}")
    print(f"{'='*50}")
    print(f"  N patients : {len(a)}")
    print(f"\n  Angle error (degrees):")
    print(f"    Mean   : {a.mean():.2f}°")
    print(f"    Std    : {a.std():.2f}°")
    print(f"    Median : {np.median(a):.2f}°")
    print(f"    Min    : {a.min():.2f}°")
    print(f"    Max    : {a.max():.2f}°")
    print(f"    < 5°   : {(a < 5).sum()}/{len(a)} patients  "
          f"({100*(a<5).mean():.0f}%)")
    print(f"\n  Centre distance (voxels):")
    print(f"    Mean   : {c.mean():.2f} vx")
    print(f"    Std    : {c.std():.2f} vx")
    print(f"    Median : {np.median(c):.2f} vx")
    print(f"    Min    : {c.min():.2f} vx")
    print(f"    Max    : {c.max():.2f} vx")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    loss_fn = PlaneLoss(w_normal=1.0, w_centre=1.0, use_huber=False)
    B = 4
    pred   = torch.randn(B, 6)
    n      = F.normalize(torch.randn(B, 3), dim=1)
    target = torch.cat([n, torch.rand(B, 3)], dim=1)
    total, comps = loss_fn(pred, target)
    print(f"Loss: {total.item():.4f}  components: {comps}")

    angles = angle_error_degrees(pred[:, :3], target[:, :3])
    dists  = centre_distance_voxels(pred[:, 3:], target[:, 3:], 64)
    print_metrics_report(angles.numpy().tolist(),
                          dists.numpy().tolist(), "Dummy test")


# ─────────────────────────────────────────────
# LANDMARK LOSS  (for M2 AnnulusLandmarkNet)
# ─────────────────────────────────────────────

class LandmarkPlaneLoss(nn.Module):
    """
    Combined loss for Type A (landmark) approach.

    Problem: manifest only has centre+normal (6D), not RC/NC/LC separately.
    Solution: supervise the DERIVED plane from predicted landmarks
              against GT plane — same angular + centre MSE as M1.
    Also adds a geometric consistency term: the three predicted landmarks
    should not collapse to the same point (spread regularisation).

    Args:
        w_normal  : weight for angular normal loss
        w_centre  : weight for centre MSE loss
        w_spread  : weight for landmark spread regularisation
    """

    def __init__(self, w_normal: float = 1.0,
                 w_centre: float = 1.0,
                 w_spread: float = 0.1):
        super().__init__()
        self.w_normal = w_normal
        self.w_centre = w_centre
        self.w_spread = w_spread

    def forward(self, landmarks: torch.Tensor,
                targets: torch.Tensor):
        """
        Args:
            landmarks : (B, 9)  predicted RC+NC+LC from AnnulusLandmarkNet
            targets   : (B, 6)  GT normal(3) + centre(3) from manifest
        Returns:
            loss       : scalar
            components : dict of individual loss terms
        """
        import torch.nn.functional as F

        rc = landmarks[:, 0:3]
        nc = landmarks[:, 3:6]
        lc = landmarks[:, 6:9]

        # Derive plane from predicted landmarks
        pred_centre = (rc + nc + lc) / 3.0
        v1 = nc - rc
        v2 = lc - rc
        pred_normal = torch.linalg.cross(v1, v2)
        pred_normal = F.normalize(pred_normal, dim=-1)

        # GT plane
        gt_normal = targets[:, :3]
        gt_centre = targets[:, 3:]

        # Angular loss (sign-ambiguity safe)
        cos_sim = (pred_normal * gt_normal).sum(dim=-1).abs()
        angular_loss = (1.0 - cos_sim).mean()

        # Centre MSE loss
        centre_loss = F.mse_loss(pred_centre, gt_centre)

        # Spread regularisation — penalise if landmarks collapse
        # Ideal: RC, NC, LC are spread apart; if they converge loss increases
        d_rc_nc = (rc - nc).norm(dim=-1)
        d_nc_lc = (nc - lc).norm(dim=-1)
        d_rc_lc = (rc - lc).norm(dim=-1)
        spread = (d_rc_nc + d_nc_lc + d_rc_lc) / 3.0
        spread_loss = torch.clamp(0.1 - spread, min=0.0).mean()

        total = (self.w_normal * angular_loss
               + self.w_centre * centre_loss
               + self.w_spread * spread_loss)

        return total, {
            'total'   : total.item(),
            'angular' : angular_loss.item(),
            'centre'  : centre_loss.item(),
            'spread'  : spread_loss.item(),
        }
