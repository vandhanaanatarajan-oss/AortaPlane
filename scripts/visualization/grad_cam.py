import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import AnnulusPlaneNet

MANIFEST_PATH = "dataset_manifest.csv"
RESULTS_DIR   = "results/gradcam"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_FOLD = {
    "CONTCT_R_08_FBA": 3,
    "CONTCT_R_27_FBA_no_extended_seg": 3,
    "D8": 1,
    "K10": 1,
    "R10": 0,
}
TEST_PATIENTS = list(BEST_FOLD.keys())
os.makedirs(RESULTS_DIR, exist_ok=True)

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, volume_tensor, gt_normal=None):
        self.model.eval()
        output = self.model(volume_tensor)
        pred_normal = output[0, :3]
        pred_normal_norm = pred_normal / (pred_normal.norm() + 1e-8)
        if gt_normal is not None:
            gt_n = torch.tensor(gt_normal, dtype=torch.float32, device=DEVICE)
            gt_n = gt_n / (gt_n.norm() + 1e-8)
            scalar = torch.abs((pred_normal_norm * gt_n).sum())
        else:
            scalar = pred_normal_norm.norm()
        self.model.zero_grad()
        scalar.backward()
        weights = self.gradients.mean(dim=(2,3,4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(64,64,64), mode="trilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, output[0].detach().cpu().numpy()

def plot_summary(vol, cam, pid, angle_err, gt_n, pred_n):
    mid = 32
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a1a")
    for ax, (vs, cs, label) in zip(axes, [
        (vol[mid,:,:], cam[mid,:,:], "Axial"),
        (vol[:,mid,:], cam[:,mid,:], "Coronal"),
        (vol[:,:,mid], cam[:,:,mid], "Sagittal"),
    ]):
        ax.set_facecolor("#1a1a1a")
        ax.imshow(vs, cmap="gray", vmin=0, vmax=1)
        ax.imshow(cs, cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax.set_title(label, color="white", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Grad-CAM — {pid}   |   Angle error: {angle_err:.2f}\n"
        f"GT normal: [{gt_n[0]:.3f}, {gt_n[1]:.3f}, {gt_n[2]:.3f}]   "
        f"Pred: [{pred_n[0]:.3f}, {pred_n[1]:.3f}, {pred_n[2]:.3f}]",
        color="white", fontsize=10)
    plt.tight_layout()
    return fig

def plot_slices(vol, cam, pid, angle_err, plane):
    n = 5
    fig, axes = plt.subplots(2, n, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a1a")
    idxs = np.linspace(10, 54, n, dtype=int)
    for col, idx in enumerate(idxs):
        if plane == "axial":
            vs, cs = vol[idx,:,:], cam[idx,:,:]
        elif plane == "coronal":
            vs, cs = vol[:,idx,:], cam[:,idx,:]
        else:
            vs, cs = vol[:,:,idx], cam[:,:,idx]
        axes[0,col].imshow(vs, cmap="gray", vmin=0, vmax=1)
        axes[0,col].set_title(f"{idx}", color="white", fontsize=9)
        axes[0,col].axis("off")
        axes[1,col].imshow(vs, cmap="gray", vmin=0, vmax=1)
        axes[1,col].imshow(cs, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[1,col].axis("off")
    fig.suptitle(f"{pid} | {plane} | angle={angle_err:.2f}°", color="white", fontsize=11)
    plt.tight_layout()
    return fig

def main():
    print(f"Device: {DEVICE}")
    manifest  = pd.read_csv(MANIFEST_PATH)
    test_rows = manifest[manifest["patient_id"].isin(TEST_PATIENTS)].set_index("patient_id")
    try:
        ens = pd.read_csv("results/ensemble_results.csv").set_index("patient_id")
    except FileNotFoundError:
        ens = None

    for pid in TEST_PATIENTS:
        if pid not in test_rows.index:
            print(f"SKIP {pid}"); continue
        fold = BEST_FOLD[pid]
        ckpt_path = f"checkpoints/M1_fold{fold}/M1_fold{fold}_best.pth"
        print(f"\n── {pid}  (fold {fold}) ──")
        model = AnnulusPlaneNet(use_cbam=True).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
        gradcam = GradCAM3D(model, target_layer=model.layer2)
        row    = test_rows.loc[pid]
        vol_np = np.load(row["image_path"]).astype(np.float32)
        vol_t  = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        gt_n   = np.array([row["normal_D"], row["normal_H"], row["normal_W"]], dtype=np.float32)
        gt_n   = gt_n / (np.linalg.norm(gt_n) + 1e-8)
        cam, pred_out = gradcam.compute(vol_t, gt_normal=gt_n)
        pred_n = pred_out[:3] / (np.linalg.norm(pred_out[:3]) + 1e-8)
        angle_err = float(ens.loc[pid, "m1_angle"]) if ens is not None and pid in ens.index else 0.0
        print(f"  angle={angle_err:.2f}°  cam_range=[{cam.min():.3f},{cam.max():.3f}]")
        np.save(os.path.join(RESULTS_DIR, f"{pid}_cam.npy"), cam)
        for plane in ["axial", "coronal", "sagittal"]:
            fig = plot_slices(vol_np, cam, pid, angle_err, plane)
            fig.savefig(os.path.join(RESULTS_DIR, f"{pid}_{plane}.png"), dpi=120, bbox_inches="tight", facecolor="#1a1a1a")
            plt.close(fig)
        fig = plot_summary(vol_np, cam, pid, angle_err, gt_n, pred_n)
        fig.savefig(os.path.join(RESULTS_DIR, f"{pid}_summary.png"), dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        print(f"  Saved 4 figures to {RESULTS_DIR}/")

    print(f"\nDone. Download with:")
    print(f"  scp -r ec25088@<hpc>:/data/DERI-ecgai/__users/Vandhanaa/results/gradcam .")

if __name__ == "__main__":
    main()
