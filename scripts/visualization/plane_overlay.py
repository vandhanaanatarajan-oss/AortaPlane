import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MANIFEST_PATH = "dataset_manifest.csv"
PRED_PATH     = "results/per_fold_predictions.csv"
RESULTS_DIR   = "results/plane_overlays"
MODELS        = ["M1", "M2"]
VOL_SIZE      = 64
DISC_RADIUS   = 12.0  # voxels, ~24mm diameter — roughly annulus scale
os.makedirs(RESULTS_DIR, exist_ok=True)


def orthonormal_basis(n):
    n = n / (np.linalg.norm(n) + 1e-8)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    return u, v


def plane_slice_intersection(c, n, axis, slice_idx, extent=45):
    u, v = orthonormal_basis(n)
    t = np.linspace(-extent, extent, 200)
    if abs(u[axis]) > abs(v[axis]):
        if abs(u[axis]) < 1e-6:
            return None
        s = (slice_idx - c[axis] - t * v[axis]) / u[axis]
        pts = c[None, :] + s[:, None] * u[None, :] + t[:, None] * v[None, :]
    else:
        if abs(v[axis]) < 1e-6:
            return None
        s = (slice_idx - c[axis] - t * u[axis]) / v[axis]
        pts = c[None, :] + t[:, None] * u[None, :] + s[:, None] * v[None, :]
    other_axes = [a for a in range(3) if a != axis]
    a1, a2 = pts[:, other_axes[0]], pts[:, other_axes[1]]
    mask = (a1 >= 0) & (a1 < VOL_SIZE) & (a2 >= 0) & (a2 < VOL_SIZE)
    if mask.sum() < 2:
        return None
    return a1[mask], a2[mask]


def plot_slices_with_planes(vol, pred_c, pred_n, gt_c, gt_n, pid, model):
    mid = VOL_SIZE // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.patch.set_facecolor("#1a1a1a")
    slice_defs = [
        (0, mid, vol[mid, :, :], "Axial"),
        (1, mid, vol[:, mid, :], "Coronal"),
        (2, mid, vol[:, :, mid], "Sagittal"),
    ]
    for ax, (axis, idx, vs, label) in zip(axes, slice_defs):
        ax.set_facecolor("#1a1a1a")
        ax.imshow(vs, cmap="gray", vmin=0, vmax=1)
        pred_line = plane_slice_intersection(pred_c, pred_n, axis, idx)
        gt_line   = plane_slice_intersection(gt_c, gt_n, axis, idx)
        if pred_line is not None:
            ax.plot(pred_line[1], pred_line[0], color="cyan", lw=2, label="predicted plane")
        if gt_line is not None:
            ax.plot(gt_line[1], gt_line[0], color="yellow", lw=2, linestyle="--", label="GT plane")
        ax.set_xlim(0, VOL_SIZE); ax.set_ylim(VOL_SIZE, 0)
        ax.set_title(label, color="white", fontsize=11)
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
                    edgecolor="#444444", labelcolor="white")
    fig.suptitle(f"{model} — Predicted vs GT plane — {pid}", color="white", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_disc(c, n, radius=DISC_RADIUS, n_pts=40):
    u, v = orthonormal_basis(n)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    circle = c[None, :] + radius * (np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * v[None, :])
    return circle


def plot_3d_planes(pred_c, pred_n, gt_c, gt_n, pid, model):
    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor("#1a1a1a")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a1a")

    pred_disc = make_disc(pred_c, pred_n)
    gt_disc   = make_disc(gt_c, gt_n)
    ax.plot(pred_disc[:, 0], pred_disc[:, 1], pred_disc[:, 2], color="cyan", lw=2.2)
    ax.plot(gt_disc[:, 0], gt_disc[:, 1], gt_disc[:, 2], color="yellow", lw=2.2)
    ax.scatter(*pred_c, color="cyan", s=40)
    ax.scatter(*gt_c, color="yellow", s=40)
    ax.quiver(*pred_c, *pred_n, length=15, color="cyan", linewidth=2)
    ax.quiver(*gt_c, *gt_n, length=15, color="yellow", linewidth=2)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#1a1a1a"); pane.set_edgecolor("#444444")
    ax.set_xlim(0, VOL_SIZE); ax.set_ylim(0, VOL_SIZE); ax.set_zlim(0, VOL_SIZE)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(f"{model} — 3D plane comparison — {pid}", color="white", fontsize=11)
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], color="cyan", lw=3, label="predicted"),
                    Line2D([0], [0], color="yellow", lw=3, label="GT")]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8,
              facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white")
    return fig


def main():
    manifest = pd.read_csv(MANIFEST_PATH).set_index("patient_id")
    preds = pd.read_csv(PRED_PATH)
    patients = sorted(preds["patient_id"].unique())
    print(f"Patients: {patients}")

    for pid in patients:
        if pid not in manifest.index:
            print(f"SKIP {pid}"); continue
        vol = np.load(manifest.loc[pid, "image_path"]).astype(np.float32)

        for model in MODELS:
            g = preds[(preds["patient_id"] == pid) & (preds["model"] == model)]
            if len(g) == 0:
                continue
            centres = g[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].values
            normals = g[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].values
            gt_c = g[["gt_centre_D", "gt_centre_H", "gt_centre_W"]].values[0]
            gt_n = g[["gt_normal_D", "gt_normal_H", "gt_normal_W"]].values[0]
            gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)

            ref = normals[0]
            normals_aligned = np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])
            pred_c = centres.mean(axis=0)
            pred_n = normals_aligned.mean(axis=0)
            pred_n = pred_n / (np.linalg.norm(pred_n) + 1e-8)

            fig1 = plot_slices_with_planes(vol, pred_c, pred_n, gt_c, gt_n, pid, model)
            out1 = os.path.join(RESULTS_DIR, f"{pid}_{model}_slices.png")
            fig1.savefig(out1, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
            plt.close(fig1)

            fig2 = plot_3d_planes(pred_c, pred_n, gt_c, gt_n, pid, model)
            out2 = os.path.join(RESULTS_DIR, f"{pid}_{model}_3d.png")
            fig2.savefig(out2, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
            plt.close(fig2)

            print(f"  Saved: {out1}")
            print(f"  Saved: {out2}")

    print(f"\nDone. Download with:")
    print(f"  scp -r ec25088@login.hpc.qmul.ac.uk:/data/DERI-ecgai/__users/Vandhanaa/{RESULTS_DIR} .")


if __name__ == "__main__":
    main()
