import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

MANIFEST_PATH = "dataset_manifest.csv"
PRED_PATH     = "results/per_fold_predictions.csv"
RESULTS_DIR   = "results/uncertainty_heatmaps"
MODELS        = ["M1", "M2"]
os.makedirs(RESULTS_DIR, exist_ok=True)


def align_normals(normals):
    """Flip sign of normals so they all point roughly the same way before
    measuring spread (a normal and its negation represent the same plane)."""
    ref = normals[0]
    return np.array([n if np.dot(n, ref) >= 0 else -n for n in normals])


def angle_between(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.degrees(np.arccos(np.clip(np.abs(np.dot(a, b)), 0, 1)))


def plot_centre_uncertainty(ax, vol_slice, centres, mean_centre, gt_centre, mid):
    """2D CT slice with the 5 fold-predicted centres overlaid, coloured by
    distance from the mean (jet colourmap = the 'heat')."""
    dists = np.linalg.norm(centres - mean_centre, axis=1)
    ax.imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
    sc = ax.scatter(centres[:, 2], centres[:, 1], c=dists, cmap="jet",
                     s=90, edgecolors="white", linewidths=0.8, vmin=0, vmax=max(dists.max(), 1e-3))
    ax.scatter(mean_centre[2], mean_centre[1], marker="D", s=70,
               facecolors="none", edgecolors="lime", linewidths=1.5, label="mean pred")
    ax.scatter(gt_centre[2], gt_centre[1], marker="*", s=180,
               facecolors="yellow", edgecolors="black", linewidths=0.8, label="GT")
    ax.set_xlim(0, vol_slice.shape[1]); ax.set_ylim(vol_slice.shape[0], 0)
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a1a",
              edgecolor="#444444", labelcolor="white")
    return sc, dists


def plot_normal_uncertainty(ax3d, normals, mean_normal, gt_normal):
    """3D quiver of the 5 fold-predicted normal directions from a common
    origin, coloured by angular deviation from the mean (jet colourmap)."""
    angles = np.array([angle_between(n, mean_normal) for n in normals])
    norm_a = angles / (angles.max() + 1e-8) if angles.max() > 1e-8 else angles
    cmap = plt.get_cmap("jet")
    for n, a in zip(normals, norm_a):
        ax3d.quiver(0, 0, 0, n[0], n[1], n[2], color=cmap(a), linewidth=2, arrow_length_ratio=0.15)
    ax3d.quiver(0, 0, 0, mean_normal[0], mean_normal[1], mean_normal[2],
                color="lime", linewidth=2.5, arrow_length_ratio=0.15)
    gt_n = gt_normal / (np.linalg.norm(gt_normal) + 1e-8)
    ax3d.quiver(0, 0, 0, gt_n[0], gt_n[1], gt_n[2],
                color="yellow", linewidth=2.5, arrow_length_ratio=0.15)
    ax3d.set_xlim([-1, 1]); ax3d.set_ylim([-1, 1]); ax3d.set_zlim([-1, 1])
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.set_facecolor("#1a1a1a")
        pane.set_edgecolor("#444444")
    ax3d.grid(False)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="lime", lw=2.5, label="mean pred"),
        Line2D([0], [0], color="yellow", lw=2.5, label="GT"),
    ]
    ax3d.legend(handles=legend_elems, loc="upper left", fontsize=7,
                facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white")
    return angles


def main():
    manifest = pd.read_csv(MANIFEST_PATH).set_index("patient_id")
    preds = pd.read_csv(PRED_PATH)

    patients = sorted(preds["patient_id"].unique())
    print(f"Patients: {patients}")

    for pid in patients:
        if pid not in manifest.index:
            print(f"SKIP {pid} — not in manifest"); continue

        vol = np.load(manifest.loc[pid, "image_path"]).astype(np.float32)
        mid = vol.shape[0] // 2
        vol_slice = vol[mid, :, :]

        fig = plt.figure(figsize=(11, 9))
        fig.patch.set_facecolor("#1a1a1a")
        fig.suptitle(f"Uncertainty heatmap — {pid}", color="white", fontsize=13)

        for row, model in enumerate(MODELS):
            g = preds[(preds["patient_id"] == pid) & (preds["model"] == model)]
            if len(g) == 0:
                print(f"  SKIP {pid}/{model} — no rows"); continue

            centres = g[["pred_centre_D", "pred_centre_H", "pred_centre_W"]].values
            normals = g[["pred_normal_D", "pred_normal_H", "pred_normal_W"]].values
            gt_centre = g[["gt_centre_D", "gt_centre_H", "gt_centre_W"]].values[0]
            gt_normal = g[["gt_normal_D", "gt_normal_H", "gt_normal_W"]].values[0]

            normals_aligned = align_normals(normals)
            mean_centre = centres.mean(axis=0)
            mean_normal = normals_aligned.mean(axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

            ax_c = fig.add_subplot(2, 2, row * 2 + 1)
            ax_c.set_facecolor("#1a1a1a")
            sc, dists = plot_centre_uncertainty(ax_c, vol_slice, centres, mean_centre, gt_centre, mid)
            ax_c.set_title(f"{model} — centre spread (mm-equiv, max={dists.max()*2:.1f}mm)",
                            color="white", fontsize=10)

            ax_n = fig.add_subplot(2, 2, row * 2 + 2, projection="3d")
            ax_n.set_facecolor("#1a1a1a")
            angles = plot_normal_uncertainty(ax_n, normals_aligned, mean_normal, gt_normal)
            ax_n.set_title(f"{model} — normal spread (max={angles.max():.1f}°)",
                            color="white", fontsize=10)

            cbar = fig.colorbar(sc, ax=ax_c, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.set_label("distance from mean (voxels)", color="white", fontsize=8)
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(RESULTS_DIR, f"{pid}_uncertainty.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#1a1a1a")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Download with:")
    print(f"  scp -r ec25088@<hpc>:/data/DERI-ecgai/__users/Vandhanaa/{RESULTS_DIR} .")


if __name__ == "__main__":
    main()
