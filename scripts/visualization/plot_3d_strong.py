"""
Item 6 simplified: single full-slice panel per example (no separate
zoomed panel), with a white box highlighting the marker region for
visual reference, pushed to high resolution for conference print quality.

USAGE (Apocrita, project root):
    PYTHONPATH=./src python plot_3d_strong.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

VOLUME_DIR   = "data_processed_128"
VOLUME_SCALE = 2.0
RESULTS_DIR  = "results/plane_3d_strong"
PATIENTS     = ["CONTCT_R_08_FBA", "CONTCT_R_27_FBA_no_extended_seg", "D8", "K10", "R10"]
MAX_POINTS   = 3000
PLANE_HALF_SIZE = 20

COLOR_BG   = "#ffffff"
COLOR_TEXT = "#000000"
COLOR_PRED = "#0072B2"
COLOR_GT   = "#2CA02C"
COLOR_CLOUD = "#87CEEB"
COLOR_M2    = "#E1BE00"   # gold/yellow, distinct from GT green and M1 blue

os.makedirs(RESULTS_DIR, exist_ok=True)

ens = pd.read_csv("results/ensembled_test_predictions.csv")


def load_seg_points(pid):
    candidates = glob.glob(f"{VOLUME_DIR}/{pid}/*_seg_resized_128.npy")
    if not candidates:
        raise FileNotFoundError(f"No segmentation mask found for {pid}")
    seg = np.load(candidates[0])
    pts = np.argwhere(seg > 0.5)
    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[idx]
    return pts


def orthonormal_basis(n):
    n = n / (np.linalg.norm(n) + 1e-8)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper); u = u / (np.linalg.norm(u) + 1e-8)
    v = np.cross(n, u)
    return u, v


def make_plane_surface(c, n, half_size=PLANE_HALF_SIZE, n_grid=10):
    u, v = orthonormal_basis(n)
    s = np.linspace(-half_size, half_size, n_grid)
    t = np.linspace(-half_size, half_size, n_grid)
    S, T = np.meshgrid(s, t)
    X = c[0] + S * u[0] + T * v[0]
    Y = c[1] + S * u[1] + T * v[1]
    Z = c[2] + S * u[2] + T * v[2]
    return X, Y, Z


def draw_panel(ax, pts, gt_c, gt_n, pred_c, pred_n, m2_c, m2_n, show_plane=True, elev=20, azim=45):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=COLOR_CLOUD, s=3, alpha=0.35, label="Segmentation")

    ax.scatter(*gt_c, color=COLOR_GT, s=60, marker="^", edgecolor="black", linewidth=0.8, label="GT centre", zorder=6)
    ax.scatter(*pred_c, color=COLOR_PRED, s=60, marker="s", edgecolor="black", linewidth=0.8, label="M1 centre", zorder=6)
    ax.scatter(*m2_c, color=COLOR_M2, s=60, marker="D", edgecolor="black", linewidth=0.8, label="M2 centre", zorder=6)

    if show_plane:
        Xg, Yg, Zg = make_plane_surface(gt_c, gt_n)
        ax.plot_surface(Xg, Yg, Zg, color=COLOR_GT, alpha=0.25, linewidth=0, shade=False)

        pn = pred_n if np.dot(pred_n, gt_n) >= 0 else -pred_n
        Xp, Yp, Zp = make_plane_surface(pred_c, pn)
        ax.plot_surface(Xp, Yp, Zp, color=COLOR_PRED, alpha=0.25, linewidth=0, shade=False)

        m2n = m2_n if np.dot(m2_n, gt_n) >= 0 else -m2_n
        Xm2, Ym2, Zm2 = make_plane_surface(m2_c, m2n)
        ax.plot_surface(Xm2, Ym2, Zm2, color=COLOR_M2, alpha=0.25, linewidth=0, shade=False)

        arrow_len = PLANE_HALF_SIZE * 0.8
        ax.quiver(*gt_c, *(gt_n * arrow_len), color=COLOR_GT, linewidth=2.2, arrow_length_ratio=0.15)
        ax.quiver(*pred_c, *(pn * arrow_len), color=COLOR_PRED, linewidth=2.2, arrow_length_ratio=0.15)
        ax.quiver(*m2_c, *(m2n * arrow_len), color=COLOR_M2, linewidth=2.2, arrow_length_ratio=0.15)

    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(COLOR_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#f5f5f5"); pane.set_edgecolor("#cccccc")
    ax.tick_params(colors=COLOR_TEXT, labelsize=7)
    ax.set_xlabel("D", color=COLOR_TEXT, fontsize=8)
    ax.set_ylabel("H", color=COLOR_TEXT, fontsize=8)
    ax.set_zlabel("W", color=COLOR_TEXT, fontsize=8)


def make_figure(pid):
    pts = load_seg_points(pid)

    row_m1 = ens[(ens.patient_id == pid) & (ens.model == "M1")].iloc[0]
    row_m2 = ens[(ens.patient_id == pid) & (ens.model == "M2")].iloc[0]
    gt_c = np.array([row_m1.gt_centre_D, row_m1.gt_centre_H, row_m1.gt_centre_W]) * VOLUME_SCALE
    gt_n = np.array([row_m1.gt_normal_D, row_m1.gt_normal_H, row_m1.gt_normal_W])
    pred_c = np.array([row_m1.pred_centre_D, row_m1.pred_centre_H, row_m1.pred_centre_W]) * VOLUME_SCALE
    pred_n = np.array([row_m1.pred_normal_D, row_m1.pred_normal_H, row_m1.pred_normal_W])
    m2_c = np.array([row_m2.pred_centre_D, row_m2.pred_centre_H, row_m2.pred_centre_W]) * VOLUME_SCALE
    m2_n = np.array([row_m2.pred_normal_D, row_m2.pred_normal_H, row_m2.pred_normal_W])
    gt_n = gt_n / (np.linalg.norm(gt_n) + 1e-8)
    pred_n = pred_n / (np.linalg.norm(pred_n) + 1e-8)
    m2_n = m2_n / (np.linalg.norm(m2_n) + 1e-8)

    fig = plt.figure(figsize=(13, 6))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(f"{pid} — 3D Segmentation with GT/M1/M2 Plane Comparison", color=COLOR_TEXT, fontsize=13)

    ax1 = fig.add_subplot(121, projection="3d")
    draw_panel(ax1, pts, gt_c, gt_n, pred_c, pred_n, m2_c, m2_n, show_plane=False)
    ax1.set_title("Full View: Segmentation + Centres", color=COLOR_TEXT, fontsize=10)
    ax1.legend(loc="upper left", fontsize=7, facecolor=COLOR_BG, edgecolor="#cccccc", labelcolor=COLOR_TEXT)

    ax2 = fig.add_subplot(122, projection="3d")
    d0, d1 = gt_c[0] - 25, gt_c[0] + 25
    h0, h1 = gt_c[1] - 25, gt_c[1] + 25
    w0, w1 = gt_c[2] - 25, gt_c[2] + 25
    mask = ((pts[:, 0] >= d0) & (pts[:, 0] <= d1) &
            (pts[:, 1] >= h0) & (pts[:, 1] <= h1) &
            (pts[:, 2] >= w0) & (pts[:, 2] <= w1))
    draw_panel(ax2, pts[mask], gt_c, gt_n, pred_c, pred_n, m2_c, m2_n, show_plane=True)
    ax2.set_xlim(d0, d1); ax2.set_ylim(h0, h1); ax2.set_zlim(w0, w1)
    ax2.set_title("Zoomed View: GT vs. M1 vs. M2 Plane + Normal", color=COLOR_TEXT, fontsize=10)
    ax2.legend(loc="upper left", fontsize=7, facecolor=COLOR_BG, edgecolor="#cccccc", labelcolor=COLOR_TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = f"{RESULTS_DIR}/{pid}_3d_strong.png"
    fig.savefig(out_path, dpi=200, facecolor=COLOR_BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


for pid in PATIENTS:
    try:
        make_figure(pid)
    except FileNotFoundError as e:
        print(f"SKIP {pid}: {e}")
