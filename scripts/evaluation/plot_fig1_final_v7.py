"""
Fig. 1, v7 (all fixes applied):
  1. Aorta segmentation colored red/pink (was light blue)
  2. Axis ticks limited to ~3-4 numbers via MaxNLocator
  3. Dashed zoom-region box drawn on ALL panels (overview + both zoom
     angles), with a "Zoomed region" legend entry
Layout: each of the 3 sample columns shows one large overview panel plus
two smaller zoomed-in sub-panels from two different camera angles.
"""

import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

MERGED_CSV = "results/fig1_merged.csv"
DATA_DIR = "data_processed"
PATIENT_ORDER = ["CONTCT_R_13_FBA", "D1", "CONTCT_R_08_FBA"]
DISPLAY_NAME = {pid: f"Sample {i+1}" for i, pid in enumerate(PATIENT_ORDER)}

AXIS_ORDER = {
    "CONTCT_R_13_FBA": (1, 2, 0),
    "D1": (0, 1, 2),
    "CONTCT_R_08_FBA": (1, 2, 0),
}
AXIS_LABELS = {0: "D", 1: "H", 2: "W"}

MAX_POINTS = 6000
PLANE_HALF_SIZE = 20
ZOOM_HALF = 30

OVERVIEW_VIEW = dict(elev=20, azim=45)
ZOOM_VIEW_A = dict(elev=20, azim=45)
ZOOM_VIEW_B = dict(elev=60, azim=120)

COLOR_BG   = "#ffffff"
COLOR_TEXT = "#000000"
COLOR_FINAL = "#0072B2"
COLOR_GT    = "#2CA02C"
COLOR_M2    = "#E1BE00"
COLOR_CLOUD = "#E63946"   # red; use "#FF6B9D" for pink instead
COLOR_CLOUD_ZOOM = "#F8C6CC"   # much lighter red, zoom panels only
COLOR_BOX   = "#444444"


def get_crop_start(pid, gt_centre_64):
    lm_path = f"{DATA_DIR}/{pid}/visualisation/processed_data/{pid}_processed_landmarks.json"
    with open(lm_path) as f:
        lm = json.load(f)
    geo = lm.get("geometric_properties", {})
    c = np.array(geo.get("center_point", [0, 0, 0]), dtype=np.float32)
    c_dhw = c[[2, 1, 0]] if c.max() > 1.0 else c
    return c_dhw - gt_centre_64 * 2.0


def to_full_space(point_64, crop_start):
    return crop_start + point_64 * 2.0


def load_full_seg_points(pid):
    seg_path = f"{DATA_DIR}/{pid}/visualisation/processed_data/{pid}_processed_seg.nii.gz"
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)
    pts = np.argwhere(arr > 0).astype(float)
    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[idx]
    return pts


def permute(arr, order):
    return arr[..., list(order)]


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


def draw_box(ax, lo, hi, color=COLOR_BOX):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    corners = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                         [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    segs = [(corners[i], corners[j]) for i, j in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=1.2, linestyles="dashed"))


def draw_markers_and_planes(ax, gt_c, gt_n, final_c, final_n, m2_c, m2_n):
    ax.scatter(*gt_c, color=COLOR_GT, s=70, marker="^", edgecolor="black", linewidth=0.8, label="Ground Truth", zorder=6)
    ax.scatter(*final_c, color=COLOR_FINAL, s=70, marker="o", edgecolor="black", linewidth=0.8,
               label="Ours (AortaPlane)", zorder=6)
    ax.scatter(*m2_c, color=COLOR_M2, s=70, marker="D", edgecolor="black", linewidth=0.8,
               label="M2 (Baseline)", zorder=6)

    Xg, Yg, Zg = make_plane_surface(gt_c, gt_n)
    ax.plot_surface(Xg, Yg, Zg, color=COLOR_GT, alpha=0.25, linewidth=0, shade=False)
    fn = final_n if np.dot(final_n, gt_n) >= 0 else -final_n
    Xf, Yf, Zf = make_plane_surface(final_c, fn)
    ax.plot_surface(Xf, Yf, Zf, color=COLOR_FINAL, alpha=0.25, linewidth=0, shade=False)
    mn = m2_n if np.dot(m2_n, gt_n) >= 0 else -m2_n
    Xm, Ym, Zm = make_plane_surface(m2_c, mn)
    ax.plot_surface(Xm, Ym, Zm, color=COLOR_M2, alpha=0.25, linewidth=0, shade=False)

    arrow_len = PLANE_HALF_SIZE * 0.8
    ax.quiver(*gt_c, *(gt_n * arrow_len), color=COLOR_GT, linewidth=2.0, arrow_length_ratio=0.15)
    ax.quiver(*final_c, *(fn * arrow_len), color=COLOR_FINAL, linewidth=2.0, arrow_length_ratio=0.15)
    ax.quiver(*m2_c, *(mn * arrow_len), color=COLOR_M2, linewidth=2.0, arrow_length_ratio=0.15)


def style_axis(ax, view, axis_labels, box_aspect=(1, 1, 1.6)):
    ax.view_init(**view)
    ax.set_facecolor(COLOR_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("#f5f5f5"); pane.set_edgecolor("#cccccc")
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax.set_xlabel(axis_labels[0], color=COLOR_TEXT, fontsize=11)
    ax.set_ylabel(axis_labels[1], color=COLOR_TEXT, fontsize=11)
    ax.set_zlabel(axis_labels[2], color=COLOR_TEXT, fontsize=11)
    ax.set_box_aspect(box_aspect)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))


def main():
    df = pd.read_csv(MERGED_CSV)
    fig = plt.figure(figsize=(18, 6.5))
    fig.patch.set_facecolor(COLOR_BG)
    outer = GridSpec(1, 3, figure=fig, wspace=0.15)

    handles = labels = None
    for col, pid in enumerate(PATIENT_ORDER):
        sub = df[df["patient_id"] == pid]
        gt_row = sub[sub["source"] == "gt"].iloc[0]
        final_row = sub[sub["source"] == "final"].iloc[0]
        m2_row = sub[sub["source"] == "m2"].iloc[0]

        gt_c64 = gt_row[["centre_D", "centre_H", "centre_W"]].to_numpy(dtype=float)
        gt_n = gt_row[["normal_D", "normal_H", "normal_W"]].to_numpy(dtype=float)
        final_c64 = final_row[["centre_D", "centre_H", "centre_W"]].to_numpy(dtype=float)
        final_n = final_row[["normal_D", "normal_H", "normal_W"]].to_numpy(dtype=float)
        m2_c64 = m2_row[["centre_D", "centre_H", "centre_W"]].to_numpy(dtype=float)
        m2_n = m2_row[["normal_D", "normal_H", "normal_W"]].to_numpy(dtype=float)

        crop_start = get_crop_start(pid, gt_c64)
        gt_c = to_full_space(gt_c64, crop_start)
        final_c = to_full_space(final_c64, crop_start)
        m2_c = to_full_space(m2_c64, crop_start)

        pts = load_full_seg_points(pid)

        order = AXIS_ORDER.get(pid, (0, 1, 2))
        pts_p = permute(pts, order)
        gt_c_p, final_c_p, m2_c_p = (permute(v, order) for v in (gt_c, final_c, m2_c))
        gt_n_p, final_n_p, m2_n_p = (permute(v, order) for v in (gt_n, final_n, m2_n))
        axis_labels = tuple(AXIS_LABELS[o] for o in order)

        d0, d1 = gt_c_p[0] - ZOOM_HALF, gt_c_p[0] + ZOOM_HALF
        h0, h1 = gt_c_p[1] - ZOOM_HALF, gt_c_p[1] + ZOOM_HALF
        w0, w1 = gt_c_p[2] - ZOOM_HALF, gt_c_p[2] + ZOOM_HALF
        zoom_mask = ((pts_p[:, 0] >= d0) & (pts_p[:, 0] <= d1) &
                     (pts_p[:, 1] >= h0) & (pts_p[:, 1] <= h1) &
                     (pts_p[:, 2] >= w0) & (pts_p[:, 2] <= w1))
        zoom_pts = pts_p[zoom_mask]

        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[col],
                                         width_ratios=[1.4, 1], wspace=0.05)
        ax_overview = fig.add_subplot(inner[0], projection="3d")
        inner_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=inner[1], hspace=0.25)
        ax_zoom_a = fig.add_subplot(inner_right[0], projection="3d")
        ax_zoom_b = fig.add_subplot(inner_right[1], projection="3d")

        # Overview
        ax_overview.scatter(pts_p[:, 0], pts_p[:, 1], pts_p[:, 2],
                             color=COLOR_CLOUD, s=2, alpha=0.3, label="Segmentation")
        draw_markers_and_planes(ax_overview, gt_c_p, gt_n_p, final_c_p, final_n_p, m2_c_p, m2_n_p)
        draw_box(ax_overview, (d0, h0, w0), (d1, h1, w1))
        style_axis(ax_overview, OVERVIEW_VIEW, axis_labels)
        ax_overview.set_title(DISPLAY_NAME[pid], color=COLOR_TEXT, fontsize=15)

        # Zoom A
        ax_zoom_a.scatter(zoom_pts[:, 0], zoom_pts[:, 1], zoom_pts[:, 2],
                           color=COLOR_CLOUD_ZOOM, s=6, alpha=0.3)
        draw_markers_and_planes(ax_zoom_a, gt_c_p, gt_n_p, final_c_p, final_n_p, m2_c_p, m2_n_p)
     
        ax_zoom_a.set_xlim(d0, d1); ax_zoom_a.set_ylim(h0, h1); ax_zoom_a.set_zlim(w0, w1)
        style_axis(ax_zoom_a, ZOOM_VIEW_A, axis_labels, box_aspect=(1, 1, 1))
        ax_zoom_a.set_title("Zoomed View", color=COLOR_TEXT, fontsize=12)

        # Zoom B
        ax_zoom_b.scatter(zoom_pts[:, 0], zoom_pts[:, 1], zoom_pts[:, 2],
                           color=COLOR_CLOUD_ZOOM, s=6, alpha=0.3)
        draw_markers_and_planes(ax_zoom_b, gt_c_p, gt_n_p, final_c_p, final_n_p, m2_c_p, m2_n_p)
        
        ax_zoom_b.set_xlim(d0, d1); ax_zoom_b.set_ylim(h0, h1); ax_zoom_b.set_zlim(w0, w1)
        style_axis(ax_zoom_b, ZOOM_VIEW_B, axis_labels, box_aspect=(1, 1, 1))
        ax_zoom_b.set_title("Zoomed View", color=COLOR_TEXT, fontsize=12)

        if col == 0:
            handles, labels = ax_zoom_a.get_legend_handles_labels()

    zoom_handle = Line2D([0], [0], color=COLOR_BOX, linestyle="dashed", linewidth=1.2)
    handles = list(handles) + [zoom_handle]
    labels = list(labels) + ["Zoomed region"]

    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = "results/fig1_three_samples_v7.png"
    fig.savefig(out_path, dpi=250, facecolor=COLOR_BG, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()