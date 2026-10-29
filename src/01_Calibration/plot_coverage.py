"""Where in the image the calibration had corners, from the files Calibration.py writes.

    src/venv/bin/python src/01_Calibration/plot_coverage.py results/Calibration/calibration/FullCalibration_water_final_merge_stereo.yaml
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import rel  # noqa: E402

CAMERAS = ("left", "right")
RATE_RAMP = ["#e9e8e4", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
KEPT, DROPPED = "#2a78d6", "#d03b3b"
SURFACE, INK, MUTED = "#fcfcfb", "#2b2b2a", "#8a8a86"
N_TAGS = 70
CELL_PX = 100                # 1600 x 1200 -> a 16 x 12 grid of cells


def read_run(path):
    """(width, height, {camera: {image: {tag id: 4x2 corners}}}, {camera: {(image, tag id)} dropped})."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    width = int(fs.getNode("image").getNode("width_px").real())
    height = int(fs.getNode("image").getNode("height_px").real())
    dropped = {}
    for camera in CAMERAS:
        node = fs.getNode("outlier_rejection").getNode(camera).getNode("rejected_tags")
        entries = [] if node.isNone() else [node.at(i).string() for i in range(node.size())]
        dropped[camera] = {(e.split("|")[0], int(e.split("|")[1])) for e in entries}
    fs.release()

    corners_path = path.replace("_stereo.yaml", "_corners.json")
    with open(corners_path) as f:
        cameras = json.load(f)["cameras"]
    tags = {camera: {name: {i: np.array(c).reshape(4, 2) for i, c in zip(d["ids"], d["corners"])
                            if 0 <= i < N_TAGS}
                     for name, d in cameras[camera].items()}
            for camera in CAMERAS}
    return width, height, tags, dropped


def cell_counts(width, height, images):
    """Per cell, the number of images with at least one kept corner in it."""
    nx, ny = -(-width // CELL_PX), -(-height // CELL_PX)
    counts = np.zeros((ny, nx), int)
    for points in images:
        if len(points):
            cx = np.clip((points[:, 0] // CELL_PX).astype(int), 0, nx - 1)
            cy = np.clip((points[:, 1] // CELL_PX).astype(int), 0, ny - 1)
            hit = np.zeros_like(counts, bool)
            hit[cy, cx] = True
            counts += hit
    return counts


def plot(path):
    width, height, tags, dropped = read_run(path)
    name = os.path.basename(path).replace("_stereo.yaml", "")
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.28, "wspace": 0.12})
    cmap = LinearSegmentedColormap.from_list("rate", RATE_RAMP)
    summary = []

    for row, camera in enumerate(CAMERAS):
        kept, lost, per_image = [], [], []
        for image, by_id in tags[camera].items():
            ok = [c for i, c in by_id.items() if (image, i) not in dropped[camera]]
            lost += [c for i, c in by_id.items() if (image, i) in dropped[camera]]
            kept += ok
            per_image.append(np.concatenate(ok) if ok else np.empty((0, 2)))
        kept = np.concatenate(kept) if kept else np.empty((0, 2))
        lost = np.concatenate(lost) if lost else np.empty((0, 2))
        counts = cell_counts(width, height, per_image)
        n_images = len(per_image)
        summary.append(f"{camera}: {np.mean(counts > 0):.0%} of cells reached, "
                       f"{np.mean(counts >= 3):.0%} by 3+ images")

        # Every corner used, and the ones rejection dropped.
        ax = axes[row, 0]
        ax.set_facecolor(SURFACE)
        ax.scatter(kept[:, 0], kept[:, 1], s=1, color=KEPT, alpha=0.25, lw=0,
                   label=f"used ({len(kept) // 4} tags)")
        if len(lost):
            ax.scatter(lost[:, 0], lost[:, 1], s=9, color=DROPPED, lw=0,
                       label=f"dropped by outlier rejection ({len(lost) // 4} tags)")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{camera}: every tag corner, {n_images} images", loc="left")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, markerscale=4)

        # Images per cell.
        ax = axes[row, 1]
        ax.set_facecolor(SURFACE)
        image = ax.imshow(counts, cmap=cmap, vmin=0, vmax=n_images, extent=(0, width, height, 0))
        for (y, x), n in np.ndenumerate(counts):
            ax.text((x + 0.5) * CELL_PX, (y + 0.5) * CELL_PX, str(n), ha="center", va="center",
                    fontsize=6.5, color=SURFACE if n > 0.55 * n_images else INK)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_title(f"{camera}: images with a corner in each {CELL_PX} px cell", loc="left")
        fig.colorbar(image, ax=ax, shrink=0.8, label="images")

    fig.suptitle(f"{name}: corner coverage.  " + ";  ".join(summary), x=0.01, ha="left", fontsize=11)
    out = path.replace("_stereo.yaml", "_coverage.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel(out)}")
    print("  " + "\n  ".join(summary))


def main():
    ap = argparse.ArgumentParser(description="Plot where calibration corners fell in the image.")
    ap.add_argument("yamls", nargs="+", help="calibration YAMLs written by Calibration.py")
    for path in ap.parse_args().yamls:
        plot(path)


if __name__ == "__main__":
    main()
