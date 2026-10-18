"""One figure comparing the fusion methods on every ice scan: a row per scan, a column per method, each tile the
top-down height map of that scan's ice box (evaluation.region in its config), seen from the camera side.

Reads what run_fusion.py saved (<work>/<method>/cloud_board.ply) through compare_fusion.py's own functions, so the
tiles match each scan's topdown_region.png. Each row has its own colour scale, fitted to the ice in its box: the
10th to 99.5th percentile of every method's heights there, which leaves out the mounting pole and background that
sit far behind the ice and would otherwise flatten its few centimetres of relief into one colour.

    src/venv/bin/python src/experiments/fusion/ice_grid.py

Writes results/experiments/fusion/ice_grid.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import compare_fusion  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

CONFIGS = ("config_pyramid_shade.json", "config_cubes.json", "config_wave.json")
TITLES = {"ice_pyramid_shade_pyr1_pass4": "Pyramid", "ice_cubes_shade_cub2": "Cubes", "ice_wave_shade_wave1": "Wave"}
HEIGHT_PERCENTILES = (10, 99.5)
# Rows shown before their scan has been fused: (row title, config whose data stands in). The stand-in is marked
# "placeholder" in every tile; replace the entry with the scan's own config once it exists.
PLACEHOLDERS = [("Overhang", "config_cubes.json")]
FONT = {"title": 16, "row": 16, "label": 14, "ticks": 12, "note": 12}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=[os.path.join(HERE, c) for c in CONFIGS])
    ap.add_argument("--out", default=None, help="output PNG (default results/experiments/fusion/ice_grid.png)")
    ap.add_argument("--no-placeholders", action="store_true", help="leave out the PLACEHOLDERS rows")
    args = ap.parse_args(argv)
    rows = [(None, common.load_config(c)) for c in args.configs]
    if not args.no_placeholders:
        rows += [(title, common.load_config(os.path.join(HERE, c))) for title, c in PLACEHOLDERS]
    cfgs = [cfg for _, cfg in rows]
    methods = cfgs[0]["methods"]
    fig, axes = plt.subplots(len(cfgs), len(methods), figsize=(4.6 * len(methods) + 1.6, 4.4 * len(cfgs)),
                             squeeze=False, constrained_layout=True)
    for row, (placeholder, cfg) in enumerate(rows):
        p = recording_paths(cfg)
        ev = cfg["evaluation"]
        if not ev.get("region"):
            raise SystemExit(f"{cfg['recording']['name']}: no evaluation.region set yet")
        x0, x1, y0, y1 = ev["region"]
        clouds = compare_fusion.board_clouds(cfg, p, compare_fusion.camera_side(p))
        inside = {m: c[(c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)] for m, c in clouds.items()}
        # evaluation.grid_height_range_mm overrides the percentiles where the pole or background fills too much of
        # the box for them to find the ice (the wave).
        low, high = ev.get("grid_height_range_mm") or np.percentile(
            np.concatenate([c[:, 2] for c in inside.values()]), HEIGHT_PERCENTILES)
        image = None
        for col, m in enumerate(methods):
            ax = axes[row, col]
            if m["name"] in inside:
                grid = compare_fusion.height_map(inside[m["name"]], (x0, x1), (y0, y1), ev["coverage_cell_mm"])
                image = ax.imshow(grid, extent=(x0, x1, y0, y1), cmap="turbo", vmin=low, vmax=high,
                                  interpolation="nearest")
                ax.text(0.02, 0.02, f"{100 * np.isfinite(grid).mean():.0f}% covered", transform=ax.transAxes,
                        fontsize=FONT["note"], color="k", bbox={"facecolor": "w", "alpha": 0.8, "lw": 0, "pad": 2})
                if placeholder:
                    ax.text(0.5, 0.5, "placeholder\n(cubes data)", transform=ax.transAxes, ha="center",
                            va="center", fontsize=FONT["row"], color="k",
                            bbox={"facecolor": "w", "alpha": 0.85, "lw": 0, "pad": 4})
            else:
                ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center",
                        fontsize=FONT["note"])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(m["label"], fontsize=FONT["title"])
            if col == 0:
                ax.set_ylabel(placeholder or TITLES.get(cfg["recording"]["name"], cfg["recording"]["title"]),
                              fontsize=FONT["row"])
        if image is not None:
            bar = fig.colorbar(image, ax=axes[row, :], shrink=0.85, pad=0.01)
            bar.set_label("height towards camera (mm)", fontsize=FONT["label"])
            bar.ax.tick_params(labelsize=FONT["ticks"])
    out = args.out or os.path.join(common.project_path(cfgs[0]["out_dir"]), "ice_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {common.rel(out)}")


if __name__ == "__main__":
    main()
