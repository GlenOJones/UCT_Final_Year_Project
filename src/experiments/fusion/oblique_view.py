"""Side-on (oblique) view of each fusion method's cloud over one scan's ice box: the same virtual camera for every
method, looking across the ice from evaluation.region's corner at --elevation degrees above the board, points
coloured by height towards the camera and shaded by their surface normal, so steps and ridges read as shape.

Reads what run_fusion.py saved (<work>/<method>/cloud_board.ply), cut to the ice box. Points are drawn with an
orthographic projection and a depth buffer (nearest point per pixel wins), --pixel-mm on a side, each splatted
over a few pixels so the 1 mm clouds have no gaps. COLMAP has no mesh, so points are used for every method alike.

    src/venv/bin/python src/experiments/fusion/oblique_view.py --config src/experiments/fusion/config_pyramid_shade.json
    src/venv/bin/python src/experiments/fusion/oblique_view.py --grid      # every ice scan: fusion/oblique_grid.png

Writes results/<out_dir>/<recording>/oblique.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from matplotlib import colormaps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import compare_fusion  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

LIGHT = np.array([-0.4, -0.6, 0.7])
FONT = {"title": 16, "label": 14, "ticks": 12}


def shading(xyz):
    """0.35..1 per point from its normal (estimated over 4 mm) against a fixed light; normals face the camera (+z)."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=30))
    normals = np.asarray(cloud.normals)
    normals[normals[:, 2] < 0] *= -1
    return 0.35 + 0.65 * np.clip(normals @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)


def render(xyz, colours, azimuth, elevation, pixel_mm, splat, extent=None):
    """Orthographic depth-buffered render: (image HxWx3 with white background, extent used)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    forward = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # from the camera towards the scene
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    u, v, depth = xyz @ right, xyz @ up, xyz @ forward
    if extent is None:
        extent = (u.min(), u.max(), v.min(), v.max())
    u0, u1, v0, v1 = extent
    w, h = int(np.ceil((u1 - u0) / pixel_mm)) + 1, int(np.ceil((v1 - v0) / pixel_mm)) + 1
    col = np.floor((u - u0) / pixel_mm).astype(int)
    row = np.floor((v1 - v) / pixel_mm).astype(int)
    image = np.ones((h, w, 3))
    zbuf = np.full((h, w), np.inf)
    order = np.argsort(-depth)            # far first, so nearer points overwrite
    for dr in range(-splat, splat + 1):
        for dc in range(-splat, splat + 1):
            r, c = row[order] + dr, col[order] + dc
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            r, c, d, idx = r[ok], c[ok], depth[order][ok], order[ok]
            nearer = d < zbuf[r, c]
            zbuf[r[nearer], c[nearer]] = d[nearer]
            image[r[nearer], c[nearer]] = colours[idx[nearer]]
    return image, extent


def scan_images(cfg, args):
    """({method name: rendered image}, (low, high) height range) for one scan's ice box, every method from one view."""
    p = recording_paths(cfg)
    ev = cfg["evaluation"]
    x0, x1, y0, y1 = ev["region"]
    clouds = compare_fusion.board_clouds(cfg, p, compare_fusion.camera_side(p))
    inside = {m: c[(c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)] for m, c in clouds.items()}
    low, high = ev.get("grid_height_range_mm") or np.percentile(np.concatenate([c[:, 2] for c in inside.values()]),
                                                                 (10, 99.5))
    # Drop what lies far behind the ice (pole, background), which would hide nothing but clutter the view.
    inside = {m: c[c[:, 2] >= low - 0.2 * (high - low)] for m, c in inside.items()}
    cmap = colormaps["turbo"]
    extent = None
    images = {}
    for m in cfg["methods"]:
        xyz = inside.get(m["name"])
        if xyz is None or not len(xyz):
            continue
        colours = cmap(np.clip((xyz[:, 2] - low) / (high - low), 0, 1))[:, :3] * shading(xyz)[:, None]
        if extent is None:   # one extent for all, from the union, so every panel shows the same view
            union = np.concatenate([c for c in inside.values()])
            _, extent = render(union[::50], np.zeros((len(union[::50]), 3)), args.azimuth, args.elevation,
                               args.pixel_mm, 0)
        images[m["name"]], _ = render(xyz, colours, args.azimuth, args.elevation, args.pixel_mm, args.splat, extent)
    return images, (low, high)


def grid(args):
    """--grid: a row per scan (--configs, then the PLACEHOLDERS rows of ice_grid.py), a column per method."""
    import ice_grid
    rows = [(None, common.load_config(c)) for c in args.configs]
    if not args.no_placeholders:
        rows += [(title, common.load_config(os.path.join(HERE, c))) for title, c in ice_grid.PLACEHOLDERS]
    methods = rows[0][1]["methods"]
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(6.0 * len(methods) + 1.4, 4.6 * len(rows)),
                             squeeze=False, constrained_layout=True)
    cmap = colormaps["turbo"]
    for r, (placeholder, cfg) in enumerate(rows):
        images, (low, high) = scan_images(cfg, args)
        for c, m in enumerate(methods):
            ax = axes[r, c]
            if m["name"] in images:
                ax.imshow(images[m["name"]], interpolation="lanczos")
                if placeholder:
                    ax.text(0.5, 0.5, "placeholder\n(cubes data)", transform=ax.transAxes, ha="center",
                            va="center", fontsize=FONT["title"], bbox={"facecolor": "w", "alpha": 0.85, "lw": 0, "pad": 4})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(m["label"], fontsize=FONT["title"])
            if c == 0:
                ax.set_ylabel(placeholder or ice_grid.TITLES.get(cfg["recording"]["name"], cfg["recording"]["title"]),
                              fontsize=FONT["title"])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(low, high), cmap),
                           ax=axes[r, :], shrink=0.8, pad=0.01)
        bar.set_label("height towards\ncamera (mm)", fontsize=FONT["label"])
        bar.ax.tick_params(labelsize=FONT["ticks"])
    out = os.path.join(common.project_path(rows[0][1]["out_dir"]), "oblique_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {common.rel(out)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="one scan: writes its oblique.png")
    ap.add_argument("--grid", action="store_true", help="all scans in one figure: results/.../fusion/oblique_grid.png")
    ap.add_argument("--configs", nargs="+", help="scans for --grid (default: ice_grid.py's CONFIGS)")
    ap.add_argument("--no-placeholders", action="store_true", help="--grid without ice_grid.py's PLACEHOLDERS rows")
    ap.add_argument("--azimuth", type=float, default=225.0, help="deg; direction the camera looks from (default 225)")
    ap.add_argument("--elevation", type=float, default=30.0, help="deg above the board (default 30)")
    ap.add_argument("--pixel-mm", type=float, default=0.6)
    ap.add_argument("--splat", type=int, default=1, help="pixels each point covers either side (default 1)")
    args = ap.parse_args(argv)
    if args.grid:
        import ice_grid
        args.configs = args.configs or [os.path.join(HERE, c) for c in ice_grid.CONFIGS]
        grid(args)
        return
    if not args.config:
        ap.error("give --config (one scan) or --grid")
    cfg = common.load_config(args.config)
    p = recording_paths(cfg)
    images, (low, high) = scan_images(cfg, args)
    cmap = colormaps["turbo"]
    methods = cfg["methods"]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.2 * len(methods), 5.6), squeeze=False, constrained_layout=True)
    for ax, m in zip(axes[0], methods):
        if m["name"] in images:
            ax.imshow(images[m["name"]], interpolation="lanczos")
        else:
            ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(m["label"], fontsize=FONT["title"])
        ax.axis("off")
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(low, high), cmap), ax=axes[0],
                       shrink=0.8, pad=0.01)
    bar.set_label("height towards camera (mm)", fontsize=FONT["label"])
    bar.ax.tick_params(labelsize=FONT["ticks"])
    out = os.path.join(p["out"], "oblique.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {common.rel(out)}")


if __name__ == "__main__":
    main()
