"""Side-on (oblique) view of each fusion method's cloud over one scan's ice box: the same virtual camera for every
method, looking across the ice from evaluation.region's corner at --elevation degrees above the board, points coloured
by height towards the camera and shaded by their surface normal, so steps and ridges read as shape.

    src/venv/bin/python src/experiments/fusion/oblique_view.py --config src/experiments/fusion/config_pyramid_shade.json
    src/venv/bin/python src/experiments/fusion/oblique_view.py --grid      # every ice scan: fusion/oblique_grid.png
    src/venv/bin/python src/experiments/fusion/oblique_view.py --grid --with-image   # plus each scan's camera view
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
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "04_pose"))
import common  # noqa: E402
import compare_fusion  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

LIGHT = np.array([-0.4, -0.6, 0.7])
FONT = {"title": 16, "heading": 20, "label": 14, "ticks": 12}


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


def vertical_scale(ax, image, pixel_mm, elevation):
    """A ruler up the panel's left edge, the full height of the image, in mm of height (0 at the bottom): the view is
    orthographic, so a vertical step of dz shows as dz * cos(elevation) anywhere in the panel, and the ruler reads a
    vertical edge's true height."""
    h = image.shape[0]
    mm_per_px = pixel_mm / np.cos(np.radians(elevation))
    ticks = matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, h * mm_per_px)
    ticks = ticks[ticks <= h * mm_per_px]
    ax.set_yticks(h - 0.5 - ticks / mm_per_px, [f"{t:g}" for t in ticks], fontsize=FONT["ticks"])
    ax.spines["left"].set_visible(True)
    ax.set_ylabel("height (mm)", fontsize=FONT["label"], labelpad=2)


def place_row_labels(fig, axes, labels, fontsize, strip=0.04):
    """Row names on one vertical line, each centred on its row: the layout is kept out of a strip of this width at the
    figure's left, and the names are centred in it."""
    fig.get_layout_engine().set(rect=(strip, 0, 1 - strip, 1))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, label in zip(axes[:, 0], labels):
        box = ax.get_position()
        fig.text(strip / 2, (box.y0 + box.y1) / 2, label, rotation=90, ha="center", va="center",
                 fontsize=fontsize, fontweight="bold")


def source_image(cfg, samples=4000, margin=0.15, aspect=4 / 3):
    """The camera frame that best sees the ice, cropped to it: the TSDF-FoundationStereo points in the ice box are
    projected into the left image of every frame used (its COLMAP pose, the board fit, the calibration with its
    distortion); the frame where they fall nearest the image centre, all inside the image, is taken, cropped to their
    bounding box plus margin."""
    import cv2
    import json
    import pose_files
    from stereo_rig import load_rig
    p = recording_paths(cfg)
    x0, x1, y0, y1 = cfg["evaluation"]["region"]
    cloud = np.asarray(o3d.io.read_point_cloud(os.path.join(p["method"]("tsdf_fs"), "cloud_board.ply")).points)
    pts = cloud[(cloud[:, 0] >= x0) & (cloud[:, 0] < x1) & (cloud[:, 1] >= y0) & (cloud[:, 1] < y1)]
    pts = pts[np.random.default_rng(0).choice(len(pts), min(samples, len(pts)), replace=False)]
    with open(os.path.join(p["out"], "alignment.json"), encoding="utf-8") as fh:
        T_world_board = np.linalg.inv(np.array(json.load(fh)["T_board_world"]))
    rig = load_rig(common.project_path(cfg["recording"]["calibration"]))
    K, D = rig.K["left"], rig.D["left"]
    width, height = rig.image_size
    _, poses, _ = pose_files.read_trajectory(p["poses_used"])
    world = pts @ T_world_board[:3, :3].T + T_world_board[:3, 3]
    best = None
    for name, T in poses.items():
        cam = world @ T[:3, :3].T + T[:3, 3]
        if (cam[:, 2] <= 0).any():
            continue
        uv = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        score = (inside.mean(), -np.linalg.norm(np.median(uv, 0) - [width / 2, height / 2]))
        if best is None or score > best[0]:
            best = (score, name, uv[inside])
    _, name, uv = best
    image = cv2.cvtColor(cv2.imread(os.path.join(common.project_path(cfg["recording"]["frames"]), "left", name)),
                         cv2.COLOR_BGR2RGB)
    (u0, v0), (u1, v1) = uv.min(0), uv.max(0)
    # Same aspect for every scan so the views fill their tiles alike.
    w, h = (u1 - u0) * (1 + 2 * margin), (v1 - v0) * (1 + 2 * margin)
    w, h = max(w, h * aspect), max(h, w / aspect)
    if w > width:
        w, h = width, width / aspect
    if h > height:
        w, h = height * aspect, height
    cu = min(max((u0 + u1) / 2, w / 2), width - w / 2)
    cv = min(max((v0 + v1) / 2, h / 2), height - h / 2)
    return image[int(cv - h / 2):int(cv + h / 2), int(cu - w / 2):int(cu + w / 2)], name


def grid(args):
    """--grid: a row per scan (--configs, then the PLACEHOLDERS rows of ice_grid.py), a column per method."""
    import ice_grid
    rows = [(None, common.load_config(c)) for c in args.configs]
    if not args.no_placeholders:
        rows += [(title, common.load_config(os.path.join(HERE, c))) for title, c in ice_grid.PLACEHOLDERS]
    methods = rows[0][1]["methods"]
    lead = 1 if args.with_image else 0   # --with-image: a column of the camera's view of each scan first
    fig, axes = plt.subplots(len(rows), lead + len(methods), figsize=(6.0 * (lead + len(methods)) + 1.4, 4.6 * len(rows)),
                             squeeze=False, constrained_layout=True)
    cmap = colormaps["turbo"]
    for r, (placeholder, cfg) in enumerate(rows):
        images, (low, high) = scan_images(cfg, args)
        if lead:
            ax = axes[r, 0]
            photo, frame = source_image(cfg)
            ax.imshow(photo, aspect="auto")   # every crop is 4:3, so "auto" fills the tile without distortion
            ax.text(0.02, 0.02, f"frame {frame}", transform=ax.transAxes, fontsize=FONT["ticks"],
                    bbox={"facecolor": "w", "alpha": 0.8, "lw": 0, "pad": 2})
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("Camera view (left)", fontsize=FONT["heading"], fontweight="bold")
        for c, m in enumerate(methods, start=lead):
            ax = axes[r, c]
            m_images = images
            if m["name"] in m_images:
                ax.imshow(m_images[m["name"]], interpolation="lanczos")
                if placeholder:
                    ax.text(0.5, 0.5, "placeholder\n(cubes data)", transform=ax.transAxes, ha="center",
                            va="center", fontsize=FONT["title"], bbox={"facecolor": "w", "alpha": 0.85, "lw": 0, "pad": 4})
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == lead and m["name"] in m_images:
                vertical_scale(ax, m_images[m["name"]], args.pixel_mm, args.elevation)
            if r == 0:
                ax.set_title(m["label"], fontsize=FONT["heading"], fontweight="bold")

        bar = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(low, high), cmap),
                           ax=axes[r, :], shrink=0.8, pad=0.01)
        bar.set_label("height towards\ncamera (mm)", fontsize=FONT["label"])
        bar.ax.tick_params(labelsize=FONT["ticks"])
    place_row_labels(fig, axes, [placeholder or ice_grid.TITLES.get(cfg["recording"]["name"], cfg["recording"]["title"])
                                 for placeholder, cfg in rows], FONT["heading"])
    out = os.path.join(common.project_path(rows[0][1]["out_dir"]),
                       "oblique_grid_with_image.png" if lead else "oblique_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {common.rel(out)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="one scan: writes its oblique.png")
    ap.add_argument("--grid", action="store_true", help="all scans in one figure: results/.../fusion/oblique_grid.png")
    ap.add_argument("--configs", nargs="+", help="scans for --grid (default: ice_grid.py's CONFIGS)")
    ap.add_argument("--no-placeholders", action="store_true", help="--grid without ice_grid.py's PLACEHOLDERS rows")
    ap.add_argument("--with-image", action="store_true",
                    help="--grid with a first column of each scan's camera view: oblique_grid_with_image.png")
    ap.add_argument("--azimuth", type=float, default=225.0, help="deg; direction the camera looks from (default 225)")
    ap.add_argument("--elevation", type=float, default=30.0, help="deg above the board (default 30)")
    ap.add_argument("--pixel-mm", type=float, default=0.6)
    ap.add_argument("--splat", type=int, default=1, help="pixels each point covers either side (default 1)")
    args =ap.parse_args(argv)
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
