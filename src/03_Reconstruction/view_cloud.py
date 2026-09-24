"""Render the fused point cloud from dense_stereo.py, to see what the reconstruction produced.

Three views, all in the board frame from tag_poses.py (mm, z = height above the board):
  - top-down height map, with the reference tags outlined;
  - a thin slice through the tallest object, as a side profile, which is where the shape shows;
  - an oblique 3D view.

    src/venv/bin/python src/03_Reconstruction/view_cloud.py --session Sep24 --scan mjpg_pyr_lights_2 --method colmap
    ... --file cloud.ply             # the raw fused cloud instead of cloud_clean.ply (or sparse.ply)
    ... --show                       # rotate it (matplotlib, sampled)
    ... --3d                         # Open3D window, every point
    src/venv/bin/python src/03_Reconstruction/view_cloud.py --cloud any.ply --poses tag_poses.yaml

Writes <cloud>.png next to the PLY. For measuring or ICP against a CAD model, open the PLY in
CloudCompare instead: it keeps the "views" field as a scalar you can filter on.
"""
import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import rel  # noqa: E402
import project_paths as paths  # noqa: E402

# Sequential single-hue ramp (light -> dark blue): height is a magnitude, so one hue, no rainbow.
HEIGHT_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#2b2b2a"
MUTED = "#8a8a86"

SLICE_HALF_WIDTH = 3.0    # mm either side of the profile line
MAX_SCATTER = 150_000     # points drawn per panel; matplotlib slows badly beyond this


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Render the fused point cloud.")
    paths.add_cloud_arguments(ap, default_file="cloud_clean.ply")
    ap.add_argument("--min-views", type=int, default=0,
                    help="draw only points seen from at least this many frames (default: all in the PLY)")
    ap.add_argument("--no-crop", action="store_true",
                    help="show every point; by default only the board area is shown (the tags plus "
                         "--margin, z from --z-min to --z-max), as COLMAP clouds include the whole room")
    ap.add_argument("--margin", type=float, default=60.0, help="mm around the tags (default 60)")
    ap.add_argument("--z-min", type=float, default=-15.0, help="mm (default -15)")
    ap.add_argument("--z-max", type=float, default=150.0, help="mm (default 150)")
    ap.add_argument("--show", action="store_true", help="open an interactive matplotlib window as well")
    ap.add_argument("--3d", dest="open3d", action="store_true",
                    help="open the cloud in an Open3D window instead (all points; mouse to rotate, "
                         "scroll to zoom, +/- for point size), coloured by height, with the tags")
    args = ap.parse_args(argv)
    args.cloud, args.poses = paths.resolve_cloud(args, ap)
    return args


def read_ply(path):
    """(xyz, views) from any point-cloud PLY. Only dense_stereo.py writes a "views" field; for
    other clouds (COLMAP's, or a cleaned one from postprocess_cloud.py) every point counts as seen
    once, so --min-views has no effect on them."""
    import open3d as o3d
    xyz = np.asarray(o3d.io.read_point_cloud(path).points, np.float64)
    with open(path, "rb") as fh:
        header = []
        while (line := fh.readline().decode("ascii").strip()) != "end_header":
            header.append(line)
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "ushort": "<u2", "int": "<i4", "uint": "<u4"}
        fields = [(p.split()[2], types[p.split()[1]]) for p in header if p.startswith("property")]
        binary = any(h.startswith("format binary_little_endian") for h in header)
        if binary and "views" in dict(fields):
            return xyz, np.fromfile(fh, dtype=np.dtype(fields))["views"].astype(int)
    return xyz, np.ones(len(xyz), int)


def read_tags(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    tags = fs.getNode("tags")
    ids = [int(tags.getNode("ids").at(i).real()) for i in range(tags.getNode("ids").size())]
    corners = [tags.getNode("corners_mm").at(i).mat() for i in range(len(ids))]
    fs.release()
    return dict(zip(ids, corners))


def show_open3d(path, xyz, tags, norm, cmap):
    """Interactive Open3D window: the cloud coloured by height, the tags outlined, board axes."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(cmap(norm(xyz[:, 2]))[:, :3])
    corners = np.vstack(list(tags.values()))
    lines = [[4 * k + i, 4 * k + (i + 1) % 4] for k in range(len(tags)) for i in range(4)]
    outline = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(lines))
    outline.paint_uniform_color((0.1, 0.1, 0.1))
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)   # x red, y green, z blue
    o3d.visualization.draw_geometries([cloud, outline, axes], window_name=os.path.basename(path),
                                      width=1400, height=900)


def subsample(n, limit, rng):
    return np.arange(n) if n <= limit else rng.choice(n, limit, replace=False)


def main():
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    xyz, views = read_ply(args.cloud)
    keep = views >= args.min_views
    xyz, views = xyz[keep], views[keep]
    tags = read_tags(args.poses)
    print(f"{len(xyz)} points from {rel(args.cloud)}")
    if not args.no_crop:
        corners = np.vstack(list(tags.values()))
        low = np.r_[corners[:, :2].min(0) - args.margin, args.z_min]
        high = np.r_[corners[:, :2].max(0) + args.margin, args.z_max]
        inside = np.all((xyz >= low) & (xyz <= high), axis=1)
        xyz, views = xyz[inside], views[inside]
        print(f"{len(xyz)} in the board area (--no-crop to show all)")
        if not len(xyz):
            raise SystemExit("no points in the board area")

    cmap = LinearSegmentedColormap.from_list("height", HEIGHT_RAMP)
    z_low, z_high = np.percentile(xyz[:, 2], [1, 99.9])
    norm = matplotlib.colors.Normalize(max(z_low, -5), max(z_high, 10))
    if args.open3d:
        show_open3d(args.cloud, xyz, tags, norm, cmap)
        return
    rng = np.random.default_rng(0)

    # Profile line through the object: the centre of the points well above the board.
    raised = xyz[xyz[:, 2] > 0.3 * norm.vmax]
    centre = np.median(raised[:, :2], axis=0) if len(raised) else np.zeros(2)

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(15, 5.4), facecolor=SURFACE)
    where = paths.locate(args.cloud)
    title = (f"{where[0]} / {where[1]} / {where[2]} / {os.path.basename(args.cloud)}" if where
             else os.path.basename(args.cloud))
    fig.suptitle(f"{title}: {len(xyz):,} points, board frame (mm)", x=0.01, ha="left", fontsize=11)

    # Top-down: draw low points first so the object sits on top of the board.
    top = fig.add_subplot(1, 3, 1, facecolor=SURFACE)
    idx = subsample(len(xyz), MAX_SCATTER, rng)
    idx = idx[np.argsort(xyz[idx, 2])]
    top.scatter(xyz[idx, 0], xyz[idx, 1], c=xyz[idx, 2], cmap=cmap, norm=norm, s=0.3, linewidths=0)
    for tag_id, corners in tags.items():
        loop = np.vstack([corners, corners[:1]])
        top.plot(loop[:, 0], loop[:, 1], color=INK, lw=1)
        top.text(*corners.mean(0)[:2], str(tag_id), color=INK, ha="center", va="center", fontsize=8)
    top.axhline(centre[1], color=MUTED, lw=0.8, ls="--")
    top.set_aspect("equal")
    top.set_title("Top-down, coloured by height", loc="left")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")

    # Side profile: a thin slab along x through the object's centre.
    side = fig.add_subplot(1, 3, 2, facecolor=SURFACE)
    slab = np.abs(xyz[:, 1] - centre[1]) < SLICE_HALF_WIDTH
    near = slab & (np.abs(xyz[:, 0] - centre[0]) < 150)
    side.scatter(xyz[near, 0], xyz[near, 2], c=xyz[near, 2], cmap=cmap, norm=norm, s=2, linewidths=0)
    side.axhline(0, color=MUTED, lw=0.8)
    side.set_aspect("equal")
    side.set_title(f"Profile: slice y = {centre[1]:.0f} ± {SLICE_HALF_WIDTH:.0f} mm (dashed line)", loc="left")
    side.set_xlabel("x (mm)")
    side.set_ylabel("height z (mm)")

    # Oblique 3D view.
    view = fig.add_subplot(1, 3, 3, projection="3d", facecolor=SURFACE)
    idx = subsample(len(xyz), MAX_SCATTER // 2, rng)
    view.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2], c=xyz[idx, 2], cmap=cmap, norm=norm, s=0.3, linewidths=0)
    view.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), max(np.ptp(xyz[:, 2]), 1)))
    view.view_init(elev=30, azim=-60)
    view.set_title("Oblique", loc="left")
    view.set_xlabel("x")
    view.set_ylabel("y")
    view.set_zlabel("z")

    colourbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=[top, side, view],
                             shrink=0.7, pad=0.02)
    colourbar.set_label("height above board (mm)")

    out_path = os.path.splitext(args.cloud)[0] + ".png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {rel(out_path)}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
