"""Compare reconstructed point clouds with the object's CAD model (ground-truth STL).

Reads the clouds 03_Reconstruction saved (board frame, mm, z = height above the board) and measures
how far every point is from the CAD surface, after placing the CAD model where the object is.

    src/venv/bin/python src/04_Comparison/compare_to_cad.py results/3dRecon/Sep24/mjpg_pyr_lights_2_SGBM_reconstruction/mjpg_pyr_lights_2_cloud_clean.ply
    src/venv/bin/python src/04_Comparison/compare_to_cad.py cloudA.ply cloudB.ply --reference data/GroundTruth/roughness_blocks_STL/PYRAMID_1.stl

Steps, for each cloud:
  1. Reference: the STL, with duplicate vertices merged. Faces pointing down are dropped: they sit on
     the board, and no camera can see them.
  2. Object points: points above --min-height (lower ones are board), in the largest connected
     cluster, which removes stray points floating over the board.
  3. Alignment: robust point-to-plane ICP against the exact CAD triangles (not a sampled copy),
     started from a set of rotations about z, keeping the best. Two fits are made:
       on-board (default 3 DOF: x, y, yaw) - the object sits flat on the board, and the board frame
           already fixes height and tilt. This is the absolute accuracy of the reconstruction: a
           height or tilt error shows up in the distances instead of being fitted away.
       shape (6 DOF) - the best rigid fit, for the accuracy of the shape alone.
  4. Metrics per fit: signed distance of every object point to the surface (positive = outside),
     its bias, spread and percentiles; per face, how much of it the scan covers and the angle of a
     plane fitted through its points against the design.

Outputs in results/comparison/<cloud name>/:
  metrics.json                 every number, for both fits
  <name>_distances.ply         object points in the board frame with a "scalar_signed_distance"
                               property (and colours); CloudCompare loads it as a scalar field
  <name>_reference_aligned.ply the CAD mesh moved into the board frame by the on-board fit
  <name>_comparison.png        top-down error map, histogram and profiles through the apex
and results/comparison/summary.csv, one row per cloud and fit, appended to on every run.
"""
import argparse
import csv
import datetime
import json
import os
import sys

import cv2
import numpy as np
import open3d as o3d

# Paths are relative to the project root, resolved from this file
# (src/04_Comparison/compare_to_cad.py -> parents: 04_Comparison, src, project root).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REFERENCE = os.path.join(PROJECT_ROOT, "data/GroundTruth/roughness_blocks_STL/PYRAMID_1.stl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/comparison")

# Degrees of freedom of each fit: indices into (rx, ry, rz, tx, ty, tz).
FITS = {"on_board": [2, 3, 4], "shape": [0, 1, 2, 3, 4, 5]}

ICP_ITERATIONS = 60
ICP_START_DISTANCE = 20.0     # mm; residuals above this are ignored in the first iteration...
ICP_MAD_FACTOR = 3.0          # ...then above 3 robust standard deviations of the current residuals
YAW_START_STEP = 15           # deg between the starting rotations tried about z

# Below this coverage a fit is flagged unreliable. The quick COLMAP cloud covered 7% of the pyramid,
# mostly its ridges, and its 6-DOF fit slid the pyramid 43 mm up them with a small residual: a low
# RMS on so little surface says nothing about the reconstruction.
MIN_RELIABLE_COVERAGE = 0.15

# Diverging map for signed distance: blue = inside the CAD surface, grey = on it, red = outside.
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#e9e8e4", "#f4b3a6", "#d9534a", "#8e2a1f"]
SURFACE = "#fcfcfb"
INK = "#2b2b2a"
MUTED = "#8a8a86"


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare reconstructed clouds with a CAD mesh.")
    ap.add_argument("clouds", nargs="+", help="board-frame PLY clouds from 03_Reconstruction")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE, help="ground-truth mesh, mm (STL/PLY/OBJ)")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--min-height", type=float, default=3.0,
                    help="mm; points below this are board, not object (default 3)")
    ap.add_argument("--margin", type=float, default=15.0,
                    help="mm; object points must lie within the placed CAD footprint plus this (default 15)")
    ap.add_argument("--eval-margin", type=float, default=2.0,
                    help="mm; points are scored only within the CAD footprint plus this. Further out they "
                         "are board raised by noise, not object (default 2)")
    ap.add_argument("--coverage-radius", type=float, default=2.0,
                    help="mm; a CAD surface patch counts as covered if a point is this close (default 2)")
    ap.add_argument("--error-range", type=float, default=5.0,
                    help="mm; colour scale of the error map runs from minus to plus this (default 5)")
    return ap.parse_args(argv)


# ====== REFERENCE ======
class Reference:
    """The CAD mesh, its visible faces grouped into facets, and a ray-casting scene for exact
    closest-point queries against the visible triangles."""

    def __init__(self, path):
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise SystemExit(f"could not read a mesh from {rel(path)}")
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.compute_triangle_normals()
        self.path = path
        self.full = mesh
        normals = np.asarray(mesh.triangle_normals)
        visible = normals[:, 2] > -0.9           # downward faces rest on the board
        self.mesh = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[visible]))
        self.mesh.compute_triangle_normals()
        self.normals = np.asarray(self.mesh.triangle_normals)
        self.facet = facet_labels(self.normals)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self.mesh))
        vertices = np.asarray(mesh.vertices)
        self.footprint = (vertices[:, :2].min(0), vertices[:, :2].max(0))
        self.centre_xy = vertices[:, :2].mean(0)
        self.height = float(np.ptp(vertices[:, 2]))

    def closest(self, points):
        """Closest point on the visible CAD surface, its triangle's normal and the triangle id."""
        result = self.scene.compute_closest_points(o3d.core.Tensor(points.astype(np.float32)))
        return (result["points"].numpy().astype(np.float64),
                result["primitive_normals"].numpy().astype(np.float64),
                result["primitive_ids"].numpy())


def facet_labels(normals, tolerance_deg=5.0):
    """Group triangles whose normals agree within tolerance: one label per flat face of the CAD model
    (a pyramid's face may be split into several triangles)."""
    labels = -np.ones(len(normals), int)
    directions = []
    for i, n in enumerate(normals):
        for k, d in enumerate(directions):
            if np.degrees(np.arccos(np.clip(n @ d, -1, 1))) < tolerance_deg:
                labels[i] = k
                break
        else:
            directions.append(n)
            labels[i] = len(directions) - 1
    return labels


# ====== OBJECT POINTS ======
def object_points(xyz, min_height, object_height):
    """Points above the board in the connected cluster that reaches highest: the one with the most
    points above half the object's height. Choosing the largest cluster instead picked a streak of
    board noise next to a tag in the sparse COLMAP cloud, not the pyramid."""
    raised = xyz[xyz[:, 2] > min_height]
    if len(raised) < 50:
        raise SystemExit(f"only {len(raised)} points above {min_height} mm: nothing to compare")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(raised))
    labels = np.asarray(cloud.cluster_dbscan(eps=4.0, min_points=10))
    if labels.max() < 0:
        return raised
    tall = raised[:, 2] > object_height / 2
    counts = np.bincount(labels[(labels >= 0) & tall], minlength=labels.max() + 1)
    if counts.max() == 0:
        counts = np.bincount(labels[labels >= 0])
    return raised[labels == counts.argmax()]


# ====== ALIGNMENT ======
def small_motion(x):
    """4x4 transform from a small motion (rx, ry, rz, tx, ty, tz)."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], np.float64))[0]
    T[:3, 3] = x[3:]
    return T


def transform(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def icp(reference, points, T_ref, dof, min_height, margin):
    """Robust point-to-plane ICP moving the REFERENCE onto the points (the scan stays in the board
    frame). Linearised on the inverse motion: each point p, taken into the reference frame, must lie
    on the plane of its closest CAD point, so the residual is n . (p' - q) and its derivative with
    respect to a small motion (w, t) of p' is (p' x n, n). Points further out than 3 robust standard
    deviations are ignored, which removes the stray points and the unseen parts of the scan.
    Returns (T_board_ref, residual RMS of the inliers, inlier count)."""
    T_ref_board = np.linalg.inv(T_ref)
    threshold = ICP_START_DISTANCE
    for _ in range(ICP_ITERATIONS):
        local = transform(T_ref_board, points)
        q, n, _ = reference.closest(local)
        r = np.sum((local - q) * n, axis=1)
        distance = np.linalg.norm(local - q, axis=1)
        inside = inside_footprint(reference, local, margin) & (local[:, 2] > min_height - 1.0)
        use = inside & (distance < threshold)
        if use.sum() < 20:
            break
        J = np.hstack([np.cross(local[use], n[use]), n[use]])[:, dof]
        step = np.zeros(6)
        step[dof] = np.linalg.lstsq(J, -r[use], rcond=None)[0]
        T_ref_board = small_motion(step) @ T_ref_board
        mad = 1.4826 * np.median(np.abs(r[use]))
        threshold = max(ICP_MAD_FACTOR * mad, 0.5)
        if np.linalg.norm(step[:3]) < 1e-6 and np.linalg.norm(step[3:]) < 1e-4:
            break
    local = transform(T_ref_board, points)
    q, n, _ = reference.closest(local)
    distance = np.linalg.norm(local - q, axis=1)
    use = inside_footprint(reference, local, margin) & (distance < threshold)
    rms = float(np.sqrt(np.mean(distance[use] ** 2))) if use.any() else np.inf
    return np.linalg.inv(T_ref_board), rms, int(use.sum())


def inside_footprint(reference, local, margin):
    low, high = reference.footprint
    return np.all((local[:, :2] >= low - margin) & (local[:, :2] <= high + margin), axis=1)


def align(reference, points, dof, min_height, margin, start=None):
    """Best ICP result over starting yaws about the object's centre (or from `start`)."""
    if start is not None:
        return icp(reference, points, start, dof, min_height, margin)
    centre = np.median(points[:, :2], axis=0)
    best = None
    for yaw in range(0, 360, YAW_START_STEP):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array([0, 0, np.radians(yaw)]))[0]
        T[:2, 3] = centre - T[:2, :2] @ reference.centre_xy
        result = icp(reference, points, T, dof, min_height, margin)
        # Prefer the fit explaining the most points, then the lowest residual.
        if best is None or (result[2], -result[1]) > (best[2], -best[1]):
            best = result
    return best


# ====== METRICS ======
def evaluate(reference, points, T_board_ref, args):
    """Signed distances, their statistics, and per-facet coverage and slope."""
    T_ref_board = np.linalg.inv(T_board_ref)
    local = transform(T_ref_board, points)
    q, n, tri = reference.closest(local)
    signed = np.sign(np.sum((local - q) * n, axis=1)) * np.linalg.norm(local - q, axis=1)
    # Scored only over the footprint. The first run (margin 15 mm) scored the board around the
    # pyramid too: board read 3-5 mm high there, above --min-height, and as "distance to the
    # pyramid" it put a +2.6 mm bias and a tail past +15 mm into the lights scan's numbers.
    keep = inside_footprint(reference, local, args.eval_margin)
    halo = int(np.sum(inside_footprint(reference, local, args.margin) & ~keep))
    local, signed, tri = local[keep], signed[keep], tri[keep]
    a = np.abs(signed)
    stats = {
        "points": int(len(signed)),
        "points_outside_footprint": halo,
        "mean_signed_mm": float(np.mean(signed)),
        "median_signed_mm": float(np.median(signed)),
        "std_mm": float(np.std(signed)),
        "rms_mm": float(np.sqrt(np.mean(signed ** 2))),
        "median_abs_mm": float(np.median(a)),
        "p90_abs_mm": float(np.percentile(a, 90)),
        "p95_abs_mm": float(np.percentile(a, 95)),
        "max_abs_mm": float(a.max()),
        "within_1mm": float(np.mean(a < 1)),
        "within_2mm": float(np.mean(a < 2)),
        "within_5mm": float(np.mean(a < 5)),
    }

    # Coverage: sample the visible CAD surface ~1 per mm^2 and check for a scan point nearby.
    samples = reference.mesh.sample_points_uniformly(max(int(reference.mesh.get_surface_area()), 1000))
    sample_xyz = np.asarray(samples.points)
    _, _, sample_tri = reference.closest(sample_xyz)
    tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local)))
    covered = np.array([tree.search_radius_vector_3d(p, args.coverage_radius)[0] > 0 for p in sample_xyz])

    facets = []
    point_facet = reference.facet[tri]
    for k in range(reference.facet.max() + 1):
        design = reference.normals[reference.facet == k][0]
        mine = point_facet == k
        entry = {"facet": k, "design_normal": design.round(4).tolist(),
                 "points": int(mine.sum()),
                 "coverage": float(np.mean(covered[reference.facet[sample_tri] == k]))}
        if mine.sum() >= 30:
            entry["mean_signed_mm"] = float(np.mean(signed[mine]))
            entry["rms_mm"] = float(np.sqrt(np.mean(signed[mine] ** 2)))
            # Plane through this facet's points; its tilt from the design face is a slope error
            # that does not depend on how much of the face was covered.
            p = local[mine]
            fitted = np.linalg.svd(p - p.mean(0))[2][2]
            fitted = fitted if fitted @ design > 0 else -fitted
            entry["plane_angle_error_deg"] = float(np.degrees(np.arccos(np.clip(fitted @ design, -1, 1))))
        facets.append(entry)
    stats["coverage"] = float(np.mean(covered))
    stats["facets"] = facets
    return stats, local, signed, keep


def describe_transform(T):
    angles = np.degrees(cv2.Rodrigues(T[:3, :3])[0].ravel())
    return {"translation_mm": T[:3, 3].round(3).tolist(), "rotation_deg_xyz": angles.round(3).tolist(),
            "matrix": T.round(6).tolist()}


# ====== OUTPUTS ======
def diverging_colours(values, limit):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("signed", DIVERGING)
    return cmap(np.clip((values + limit) / (2 * limit), 0, 1))[:, :3]


def write_distance_ply(path, xyz, signed, limit):
    """Binary PLY: x y z, colour from the error map, and signed_distance as a float scalar field."""
    colours = (diverging_colours(signed, limit) * 255).round().astype(np.uint8)
    record = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                       ("blue", "u1"), ("scalar_signed_distance", "<f4")])
    data = np.empty(len(xyz), record)
    data["x"], data["y"], data["z"] = xyz.T
    data["red"], data["green"], data["blue"] = colours.T
    data["scalar_signed_distance"] = signed
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment board frame, mm; signed_distance to the CAD surface, positive = outside\n"
              f"element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float scalar_signed_distance\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def plot(path, name, reference, xyz_board, signed, T_board_ref, stats, limit):
    """Top-down error map with the placed CAD outline, histogram, and profiles through the apex."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("signed", DIVERGING)
    norm = Normalize(-limit, limit)
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(15, 5.2), facecolor=SURFACE)
    fig.suptitle(f"{name} vs {os.path.basename(reference.path)}: RMS {stats['rms_mm']:.2f} mm, "
                 f"bias {stats['mean_signed_mm']:+.2f} mm, coverage {100 * stats['coverage']:.0f}% "
                 f"(on-board fit)", x=0.01, ha="left", fontsize=11)

    ref_vertices = transform(T_board_ref, np.asarray(reference.full.vertices))
    triangles = np.asarray(reference.full.triangles)

    top = fig.add_subplot(1, 3, 1, facecolor=SURFACE)
    order = np.argsort(np.abs(signed))
    top.scatter(xyz_board[order, 0], xyz_board[order, 1], c=signed[order], cmap=cmap, norm=norm, s=1, linewidths=0)
    for tri in triangles:
        loop = ref_vertices[np.r_[tri, tri[0]]]
        top.plot(loop[:, 0], loop[:, 1], color=INK, lw=0.6, alpha=0.6)
    top.set_aspect("equal")
    top.set_title("Signed distance to CAD, top-down", loc="left")
    top.set_xlabel("x (mm)")
    top.set_ylabel("y (mm)")

    hist = fig.add_subplot(1, 3, 2, facecolor=SURFACE)
    bins = np.linspace(-3 * limit, 3 * limit, 121)
    hist.hist(np.clip(signed, bins[0], bins[-1]), bins=bins, color="#3987e5", edgecolor=SURFACE, linewidth=0.3)
    hist.axvline(0, color=INK, lw=0.8)
    hist.axvline(stats["mean_signed_mm"], color="#d9534a", lw=1, ls="--")
    hist.set_title(f"Distribution: median |d| {stats['median_abs_mm']:.2f} mm, "
                   f"p95 {stats['p95_abs_mm']:.2f} mm", loc="left")
    beyond = int(np.sum(np.abs(signed) > bins[-1]))
    if beyond:
        hist.text(0.99, 0.97, f"{beyond} points beyond +/-{bins[-1]:g} mm\ndrawn in the end bins",
                  transform=hist.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    hist.set_xlabel("signed distance (mm), positive = outside the CAD surface")
    hist.set_ylabel("points")

    # Profiles: slabs through the apex along the CAD model's own x and y axes.
    prof = fig.add_subplot(1, 3, 3, facecolor=SURFACE)
    T_ref_board = np.linalg.inv(T_board_ref)
    local = transform(T_ref_board, xyz_board)
    apex = np.asarray(reference.full.vertices)[np.argmax(np.asarray(reference.full.vertices)[:, 2])]
    low, high = reference.footprint
    for axis, other, colour, label in ((0, 1, "#184f95", "along CAD x"), (1, 0, "#8e2a1f", "along CAD y")):
        slab = np.abs(local[:, other] - apex[other]) < 3.0
        prof.scatter(local[slab, axis] - apex[axis], local[slab, 2], s=2, color=colour, alpha=0.6,
                     linewidths=0, label=f"scan, {label}")
    half = (high[0] - low[0]) / 2
    prof.plot([-half, 0, half], [0, apex[2], 0], color=INK, lw=1.2, label="CAD profile")
    prof.set_aspect("equal")
    prof.set_title("Profiles through the apex (slabs of +/-3 mm)", loc="left")
    prof.set_xlabel("distance from apex (mm)")
    prof.set_ylabel("height (mm)")
    prof.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    colourbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=top, orientation="horizontal",
                             shrink=0.8, pad=0.14, aspect=30)
    colourbar.set_label(f"signed distance (mm), clipped at +/-{limit:g}")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def append_summary(path, name, cloud_path, reference, fits):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["created", "cloud", "reference", "fit", "points", "coverage", "mean_signed_mm",
                             "rms_mm", "median_abs_mm", "p95_abs_mm", "within_2mm", "reliable", "tx_mm", "ty_mm", "tz_mm",
                             "yaw_deg", "facet_angle_errors_deg"])
        for fit, (T, stats) in fits.items():
            angles = np.degrees(cv2.Rodrigues(T[:3, :3])[0].ravel())
            facet_angles = ";".join(f"{f['plane_angle_error_deg']:.2f}" if "plane_angle_error_deg" in f else "-"
                                    for f in stats["facets"])
            writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"), rel(cloud_path),
                             os.path.basename(reference.path), fit, stats["points"], f"{stats['coverage']:.3f}",
                             f"{stats['mean_signed_mm']:.3f}", f"{stats['rms_mm']:.3f}",
                             f"{stats['median_abs_mm']:.3f}", f"{stats['p95_abs_mm']:.3f}",
                             f"{stats['within_2mm']:.3f}", stats["reliable"], *(f"{v:.2f}" for v in T[:3, 3]),
                             f"{angles[2]:.2f}", facet_angles])


def report(name, fits):
    print(f"\n{name}")
    print(f"  scored inside the CAD footprint; {fits['on_board'][1]['points_outside_footprint']} raised points "
          f"just outside it (board noise) not scored")
    print(f"  {'fit':9s} {'points':>7s} {'cover':>6s} {'bias':>7s} {'RMS':>6s} {'med|d|':>7s} {'p95':>6s} "
          f"{'<2mm':>5s}   placement (x, y, z mm; yaw deg)")
    for fit, (T, s) in fits.items():
        yaw = np.degrees(cv2.Rodrigues(T[:3, :3])[0].ravel())[2]
        print(f"  {fit:9s} {s['points']:7d} {100 * s['coverage']:5.0f}% {s['mean_signed_mm']:+7.2f} "
              f"{s['rms_mm']:6.2f} {s['median_abs_mm']:7.2f} {s['p95_abs_mm']:6.2f} {100 * s['within_2mm']:4.0f}%   "
              f"({T[0, 3]:.1f}, {T[1, 3]:.1f}, {T[2, 3]:.2f}; {yaw:.1f})"
              + ("" if s["reliable"] else f"   UNRELIABLE: covers under {100 * MIN_RELIABLE_COVERAGE:.0f}% of the surface"))
        faces = ", ".join(f"{100 * f['coverage']:.0f}%" + (f"/{f['plane_angle_error_deg']:.1f}deg"
                          if "plane_angle_error_deg" in f else "") for f in s["facets"])
        print(f"  {'':9s} per face (coverage/slope error): {faces}")


def main():
    args = parse_args()
    reference = Reference(args.reference)
    print(f"reference {rel(args.reference)}: {len(reference.mesh.triangles)} visible triangles in "
          f"{reference.facet.max() + 1} faces, footprint {np.ptp(np.array(reference.footprint), 0).round(1).tolist()} mm, "
          f"height {reference.height:.1f} mm")
    os.makedirs(args.out_dir, exist_ok=True)

    for cloud_path in args.clouds:
        xyz = np.asarray(o3d.io.read_point_cloud(cloud_path).points, np.float64)
        name = os.path.splitext(os.path.basename(cloud_path))[0]
        points = object_points(xyz, args.min_height, reference.height)
        print(f"\n{rel(cloud_path)}: {len(xyz)} points, {len(points)} on the object")

        fits = {}
        T_on_board, _, _ = align(reference, points, FITS["on_board"], args.min_height, args.margin)
        T_shape, _, _ = align(reference, points, FITS["shape"], args.min_height, args.margin, start=T_on_board)
        for fit, T in (("on_board", T_on_board), ("shape", T_shape)):
            stats, _, _, _ = evaluate(reference, points, T, args)
            stats["reliable"] = bool(stats["coverage"] >= MIN_RELIABLE_COVERAGE)
            fits[fit] = (T, stats)

        out_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        T, stats = fits["on_board"]
        _, _, signed, keep = evaluate(reference, points, T, args)
        write_distance_ply(os.path.join(out_dir, f"{name}_distances.ply"), points[keep], signed, args.error_range)
        placed = o3d.geometry.TriangleMesh(reference.full)
        placed.transform(T)
        placed.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(out_dir, f"{name}_reference_aligned.ply"), placed)
        plot(os.path.join(out_dir, f"{name}_comparison.png"), name, reference, points[keep], signed, T, stats,
             args.error_range)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                       "cloud": rel(cloud_path), "reference": rel(args.reference),
                       "settings": {k: v for k, v in vars(args).items() if k not in ("clouds",)},
                       "fits": {fit: {"transform_board_from_cad": describe_transform(T), **s}
                                for fit, (T, s) in fits.items()}}, fh, indent=2, default=str)
        append_summary(os.path.join(args.out_dir, "summary.csv"), name, cloud_path, reference, fits)
        report(name, fits)
        print(f"  -> {rel(out_dir)}/")


if __name__ == "__main__":
    main()
