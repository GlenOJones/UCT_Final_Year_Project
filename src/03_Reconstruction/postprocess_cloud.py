"""Clean a reconstructed cloud and turn it into a surface mesh, with Open3D.

Takes any cloud already in the board frame (dense_stereo.py's or colmap_mvs.py's) and:
  1. crops it to the tag area plus a margin, from just behind the board to --z-max in front;
  2. removes statistical outliers: points whose mean distance to their neighbours is far above the
     typical one, which is what isolated wrong matches look like;
  3. estimates normals where the cloud has none, pointing them towards the cameras (+z);
  4. meshes it with screened Poisson reconstruction, then trims the triangles Poisson invented
     where there were no points (low-density vertices) - otherwise the mesh balloons over the gaps.

    src/venv/bin/python src/03_Reconstruction/postprocess_cloud.py --session Sep24 --scan mjpg_pyr_lights_2 --method colmap
    src/venv/bin/python src/03_Reconstruction/postprocess_cloud.py --cloud path/to/cloud.ply --poses path/to/tag_poses.yaml --show

Writes cloud_clean.ply, mesh.ply and a shaded render mesh.png beside the input. The clean cloud is what to align to
the CAD model; the mesh is for looking at and for cloud-to-mesh distances the other way round.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import rel  # noqa: E402
import project_paths as paths  # noqa: E402

# Same light -> dark blue height ramp as view_cloud.py.
HEIGHT_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE = (0.988, 0.988, 0.984)
# Oblique viewpoint for the render, board frame (mm): in front of and above the board.
RENDER_EYE = (300.0, -520.0, 480.0)
RENDER_SIZE = (1400, 900)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Crop, de-noise and mesh a board-frame point cloud.")
    paths.add_cloud_arguments(ap)
    ap.add_argument("--margin", type=float, default=60.0, help="mm kept around the tags (default 60)")
    ap.add_argument("--z-min", type=float, default=-15.0, help="mm (default -15)")
    ap.add_argument("--z-max", type=float, default=150.0, help="mm (default 150)")
    ap.add_argument("--neighbours", type=int, default=20, help="outlier test neighbourhood (default 20)")
    ap.add_argument("--std-ratio", type=float, default=2.0,
                    help="drop points this many std above the mean neighbour distance (default 2)")
    ap.add_argument("--poisson-depth", type=int, default=10,
                    help="octree depth; 10 gives ~0.6 mm cells over this ~600 mm box (default 10)")
    ap.add_argument("--density-quantile", type=float, default=0.05,
                    help="trim mesh vertices below this quantile of Poisson density (default 0.05)")
    ap.add_argument("--show", action="store_true", help="open an interactive Open3D window")
    args = ap.parse_args(argv)
    args.cloud, args.poses = paths.resolve_cloud(args, ap)
    return args


def read_tags(path):
    """{id: 4x3 corners} in the board frame, from tag_poses.py."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    tags = fs.getNode("tags")
    ids = [int(tags.getNode("ids").at(i).real()) for i in range(tags.getNode("ids").size())]
    corners = [tags.getNode("corners_mm").at(i).mat() for i in range(len(ids))]
    fs.release()
    return dict(zip(ids, corners))


def tag_outlines(tags):
    """The tags as a LineSet, for the viewer."""
    points = np.vstack(list(tags.values()))
    lines = [[4 * k + i, 4 * k + (i + 1) % 4] for k in range(len(tags)) for i in range(4)]
    outline = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
    outline.paint_uniform_color((0.1, 0.1, 0.1))
    return outline


def clean(cloud, tags, args):
    corners = np.vstack(list(tags.values()))
    box = o3d.geometry.AxisAlignedBoundingBox(
        np.r_[corners[:, :2].min(0) - args.margin, args.z_min],
        np.r_[corners[:, :2].max(0) + args.margin, args.z_max])
    cropped = cloud.crop(box)
    filtered, _ = cropped.remove_statistical_outlier(args.neighbours, args.std_ratio)
    print(f"  {len(cloud.points)} points -> {len(cropped.points)} in box -> "
          f"{len(filtered.points)} after outlier removal")
    if not filtered.has_normals():
        spacing = np.median(np.asarray(filtered.compute_nearest_neighbor_distance()))
        filtered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(6 * spacing, 2.0), max_nn=30))
    # Every surface here was seen from the cameras' side of the board, so +z is "outwards".
    filtered.orient_normals_to_align_with_direction((0.0, 0.0, 1.0))
    return filtered, box


def mesh(cloud, box, args):
    surface, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=args.poisson_depth)
    densities = np.asarray(densities)
    surface.remove_vertices_by_mask(densities < np.quantile(densities, args.density_quantile))
    surface = surface.crop(box)
    surface.remove_unreferenced_vertices()
    surface.compute_vertex_normals()
    print(f"  Poisson depth {args.poisson_depth}: {len(surface.triangles)} triangles after trimming")
    return surface


def render(surface, path, z_range):
    """Shaded image of the mesh, coloured by height, by casting rays on the CPU.

    Open3D's OffscreenRenderer produced blank images on this machine (Filament on Vulkan, no
    display), so the render is ray cast instead: it needs no GPU context and is exact.
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("height", HEIGHT_RAMP)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface))
    width, height = RENDER_SIZE
    rays = scene.create_rays_pinhole(fov_deg=40, center=[0, 0, 10], eye=list(RENDER_EYE), up=[0, 0, 1],
                                     width_px=width, height_px=height)
    hits = scene.cast_rays(rays)
    t, normals, rays = hits["t_hit"].numpy(), hits["primitive_normals"].numpy(), rays.numpy()
    hit = np.isfinite(t)
    points = rays[..., :3] + rays[..., 3:] * t[..., None]
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    shade = 0.35 + 0.65 * np.abs(normals @ light)
    image = np.ones((height, width, 3)) * SURFACE
    heights = np.clip((points[hit, 2] - z_range[0]) / (z_range[1] - z_range[0]), 0, 1)
    image[hit] = cmap(heights)[:, :3] * shade[hit, None]
    cv2.imwrite(path, (image[..., ::-1] * 255).astype(np.uint8))


def main():
    args = parse_args()
    cloud = o3d.io.read_point_cloud(args.cloud)
    if cloud.is_empty():
        raise SystemExit(f"no points in {rel(args.cloud)}")
    tags = read_tags(args.poses)
    print(f"{rel(args.cloud)}")

    cleaned, box = clean(cloud, tags, args)
    surface = mesh(cleaned, box, args)

    folder = os.path.dirname(os.path.abspath(args.cloud))
    clean_path, mesh_path = os.path.join(folder, paths.CLOUD_CLEAN), os.path.join(folder, paths.MESH)
    render_path = os.path.splitext(mesh_path)[0] + ".png"
    o3d.io.write_point_cloud(clean_path, cleaned)
    o3d.io.write_triangle_mesh(mesh_path, surface)
    z = np.asarray(cleaned.points)[:, 2]
    render(surface, render_path, (max(np.percentile(z, 1), -5.0), max(np.percentile(z, 99.9), 10.0)))
    print(f"saved {rel(clean_path)}, {rel(mesh_path)} and {rel(render_path)}")

    if args.show:
        o3d.visualization.draw_geometries([cleaned, tag_outlines(tags)], window_name="cleaned cloud")
        o3d.visualization.draw_geometries([surface, tag_outlines(tags)], window_name="Poisson mesh",
                                          mesh_show_back_face=True)


if __name__ == "__main__":
    main()
