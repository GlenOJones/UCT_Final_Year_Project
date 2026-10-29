"""Web-sized copies of the fusion clouds for the GitHub Pages viewer (docs/viewer/index.html).

    S30  : results/experiments/fusion/3DPrintFinal_s30/objects/<method>/<object>.ply  (already one object each)
    Ice  : <work>/<scan>/<method>/cloud_board.ply cut to evaluation.region and to the ice's height band,
           with z flipped to 'height towards the camera' as in the figures.

    src/venv/bin/python src/experiments/fusion/export_web_clouds.py
    src/venv/bin/python src/experiments/fusion/export_web_clouds.py --voxel 2.5     # smaller files
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
import compare_fusion  # noqa: E402
from run_fusion import recording_paths  # noqa: E402

ICE_CONFIGS = {"config_pyramid_shade.json": "Ice pyramid (shade)", "config_cubes.json": "Ice cubes (shade)",
               "config_wave.json": "Ice wave (shade)", "config_overhang.json": "Ice overhang (shade)"}
S30_CONFIG = "config.json"
S30_TITLES = {"PYRAMID_1": "Printed pyramid", "PYRAMID_2X2": "Printed pyramids 2x2", "EGG_1PK": "Egg carton, 1 pack",
              "EGG_2PKS": "Egg carton, 2 packs"}
HEIGHT_PERCENTILES = (10, 99.5)   # ice_grid.py's band: leaves out the pole and background behind the ice
HEIGHT_MARGIN_MM = 10
OUT_DIR = os.path.join(common.PROJECT_ROOT, "docs", "viewer", "clouds")


def read_cloud(path):
    cloud = o3d.io.read_point_cloud(path)
    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors) if cloud.has_colors() else np.full_like(xyz, 0.7)
    return xyz, rgb


def thin(xyz, rgb, voxel):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    cloud = cloud.voxel_down_sample(voxel)
    return np.asarray(cloud.points), np.asarray(cloud.colors)


def write_ply(path, xyz, rgb):
    """Binary little-endian PLY, 15 bytes a point."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vertices = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertices["x"], vertices["y"], vertices["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertices["r"], vertices["g"], vertices["b"] = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8).T
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(vertices.tobytes())


def export_s30(voxel, entries):
    cfg = common.load_config(os.path.join(HERE, S30_CONFIG))
    p = recording_paths(cfg)
    for obj, title in S30_TITLES.items():
        methods = []
        for m in cfg["methods"]:
            path = os.path.join(p["out"], "objects", m["name"], f"{obj}.ply")
            if not os.path.exists(path):
                continue
            xyz, rgb = thin(*read_cloud(path), voxel)
            write_ply(os.path.join(OUT_DIR, f"s30_{obj}", f"{m['name']}.ply"), xyz, rgb)
            methods.append({"name": m["name"], "label": m["label"], "file": f"s30_{obj}/{m['name']}.ply", "points": len(xyz)})
            print(f"  {obj} {m['name']}: {len(xyz):,} points")
        entries.append({"id": f"s30_{obj}", "title": title, "group": "Printed objects (S30)", "methods": methods})


def export_ice(voxel, entries):
    for config, title in ICE_CONFIGS.items():
        cfg = common.load_config(os.path.join(HERE, config))
        p = recording_paths(cfg)
        ev = cfg["evaluation"]
        x0, x1, y0, y1 = ev["region"]
        side = compare_fusion.camera_side(p)
        raw = {}
        for m in cfg["methods"]:
            path = os.path.join(p["method"](m["name"]), "cloud_board.ply")
            if not os.path.exists(path):
                continue
            xyz, rgb = read_cloud(path)
            xyz = xyz * np.array([1.0, 1.0, side])
            keep = (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            raw[m["name"]] = (xyz[keep], rgb[keep])
        low, high = ev.get("grid_height_range_mm") or np.percentile(
            np.concatenate([x[:, 2] for x, _ in raw.values()]), HEIGHT_PERCENTILES)
        low, high = low - HEIGHT_MARGIN_MM, high + HEIGHT_MARGIN_MM
        methods = []
        for m in cfg["methods"]:
            if m["name"] not in raw:
                continue
            xyz, rgb = raw[m["name"]]
            keep = (xyz[:, 2] >= low) & (xyz[:, 2] <= high)
            xyz, rgb = thin(xyz[keep], rgb[keep], voxel)
            name = cfg["recording"]["name"]
            write_ply(os.path.join(OUT_DIR, name, f"{m['name']}.ply"), xyz, rgb)
            methods.append({"name": m["name"], "label": m["label"], "file": f"{name}/{m['name']}.ply", "points": len(xyz)})
            print(f"  {name} {m['name']}: {len(xyz):,} points")
        entries.append({"id": cfg["recording"]["name"], "title": title, "group": "Ice (shade)", "methods": methods})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxel", type=float, default=1.5, help="mm; one point kept per voxel (default 1.5)")
    args = ap.parse_args(argv)
    entries = []
    print("S30 objects")
    export_s30(args.voxel, entries)
    print("Ice scans")
    export_ice(args.voxel, entries)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"voxel_mm": args.voxel, "scans": entries}, fh, indent=1)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT_DIR) for f in fs)
    print(f"wrote {OUT_DIR}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
