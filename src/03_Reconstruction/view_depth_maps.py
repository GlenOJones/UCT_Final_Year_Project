"""Look at COLMAP's PatchMatch depth maps, including while patch_match_stereo is still running.

Each depth map is shown beside the undistorted image it belongs to, near = red, far = blue, no
depth = black. The first pass writes <name>.photometric.bin (unfiltered, so noisy); the second
writes <name>.geometric.bin (checked against the neighbouring views), and --type picks which.

    src/venv/bin/python src/03_Reconstruction/view_depth_maps.py --watch          # live window
    src/venv/bin/python src/03_Reconstruction/view_depth_maps.py --image left/0110.png

--watch shows the newest depth map and refreshes as new ones are written (q or Esc to quit).
Without it, the chosen (or newest) map is saved as a PNG in the workspace's depth_previews/.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import RESULTS_DIR, rel  # noqa: E402

DEPTH_PERCENTILES = (2, 98)   # colour range, so a few wild depths do not wash the map out
DISPLAY_WIDTH = 1600


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="View COLMAP PatchMatch depth maps.")
    ap.add_argument("--workspace", default=os.path.join(RESULTS_DIR, "mjpg_pyr2_colmap"))
    ap.add_argument("--type", default="auto", choices=("auto", "photometric", "geometric"),
                    help="which pass; auto = geometric if any exist yet, else photometric")
    ap.add_argument("--image", default=None, help='e.g. "left/0110.png" (default: the newest map)')
    ap.add_argument("--watch", action="store_true", help="live window following the newest map")
    ap.add_argument("--range", type=float, nargs=2, metavar=("NEAR", "FAR"), default=None,
                    help="colour range in mm, e.g. --range 350 550 to spread the board and object "
                         "over the whole colour scale (default: 2nd-98th percentile of the map)")
    return ap.parse_args(argv)


def read_colmap_array(path):
    """COLMAP's .bin depth/normal format: an ASCII "width&height&channels&" header, then float32
    values stored column-major."""
    with open(path, "rb") as fh:
        header = b""
        while header.count(b"&") < 3:
            header += fh.read(1)
        width, height, channels = (int(v) for v in header.split(b"&")[:3])
        data = np.fromfile(fh, np.float32)
    return data.reshape((width, height, channels), order="F").transpose(1, 0, 2).squeeze()


def depth_dir(workspace):
    return os.path.join(workspace, "dense", "stereo", "depth_maps")


def list_maps(workspace, kind):
    """All depth maps of one pass, oldest first by modification time."""
    if kind == "auto":
        kind = "geometric" if glob.glob(os.path.join(depth_dir(workspace), "*", "*.geometric.bin")) else "photometric"
    paths = glob.glob(os.path.join(depth_dir(workspace), "*", f"*.{kind}.bin"))
    return sorted(paths, key=os.path.getmtime), kind


def render(workspace, map_path, depth_range=None):
    """Image and colour-coded depth side by side, with a caption."""
    name = os.path.relpath(map_path, depth_dir(workspace)).split(".png.")[0] + ".png"
    kind = map_path.rsplit(".", 2)[-2]
    depth = read_colmap_array(map_path)
    image = cv2.imread(os.path.join(workspace, "dense", "images", name))
    valid = depth > 0
    if depth_range:
        near, far = depth_range
    elif valid.any():
        near, far = np.percentile(depth[valid], DEPTH_PERCENTILES)
    else:
        near, far = 0.0, 1.0
    scaled = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)
    colour = cv2.applyColorMap((255 * (1 - scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    panel = np.hstack([image, colour])
    panel = cv2.resize(panel, (DISPLAY_WIDTH, int(panel.shape[0] * DISPLAY_WIDTH / panel.shape[1])))
    caption = (f"{name}  {kind}  depth {near:.0f}-{far:.0f} mm (red near, blue far)  "
               f"{100 * valid.mean():.0f}% of pixels have depth")
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(panel, caption, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel, name, kind


def main():
    args = parse_args()
    if not os.path.isdir(depth_dir(args.workspace)):
        raise SystemExit(f"no depth maps under {rel(args.workspace)} yet - has patch_match_stereo started?")

    if args.watch:
        shown = None
        while True:
            maps, _ = list_maps(args.workspace, args.type)
            # The newest file may still be being written; show the one before it.
            if len(maps) >= 2 and maps[-2] != shown:
                shown = maps[-2]
                try:
                    panel, _, _ = render(args.workspace, shown, args.range)
                    cv2.imshow("PatchMatch depth (q to quit)", panel)
                except (ValueError, OSError):
                    shown = None   # caught mid-write; try again next round
            if cv2.waitKey(2000) in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        return

    maps, kind = list_maps(args.workspace, args.type)
    if args.image:
        path = os.path.join(depth_dir(args.workspace), f"{args.image}.{kind}.bin")
        if not os.path.exists(path):
            raise SystemExit(f"no {kind} depth map for {args.image} yet")
    elif len(maps) >= 2:
        path = maps[-2]
    else:
        raise SystemExit("no finished depth map yet")
    panel, name, kind = render(args.workspace, path, args.range)
    out_dir = os.path.join(args.workspace, "depth_previews")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name.replace('/', '_')}.{kind}.png")
    cv2.imwrite(out_path, panel)
    print(f"saved {rel(out_path)}")


if __name__ == "__main__":
    main()
