"""Dense point cloud of a scan: stereo matching on every posed frame pair, fused in the board frame.

Each frame pair is rectified with the transforms saved by Calibration.py, matched with semi-global
block matching (SGBM), and the disparity turned into 3D points in the left camera frame. The pose
from tag_poses.py then moves those points into the board frame, where every frame lands in the same
place and they can be merged.

Merging is on a voxel grid. Each voxel keeps the mean of the points that fell in it and the number
of FRAMES that put a point there. Keeping only voxels seen from several frames is the main noise
filter: a wrong match lands somewhere different in each frame, while a real surface keeps coming
back to the same place as the rig slides past.

Only a box around the tags is kept: the tag area plus a margin, from slightly behind the board to
--z-max in front of it. The same box also sets each frame's disparity search range, so SGBM does
not spend time on depths that would be cropped anyway.

    src/venv/bin/python src/03_Reconstruction/dense_stereo.py
    src/venv/bin/python src/03_Reconstruction/dense_stereo.py --step 5 --debug-every 1

Writes results/reconstruction/<recording>_cloud.ply (binary PLY: x, y, z in mm in the board frame,
grey as RGB, and "views", the number of frames that saw each point). CloudCompare and MeshLab open
it directly; view_cloud.py renders it.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import PROJECT_ROOT, RESULTS_DIR, invert, load_rig, rel, transform  # noqa: E402

# ====== SGBM ======
# Matching is on grey: the frames are the single-channel luma (Y) plane that extract_stereo_frames.py
# --gray saves; the scene is achromatic and the MJPG chroma is half resolution, so colour would add
# little. Block size and uniqueness are arguments. The rest, measured on Sep24/mjpg_pyr2 frame 0110 (board plain wood, pyramid matte black):
#   MODE_HH (8 paths) instead of 3WAY removes the horizontal streaking on the low-texture board
#   block 9 / uniqueness 5 finds ~1.7x the pyramid points of block 5 / uniqueness 10
#   CLAHE before matching does NOT help: it adds speckle at wrong depths all over the board
SGBM_MODE = cv2.STEREO_SGBM_MODE_HH
SPECKLE_WINDOW = 200      # px, connected disparity blobs smaller than this are removed
SPECKLE_RANGE = 2         # disparity steps allowed within a blob
DISP12_MAX_DIFF = 1       # px, left-right consistency check

MERGE_EVERY = 10          # frames between voxel merges, which keeps memory bounded


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Fuse per-frame stereo depth into one point cloud.")
    ap.add_argument("--poses", default=os.path.join(RESULTS_DIR, "mjpg_pyr2_tag_poses.yaml"),
                    help="per-frame poses from tag_poses.py; also names the frames and calibration")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--step", type=int, default=1, help="use every Nth posed frame (default 1)")
    ap.add_argument("--voxel", type=float, default=1.0, help="mm, fusion grid size (default 1)")
    ap.add_argument("--min-views", type=int, default=3,
                    help="frames that must put a point in a voxel for it to be kept (default 3)")
    ap.add_argument("--margin", type=float, default=60.0,
                    help="mm kept around the tags in x and y (default 60)")
    ap.add_argument("--z-min", type=float, default=-15.0, help="mm, behind the board plane (default -15)")
    ap.add_argument("--z-max", type=float, default=150.0, help="mm, in front of the board (default 150)")
    ap.add_argument("--block-size", type=int, default=9, help="SGBM window, px, odd (default 9)")
    ap.add_argument("--uniqueness", type=int, default=5,
                    help="SGBM uniqueness ratio, %%: best match must beat the next by this (default 5)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output name, to keep runs with different settings apart")
    ap.add_argument("--debug-every", type=int, default=0,
                    help="save rectified image + disparity for every Nth used frame (0 = never)")
    return ap.parse_args(argv)


# ====== INPUTS ======
def load_poses(path):
    """Poses YAML from tag_poses.py -> (metadata dict, tag corners (N, 3) in the board frame,
    [(name, T_cam_board)] for the frames that have a pose)."""
    if not os.path.exists(path):
        raise SystemExit(f"no poses at {rel(path)} - run src/03_Reconstruction/tag_poses.py first")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    meta = fs.getNode("metadata")
    metadata = {key: meta.getNode(key).string() for key in ("frames_dir", "calibration")}
    corners_node = fs.getNode("tags").getNode("corners_mm")
    tag_corners = np.vstack([corners_node.at(i).mat() for i in range(corners_node.size())])
    frames = []
    frames_node = fs.getNode("frames")
    for i in range(frames_node.size()):
        node = frames_node.at(i)
        if not node.getNode("T_cam_board").isNone():
            frames.append((node.getNode("name").string(), node.getNode("T_cam_board").mat()))
    fs.release()
    return metadata, tag_corners, frames


def crop_box(tag_corners, args):
    """(low, high) corners of the kept box in the board frame, mm."""
    low = np.r_[tag_corners[:, :2].min(0) - args.margin, args.z_min]
    high = np.r_[tag_corners[:, :2].max(0) + args.margin, args.z_max]
    return low, high


# ====== STEREO ======
class Rectifier:
    """Rectification maps for the pair, built once, and the geometry needed to go back from
    rectified-left coordinates to the left camera frame."""

    def __init__(self, rig):
        self.rig = rig
        self.maps = {camera: cv2.initUndistortRectifyMap(rig.K[camera], rig.D[camera], R, P, rig.image_size, cv2.CV_16SC2)
                     for camera, R, P in (("left", rig.R1, rig.P1), ("right", rig.R2, rig.P2))}
        self.focal = float(rig.P1[0, 0])
        self.baseline = float(-rig.P2[0, 3] / rig.P2[0, 0])
        # Points from reprojectImageTo3D are in the rectified left frame; R1 rotated the camera
        # into it, so R1 transposed rotates them back.
        self.T_cam_rect = np.eye(4)
        self.T_cam_rect[:3, :3] = rig.R1.T

    def rectify(self, camera, path):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.remap(gray, *self.maps[camera], cv2.INTER_LINEAR)

    def disparity_range(self, T_cam_board, low, high):
        """(min disparity, number of disparities) covering the crop box as seen in this frame."""
        box = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        depth = transform(invert(self.T_cam_rect) @ T_cam_board, box)[:, 2]
        d_min = self.focal * self.baseline / depth.max()
        d_max = self.focal * self.baseline / max(depth.min(), 1.0)
        minimum = max(int(np.floor(d_min)) - 8, 0)
        count = int(np.ceil((d_max - minimum + 8) / 16) * 16)
        return minimum, count


def match(left, right, minimum, count, block_size, uniqueness):
    """SGBM disparity (px, float) of a rectified pair; pixels with no match are below minimum.

    SGBM leaves the first minimum + count columns unmatched, which with the ~230 + ~200 px search
    this rig needs is a quarter of the image. Padding both images on the left by that much moves
    the dead band into the padding, and it is cropped off again afterwards.
    """
    pad = minimum + count
    padded = [cv2.copyMakeBorder(image, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=0) for image in (left, right)]
    matcher = cv2.StereoSGBM.create(
        minDisparity=minimum, numDisparities=count, blockSize=block_size,
        P1=8 * block_size ** 2, P2=32 * block_size ** 2,
        disp12MaxDiff=DISP12_MAX_DIFF, uniquenessRatio=uniqueness,
        speckleWindowSize=SPECKLE_WINDOW, speckleRange=SPECKLE_RANGE, mode=SGBM_MODE)
    disparity = matcher.compute(*padded)[:, pad:].astype(np.float32) / 16.0
    # A left pixel whose match would lie left of the right image's edge can only have matched the
    # padding, so it is not a measurement.
    disparity[disparity > np.arange(left.shape[1], dtype=np.float32)] = minimum - 1
    return disparity


def frame_points(rectifier, left_path, right_path, T_cam_board, low, high, args):
    """Board-frame points and grey values for one pair, cropped to the box. Also returns the
    rectified left image and disparity for debugging."""
    left, right = rectifier.rectify("left", left_path), rectifier.rectify("right", right_path)
    minimum, count = rectifier.disparity_range(T_cam_board, low, high)
    disparity = match(left, right, minimum, count, args.block_size, args.uniqueness)
    valid = disparity >= minimum   # SGBM marks unmatched pixels minimum - 1
    xyz_rect = cv2.reprojectImageTo3D(disparity, rectifier.rig.Q)[valid]
    T_board_rect = invert(T_cam_board) @ rectifier.T_cam_rect
    xyz = transform(T_board_rect, xyz_rect.astype(np.float64))
    inside = np.all((xyz >= low) & (xyz <= high), axis=1)
    return xyz[inside], left[valid][inside], left, disparity, (minimum, count)


# ====== FUSION ======
class VoxelGrid:
    """Running per-voxel sums: position, grey, point count and the number of frames that hit it."""

    def __init__(self, voxel):
        self.voxel = voxel
        self.keys = np.empty(0, np.int64)
        self.sums = np.empty((0, 5))   # x, y, z, grey, points
        self.views = np.empty(0, np.int64)
        self.pending = []

    def key(self, xyz):
        # 21 bits per axis: +-1 km at 1 mm, far beyond any box this is used with.
        index = np.floor(xyz / self.voxel).astype(np.int64) + (1 << 20)
        return (index[:, 0] << 42) | (index[:, 1] << 21) | index[:, 2]

    def add(self, xyz, grey):
        keys = self.key(xyz)
        # Collapse within the frame first, so a frame counts once per voxel however many pixels hit it.
        unique, inverse = np.unique(keys, return_inverse=True)
        sums = np.zeros((len(unique), 5))
        np.add.at(sums, inverse, np.c_[xyz, grey, np.ones(len(xyz))])
        self.pending.append((unique, sums, np.ones(len(unique), np.int64)))

    def merge(self):
        if not self.pending:
            return
        keys = np.concatenate([self.keys] + [p[0] for p in self.pending])
        sums = np.vstack([self.sums] + [p[1] for p in self.pending])
        views = np.concatenate([self.views] + [p[2] for p in self.pending])
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.sums = np.zeros((len(self.keys), 5))
        np.add.at(self.sums, inverse, sums)
        self.views = np.bincount(inverse, weights=views).astype(np.int64)
        self.pending = []

    def points(self, min_views):
        self.merge()
        keep = self.views >= min_views
        sums = self.sums[keep]
        return sums[:, :3] / sums[:, 4:5], sums[:, 3] / sums[:, 4], self.views[keep]


def write_ply(path, xyz, grey, views):
    """Binary little-endian PLY: float x y z, uchar red green blue, ushort views."""
    record = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("views", "<u2")])
    data = np.empty(len(xyz), record)
    data["x"], data["y"], data["z"] = xyz.T
    grey = np.clip(np.round(grey), 0, 255).astype(np.uint8)
    data["red"] = data["green"] = data["blue"] = grey
    data["views"] = np.minimum(views, 65535)
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment board frame, mm; written by src/03_Reconstruction/dense_stereo.py\n"
              f"element vertex {len(data)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property ushort views\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.tobytes())


def save_debug(debug_dir, name, left, disparity, disparity_range):
    minimum, count = disparity_range
    scaled = np.clip((disparity - minimum) / count * 255, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colour[disparity <= minimum] = 0
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, name), np.hstack([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), colour]))


def main():
    args = parse_args()
    metadata, tag_corners, frames = load_poses(args.poses)
    frames_dir = os.path.join(PROJECT_ROOT, metadata["frames_dir"])
    rig = load_rig(os.path.join(PROJECT_ROOT, metadata["calibration"]))
    rectifier = Rectifier(rig)
    low, high = crop_box(tag_corners, args)
    recording = os.path.basename(os.path.normpath(frames_dir)) + (f"_{args.tag}" if args.tag else "")
    debug_dir = os.path.join(args.out_dir, f"{recording}_disparity")

    used = frames[::args.step]
    print(f"{len(used)} of {len(frames)} posed frames from {rel(frames_dir)}")
    print(f"box (board frame, mm): x {low[0]:.0f}..{high[0]:.0f}, y {low[1]:.0f}..{high[1]:.0f}, "
          f"z {low[2]:.0f}..{high[2]:.0f}; voxel {args.voxel} mm; "
          f"SGBM block {args.block_size}, uniqueness {args.uniqueness}")

    grid = VoxelGrid(args.voxel)
    start = time.time()
    for k, (name, T_cam_board) in enumerate(used):
        xyz, grey, left, disparity, disparity_range = frame_points(
            rectifier, os.path.join(frames_dir, "left", name), os.path.join(frames_dir, "right", name),
            T_cam_board, low, high, args)
        grid.add(xyz, grey)
        if args.debug_every and k % args.debug_every == 0:
            save_debug(debug_dir, name, left, disparity, disparity_range)
        if (k + 1) % MERGE_EVERY == 0:
            grid.merge()
            print(f"  {k + 1}/{len(used)}  {name}: {len(xyz)} points in box, "
                  f"disparity {disparity_range[0]}+{disparity_range[1]}, "
                  f"{len(grid.keys)} voxels so far, {time.time() - start:.0f} s")

    xyz, grey, views = grid.points(args.min_views)
    print(f"\n{len(xyz)} points seen from >= {args.min_views} frames "
          f"(of {len(grid.keys)} voxels hit at all)")
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{recording}_cloud.ply")
    write_ply(out_path, xyz, grey, views)
    print(f"saved {rel(out_path)}")
    if args.debug_every:
        print(f"disparity images in {rel(debug_dir)}")


if __name__ == "__main__":
    main()
