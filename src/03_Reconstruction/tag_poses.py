"""Pose of the stereo rig in every frame of a scan, from reference AprilTags triangulated in stereo.

The scan target is a flat board carrying a few large AprilTags (36h11, standard 1-bit border) around
the object. In each frame pair the tags both cameras see are detected, their corners refined to
sub-pixel and triangulated with the calibrated rig, which gives the tag corners in 3D in the left
camera frame. From those:

  1. the LAYOUT of the tags on the board is estimated from all frames at once (where each tag sits
     relative to the others), and a board frame is defined on it: origin at the centre of the tags,
     z out of the board towards the cameras, x along the first reference tag's x edge;
  2. each frame's POSE, T_cam_board, is the rigid fit of the layout onto that frame's triangulated
     corners.

Triangulating in stereo rather than solving PnP in one camera is deliberate: the depth then comes
from the 150 mm baseline instead of from the apparent size of a tag, which is the weakest direction
of a single-camera pose.

The layout is estimated from the triangulated corners, not built from the nominal tag size, so the
stereo scale is left untouched and the printed size becomes a CHECK: a tag edge that triangulates to
81 mm on an 80 mm print means the calibration scale is 1.3% off, and every reconstructed distance
carries the same error. (Sep24: 80.79 mm, so ~1.0% high.)

    src/venv/bin/python src/03_Reconstruction/tag_poses.py --session Sep24 --scan mjpg_pyr_lights_2

Reads the scan's frames from data/.../<session>/<scan>/frames and the session's calibration from
results/<session>/calibration/; writes results/<session>/<scan>/poses/tag_poses.yaml, which every
later stage reads (see src/project_paths.py for the layout).
"""
import argparse
import collections
import datetime
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_rig import (CAMERAS, find_frame_pairs, invert, load_rig, mean_rotation, rel,  # noqa: E402
                        rigid_fit, transform)
import project_paths as paths  # noqa: E402

# ====== REFERENCE TAGS ======
# Edge of the outer black square as printed, mm. Used to seed the layout and as the scale check;
# the layout itself is refined from the triangulated corners.
TAG_SIZE = 80.0

# The reference tags are ordinary AprilTags: 1-bit border, unlike the Kalibr calibration board (2).
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# Same refinement as Calibration.py: the detector's own refinement off, cornerSubPix on the image.
SUBPIX_WIN = (5, 5)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# Tag corners in the tag's own frame, in the order OpenCV reports them (top-left, top-right,
# bottom-right, bottom-left): x right, y up, so z = x cross y points out of the tag at the viewer.
TAG_MODEL = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float64) * TAG_SIZE / 2

LAYOUT_ITERATIONS = 10


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-frame rig pose from stereo-triangulated AprilTags.")
    paths.add_scan_arguments(ap)
    ap.add_argument("--calibration", default=None,
                    help="stereo calibration YAML from src/01_Calibration/Calibration.py "
                         "(default: the one in results/<session>/calibration/)")
    # The gate matters more than it looks. On Sep24/mjpg_pyr2, tags near the image centre triangulate
    # to 80.5-80.9 mm edges at ~0.35 px, but in the right camera's left 200 px they come out at 86 mm
    # with ~1.0 px: the calibration is weak at the periphery. At 2.0 px those tags stay in, and the
    # layout gets 84-86 mm edges and a 2.7 mm median fit residual; at 0.8 px every tag is 80.1-81.3 mm
    # and the residual 0.94 mm, at the cost of posing 211 of 365 frames instead of all of them.
    ap.add_argument("--max-reprojection", type=float, default=0.8,
                    help="px; a triangulated tag whose corners reproject worse than this is dropped "
                         "from that frame (default 0.8)")
    args = ap.parse_args(argv)
    paths.require_scan(args, ap)
    args.frames_dir = paths.frames_dir(args.session, args.scan)
    args.calibration = args.calibration or paths.default_calibration(args.session)
    return args


# ====== DETECTION + TRIANGULATION ======
def make_detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect(detector, path):
    """{tag id: 4x2 sub-pixel corners} for one image."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)
            .reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def triangulate_frame(rig, detector, left_path, right_path, max_reprojection):
    """Tags seen by both cameras in one pair, triangulated.

    Returns ({id: 4x3 corners in the left camera frame, mm}, {id: reprojection px}, [rejected ids]).
    """
    left, right = detect(detector, left_path), detect(detector, right_path)
    tags, errors, rejected = {}, {}, []
    for tag_id in sorted(left.keys() & right.keys()):
        X = rig.triangulate(left[tag_id], right[tag_id])
        error = rig.reprojection_error(X, left[tag_id], right[tag_id])
        if error > max_reprojection or np.any(X[:, 2] <= 0):
            rejected.append(tag_id)
            continue
        tags[tag_id], errors[tag_id] = X, error
    return tags, errors, rejected


# ====== LAYOUT ======
def initial_layout(observations):
    """First guess of each tag's corners, in the frame of the most often seen tag.

    Each tag is placed by averaging, over the frames where it was seen together with an already
    placed tag, the transform between the two. Tags are placed outwards from the reference, so a tag
    never seen with the reference is still reached through one that was.
    """
    counts = collections.Counter(t for frame in observations for t in frame)
    reference = counts.most_common(1)[0][0]
    # Pose of each tag in each frame, from the nominal square.
    tag_poses = [{t: rigid_fit(TAG_MODEL, X)[0] for t, X in frame.items()} for frame in observations]

    placed = {reference: np.eye(4)}   # tag id -> T_ref_tag
    while True:
        estimates = collections.defaultdict(list)
        for poses in tag_poses:
            anchors = [t for t in poses if t in placed]
            if not anchors:
                continue
            anchor = anchors[0]
            T_ref_cam = placed[anchor] @ invert(poses[anchor])
            for t in poses:
                if t not in placed:
                    estimates[t].append(T_ref_cam @ poses[t])
        if not estimates:
            break
        for t, Ts in estimates.items():
            T = np.eye(4)
            T[:3, :3] = mean_rotation([E[:3, :3] for E in Ts])
            T[:3, 3] = np.median([E[:3, 3] for E in Ts], axis=0)
            placed[t] = T

    unplaced = set(counts) - set(placed)
    if unplaced:
        print(f"  tags never seen together with the others, ignored: {sorted(unplaced)}")
    return reference, {t: transform(T, TAG_MODEL) for t, T in placed.items()}


def refine_layout(observations, layout):
    """Refine the tag corners by alternating: fit every frame to the layout, then move each corner to
    the mean of where the frames put it (generalised Procrustes). The nominal tag size drops out
    here, and what remains is the geometry the stereo rig actually measured."""
    for _ in range(LAYOUT_ITERATIONS):
        sums = {t: np.zeros((4, 3)) for t in layout}
        n = collections.Counter()
        for frame in observations:
            ids = [t for t in frame if t in layout]
            if not ids:
                continue
            T_cam_layout, _ = rigid_fit(np.vstack([layout[t] for t in ids]), np.vstack([frame[t] for t in ids]))
            T_layout_cam = invert(T_cam_layout)
            for t in ids:
                sums[t] += transform(T_layout_cam, frame[t])
                n[t] += 1
        layout = {t: sums[t] / n[t] for t in layout}
    return layout


def board_frame(layout, reference, camera_centres):
    """T_board_layout: origin at the centroid of all tag corners, z normal to the best-fit plane and
    pointing at the cameras, x along the reference tag's top edge projected onto the plane."""
    points = np.vstack(list(layout.values()))
    origin = points.mean(0)
    normal = np.linalg.svd(points - origin)[2][2]
    if np.dot(np.mean(camera_centres, 0) - origin, normal) < 0:
        normal = -normal
    x = layout[reference][1] - layout[reference][0]
    x -= np.dot(x, normal) * normal
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(normal, x), normal])   # rows: board axes in layout coordinates
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ origin
    return T


def estimate_layout(observations):
    """The tag layout in the board frame: {id: 4x3 corners, mm}, and the reference tag id."""
    reference, layout = initial_layout(observations)
    layout = refine_layout(observations, layout)
    # Camera centres in the layout frame, only to decide which side of the board is the front.
    centres = []
    for frame in observations:
        ids = [t for t in frame if t in layout]
        if ids:
            T_cam_layout, _ = rigid_fit(np.vstack([layout[t] for t in ids]), np.vstack([frame[t] for t in ids]))
            centres.append(invert(T_cam_layout)[:3, 3])
    T_board_layout = board_frame(layout, reference, centres)
    return reference, {t: transform(T_board_layout, c) for t, c in sorted(layout.items())}


# ====== POSES ======
def frame_pose(layout, tags):
    """T_cam_board for one frame and the RMS (mm) of the fit, or (None, None) with no usable tag."""
    ids = [t for t in tags if t in layout]
    if not ids:
        return None, None
    return rigid_fit(np.vstack([layout[t] for t in ids]), np.vstack([tags[t] for t in ids]))


def edge_lengths(corners):
    return [float(np.linalg.norm(corners[k] - corners[(k + 1) % 4])) for k in range(4)]


# ====== REPORT + SAVE ======
def report(layout, frames, observations):
    print("\nTAG LAYOUT (board frame, mm)")
    for t, corners in layout.items():
        centre = corners.mean(0)
        edges = edge_lengths(corners)
        print(f"  tag {t}: centre ({centre[0]:7.1f}, {centre[1]:7.1f}, {centre[2]:5.2f})  "
              f"edges {', '.join(f'{e:.2f}' for e in edges)}")
    all_edges = [e for c in layout.values() for e in edge_lengths(c)]
    print(f"  mean edge {np.mean(all_edges):.2f} mm vs {TAG_SIZE:.2f} printed -> stereo scale "
          f"{np.mean(all_edges) / TAG_SIZE:.4f} (1.0 = exact)")
    flatness = np.sqrt(np.mean(np.vstack(list(layout.values()))[:, 2] ** 2))
    print(f"  corners off the best-fit board plane: {flatness:.2f} mm RMS")

    single = [e for frame in observations for X in frame.values() for e in edge_lengths(X)]
    print(f"  single-frame edges: median {np.median(single):.2f}, "
          f"5-95% {np.percentile(single, 5):.2f} .. {np.percentile(single, 95):.2f} mm")

    posed = [f for f in frames if f["T_cam_board"] is not None]
    fits = np.array([f["fit_rms_mm"] for f in posed])
    print(f"\nPOSES: {len(posed)} of {len(frames)} frames")
    print(f"  tags per posed frame: " + ", ".join(
        f"{k}: {v}" for k, v in sorted(collections.Counter(len(f["tags"]) for f in posed).items())))
    print(f"  layout fit residual: median {np.median(fits):.2f} mm, max {fits.max():.2f} mm")
    centres = np.array([invert(f["T_cam_board"])[:3, 3] for f in posed])
    if len(centres) > 1:
        centred = centres - centres.mean(0)
        _, s, Vt = np.linalg.svd(centred)
        off_line = np.sqrt(np.mean(np.sum(centred ** 2, 1) - (centred @ Vt[0]) ** 2))
        print(f"  camera travel {np.ptp(centred @ Vt[0]):.0f} mm along direction "
              f"({', '.join(f'{v:.3f}' for v in Vt[0])}), {off_line:.1f} mm RMS off a straight line")


def write_yaml(path, args, rig, reference, layout, frames):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.writeComment("=====================================================================")
    fs.writeComment("RIG POSE PER FRAME, FROM STEREO-TRIANGULATED REFERENCE TAGS")
    fs.writeComment("Generated by src/03_Reconstruction/tag_poses.py - re-run the script rather than editing by hand.")
    fs.writeComment("T_cam_board maps a point in the board frame (mm) into the LEFT camera frame (mm):")
    fs.writeComment("  X_cam = T_cam_board @ [X_board, 1]")
    fs.writeComment("Board frame: origin at the centroid of the tag corners, z out of the board towards the")
    fs.writeComment(f"cameras, x along the top edge of tag {reference}.")
    fs.writeComment("=====================================================================")

    fs.startWriteStruct("metadata", cv2.FileNode_MAP)
    fs.write("created", datetime.datetime.now().isoformat(timespec="seconds"))
    fs.write("script", "src/03_Reconstruction/tag_poses.py")
    fs.write("opencv_version", cv2.__version__)
    fs.write("session", args.session)
    fs.write("scan", args.scan)
    fs.write("frames_dir", rel(args.frames_dir))
    fs.write("calibration", rel(args.calibration))
    fs.write("baseline_mm", rig.baseline_mm)
    fs.write("max_reprojection_px", args.max_reprojection)
    fs.endWriteStruct()

    fs.startWriteStruct("tags", cv2.FileNode_MAP)
    fs.write("family", "AprilTag 36h11")
    fs.write("printed_size_mm", TAG_SIZE)
    fs.write("reference_tag", reference)
    all_edges = [e for c in layout.values() for e in edge_lengths(c)]
    fs.writeComment("Mean triangulated edge / printed size: the scale error of the calibration, if the print is exact")
    fs.write("measured_size_mm", float(np.mean(all_edges)))
    fs.write("stereo_scale", float(np.mean(all_edges) / TAG_SIZE))
    fs.writeComment("Corners of each tag in the board frame, mm, in OpenCV's order (TL, TR, BR, BL)")
    fs.startWriteStruct("ids", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for t in layout:
        fs.write("", t)
    fs.endWriteStruct()
    fs.startWriteStruct("corners_mm", cv2.FileNode_SEQ)
    for corners in layout.values():
        fs.write("", corners)
    fs.endWriteStruct()
    fs.endWriteStruct()

    fs.writeComment("One entry per frame pair; frames without a usable tag have no T_cam_board.")
    fs.startWriteStruct("frames", cv2.FileNode_SEQ)
    for f in frames:
        fs.startWriteStruct("", cv2.FileNode_MAP)   # not FLOW: OpenCV cannot read a matrix back from a flow map
        fs.write("name", f["name"])
        fs.startWriteStruct("tags", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
        for t in f["tags"]:
            fs.write("", t)
        fs.endWriteStruct()
        if f["T_cam_board"] is not None:
            fs.write("fit_rms_mm", f["fit_rms_mm"])
            fs.write("reprojection_px", f["reprojection_px"])
            fs.write("T_cam_board", f["T_cam_board"])
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()


def main():
    args = parse_args()
    rig = load_rig(args.calibration)
    pairs = find_frame_pairs(args.frames_dir)
    detector = make_detector()
    print(f"{len(pairs)} frame pairs in {rel(args.frames_dir)}, baseline {rig.baseline_mm:.2f} mm")

    observations, errors, n_rejected = [], [], 0
    for name, left_path, right_path in pairs:
        tags, tag_errors, rejected = triangulate_frame(rig, detector, left_path, right_path, args.max_reprojection)
        observations.append(tags)
        errors.append(tag_errors)
        n_rejected += len(rejected)
    counts = collections.Counter(t for frame in observations for t in frame)
    print("tags triangulated (frames): " + ", ".join(f"{t}: {n}" for t, n in sorted(counts.items())))
    print(f"tags dropped for reprojection > {args.max_reprojection} px: {n_rejected}")
    if not counts:
        raise SystemExit("no tag was seen by both cameras in any frame")

    reference, layout = estimate_layout(observations)

    frames = []
    for (name, _, _), tags, tag_errors in zip(pairs, observations, errors):
        T, fit = frame_pose(layout, tags)
        frames.append({"name": name, "tags": sorted(t for t in tags if t in layout),
                       "T_cam_board": T, "fit_rms_mm": fit,
                       "reprojection_px": float(np.mean(list(tag_errors.values()))) if tag_errors else None})
    report(layout, frames, observations)

    out_path = paths.poses_path(args.session, args.scan)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_yaml(out_path, args, rig, reference, layout, frames)
    print(f"\nsaved poses to {rel(out_path)}")


if __name__ == "__main__":
    main()
