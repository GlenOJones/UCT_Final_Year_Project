"""Reading and writing camera trajectories: one rig pose per frame pair, from any pose estimator.

    frames:
       - { name: 0123.png, T_cam_world: 4x4 }      X_cam = T_cam_world @ [X_world, 1], mm
"""
import datetime
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stereo_rig import rel  # noqa: E402


def write_trajectory(path, script, world, metadata, frames, comment=None):
    """frames: [{"name": str, "T_cam_world": 4x4 or None, <diagnostic>: float, ...}], in frame order."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.writeComment("=====================================================================")
    fs.writeComment(f"RIG POSE PER FRAME, from {script} - re-run it rather than editing by hand.")
    fs.writeComment("T_cam_world maps a point in the world frame (mm) into the LEFT camera frame (mm):")
    fs.writeComment("  X_cam = T_cam_world @ [X_world, 1]")
    fs.writeComment(f"World frame: {world}")
    for line in (comment or "").splitlines():
        fs.writeComment(line)
    fs.writeComment("=====================================================================")
    fs.startWriteStruct("metadata", cv2.FileNode_MAP)
    fs.write("created", datetime.datetime.now().isoformat(timespec="seconds"))
    fs.write("script", script)
    fs.write("world_frame", world)
    for key, value in metadata.items():
        fs.write(key, value if not isinstance(value, bool) else int(value))
    fs.endWriteStruct()
    fs.startWriteStruct("frames", cv2.FileNode_SEQ)
    for f in frames:
        fs.startWriteStruct("", cv2.FileNode_MAP)
        fs.write("name", f["name"])
        for key, value in f.items():
            if key in ("name", "T_cam_world") or value is None:
                continue
            fs.write(key, float(value))
        if f.get("T_cam_world") is not None:
            fs.write("T_cam_world", np.asarray(f["T_cam_world"], np.float64))
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()


def read_trajectory(path):
    """(metadata {key: str or float}, {frame name: T_cam_world}, {frame name: {diagnostic: float}}) from a trajectory
    file or from tag_poses.py's YAML (T_cam_board)."""
    if not os.path.exists(path):
        raise SystemExit(f"no poses at {rel(path)}")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    meta_node = fs.getNode("metadata")
    metadata = {}
    for key in meta_node.keys():
        node = meta_node.getNode(key)
        metadata[key] = node.string() if node.isString() else node.real()
    poses, diagnostics = {}, {}
    frames = fs.getNode("frames")
    for i in range(frames.size()):
        node = frames.at(i)
        name = node.getNode("name").string()
        for key in ("T_cam_world", "T_cam_board"):
            if not node.getNode(key).isNone():
                poses[name] = node.getNode(key).mat()
                break
        diagnostics[name] = {k: node.getNode(k).real() for k in node.keys()
                             if k not in ("name", "T_cam_world", "T_cam_board") and node.getNode(k).isReal()}
    fs.release()
    return metadata, poses, diagnostics
