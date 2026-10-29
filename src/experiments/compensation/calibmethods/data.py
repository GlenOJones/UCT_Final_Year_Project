"""The board observations of one Calibration.py run, read from what that run saved. Nothing is detected here."""
import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(SRC_DIR, "01_Calibration"))
import Calibration  # noqa: E402  (board geometry: one definition, shared with the pipeline)

CAMERAS = Calibration.CAMERAS


@dataclass
class Frame:
    name: str                 # "Water_C1_frames/c1_Left_100.png", as Calibration.py names it
    group: str                # recording folder ("Water_C1_frames"): the unit train/validation splits hold out
    ids: np.ndarray           # (n,) board tag ids
    obj: np.ndarray           # (4n, 3) board points, mm, z = 0
    img: np.ndarray           # (4n, 2) detected corners, px
    distance_mm: float = np.nan   # camera to board centre, from the pipeline's own calibration

    def subset(self, ids):
        """This frame reduced to the given tag ids (in this frame's order)."""
        keep = np.isin(self.ids, list(ids))
        k4 = np.repeat(keep, 4)
        return Frame(self.name, self.group, self.ids[keep], self.obj[k4], self.img[k4], self.distance_mm)


@dataclass
class PipelineRun:
    yaml_path: str
    corners_path: str
    image_size: tuple                         # (width, height)
    preset: str
    channel: str
    min_common_tags: int                      # stereo pairs need this many tags seen by both cameras
    frames: dict = field(default_factory=dict)        # camera -> [Frame], sorted by name
    pipeline_calibration: dict = field(default_factory=dict)   # camera -> (K, D) as Calibration.py fitted

    def groups(self):
        return sorted({f.group for frames in self.frames.values() for f in frames})

    def stereo_pairs(self, left_names=None):
        """[(left Frame, right Frame)] restricted to the tags both saw; same pairing as Calibration.py."""
        right = {f.name: f for f in self.frames["right"]}
        pairs = []
        for lf in self.frames["left"]:
            if left_names is not None and lf.name not in left_names:
                continue
            rf = right.get(lf.name.replace("_Left_", "_Right_"))
            if rf is None:
                continue
            shared = sorted(set(lf.ids.tolist()) & set(rf.ids.tolist()))
            if len(shared) >= self.min_common_tags:
                pairs.append((sorted_by_id(lf.subset(shared)), sorted_by_id(rf.subset(shared))))
        return pairs


def sorted_by_id(frame):
    """Frame with its tags in id order, so two cameras' points line up one to one."""
    order = np.argsort(frame.ids, kind="stable")
    k4 = (order[:, None] * 4 + np.arange(4)).ravel()
    return Frame(frame.name, frame.group, frame.ids[order], frame.obj[k4], frame.img[k4], frame.distance_mm)


def board_points():
    """(70, 4, 3) board corners per tag id, mm, in the corner order the detector reports."""
    return np.array(Calibration.build_board().getObjPoints(), np.float64).reshape(Calibration.N_TAGS, 4, 3)


def board_centre():
    obj = board_points().reshape(-1, 3)
    return 0.5 * (obj.min(axis=0) + obj.max(axis=0))


def _strings(node):
    return [node.at(i).string() for i in range(node.size())] if not node.empty() else []


def load(yaml_path):
    """Read a Calibration.py run."""
    corners_path = yaml_path.replace("_stereo.yaml", "_corners.json")
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"cannot read {yaml_path}")
    with open(corners_path, encoding="utf-8") as fh:
        corners = json.load(fh)

    image = fs.getNode("image")
    detection = fs.getNode("detection")
    run = PipelineRun(yaml_path=yaml_path, corners_path=corners_path,
                      image_size=(int(image.getNode("width_px").real()), int(image.getNode("height_px").real())),
                      preset=detection.getNode("detector_preset").string() or corners.get("preset", ""),
                      channel=detection.getNode("channel").string() or corners.get("channel", ""),
                      min_common_tags=int(fs.getNode("stereo").getNode("min_common_tags").real() or 10))
    tags = board_points()
    centre = board_centre()

    for camera in CAMERAS:
        node = fs.getNode(f"{camera}_camera")
        K = node.getNode("camera_matrix").mat()
        D = node.getNode("distortion_coefficients").mat()
        run.pipeline_calibration[camera] = (K, D)
        rejected = {(s.split("|")[0], int(s.split("|")[1]))
                    for s in _strings(fs.getNode("outlier_rejection").getNode(camera).getNode("rejected_tags"))}
        frames = []
        for name in _strings(node.getNode("images")):
            det = corners["cameras"][camera][name]
            ids = det["ids"]
            keep = [k for k, i in enumerate(ids)
                    if 0 <= i < Calibration.N_TAGS and ids.count(i) == 1 and (name, i) not in rejected]
            kept_ids = np.array([ids[k] for k in keep], int)
            f = Frame(name=name, group=name.split("/")[0], ids=kept_ids,
                      obj=tags[kept_ids].reshape(-1, 3),
                      img=np.array([det["corners"][k] for k in keep], np.float64).reshape(-1, 2))
            ok, rvec, tvec = cv2.solvePnP(f.obj, f.img, K, D)
            if ok:
                f.distance_mm = float(np.linalg.norm(cv2.Rodrigues(rvec)[0] @ centre + tvec.ravel()))
            frames.append(f)
        run.frames[camera] = sorted(frames, key=lambda f: f.name)
    fs.release()
    return run
