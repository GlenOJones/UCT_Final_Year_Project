"""UWARUco: the underwater square-marker detector of Cejka et al., implemented for this project.

    J. Cejka, F. Bruno, D. Skarlatos, F. Liarokapis, "Detecting Square Markers in Underwater
    Environments", Remote Sensing 11(4):459, 2019.  doi:10.3390/rs11040459

Why this is a re-implementation rather than a setting on cv2.aruco.ArucoDetector: the paper's central
change is an ORDERING change, not a parameter change. Standard ARUco thresholds the image three times,
finds contours in each, merges the contours, and only then identifies each marker by thresholding the
ORIGINAL grey-scale image again with Otsu's method. UWARUco identifies the marker inside each
thresholding pass, from that pass's own binary image, and merges markers instead of contours. OpenCV
exposes no parameter for that, so the workflow is rebuilt here on top of OpenCV primitives
(adaptiveThreshold, findContours, warpPerspective, Dictionary.identify).

Two detectors live here, both with the same interface as cv2.aruco.ArucoDetector:

    detector.detectMarkers(gray) -> (corners, ids, rejected)

with corners a tuple of (1, 4, 2) float32 arrays, ids an (N, 1) int32 array or None, and rejected a
list of the quadrilaterals that were found but not identified. That is deliberate: Calibration.py
can use one of these in place of its ArucoDetector without knowing which it has, so the calibration
that comes out is comparable with the ArUco one line for line.

    Base    (Section 2.3, Figure 2b) threshold at three window sizes with the threshold constant at
            zero, find contours, identify inside each pass, merge markers.
    Masked  (Figure 2c) the same, plus a brightness/noise mask (Algorithm 1) applied to each
            thresholded image and a 3x3 median filter, to cut the number of noise contours that
            leaving the threshold constant at zero creates.

Corner positions from either detector come from a binarised image and are no better than ArUco's.
They are not meant to be: Calibration.py refines every corner with cv2.cornerSubPix on the original
grey-scale image afterwards, so what a detector contributes to the calibration is WHICH tags it finds
and roughly where, not the final sub-pixel corner.

README.md records what this detector does on this project's footage, and why one of the paper's
two headline settings transfers to it and the other does not.

Run this file to compare the detectors on a folder of frames, without calibrating:

    src/venv/bin/python src/UWARUco/uwaruco.py data/Calibration/18_Sep/Rigframes

Or calibrate with it, which is the comparison that counts:

    src/venv/bin/python src/01_Calibration/Calibration.py --detector uwaruco --preset tuned
    src/venv/bin/python src/01_Calibration/compare_calibrations.py --run wide uwaruco:tuned
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_Calibration"))
from detector_config import parameters, validate  # noqa: E402

PRESETS_FILE = os.path.join(HERE, "uwaruco_presets.json")

# The tags on the board and how they are printed are described once, in the calibration module's
# detector_presets.json. Reading them from there rather than restating them keeps a change to the
# target (a different tag family, a different border) from having to be made in two files, one of
# which would then be silently out of date.
BOARD_PRESETS_FILE = os.path.join(os.path.dirname(HERE), "01_Calibration", "detector_presets.json")


def board_params():
    """markerBorderBits and errorCorrectionRate as detector_presets.json states them for this board."""
    with open(BOARD_PRESETS_FILE, encoding="utf-8") as fh:
        return parameters(validate(json.load(fh), BOARD_PRESETS_FILE)["board_params"])


# ====== PARAMETERS ======
@dataclass
class Params:
    """Everything the detector can be tuned by, with the paper's values as the defaults.

    Names follow cv2.aruco.DetectorParameters wherever OpenCV has an equivalent, so a value can be
    carried straight across from a detector_presets.json preset and mean the same thing.
    """
    # --- Threshold step. The paper's whole first contribution is on these two lines. ---
    # Window sizes in pixels. The paper found 10, 20 and 40 optimal on 1920x1080 footage; these
    # frames are 1280x720, so a preset may scale them (see uwaruco_presets.json).
    adaptiveThreshWinSizes: tuple = (11, 21, 41)
    # The constant subtracted from the local mean. ARUco defaults to 7 to suppress noise; the paper
    # sets it to 0 because subtracting it is what breaks a low-contrast marker's border into pieces.
    adaptiveThreshConstant: float = 0.0

    # --- Mask step (Masked version only), Algorithm 1. ---
    maskBlock: int = 4          # block of pixels reduced to one min and one max
    maskBlur: int = 3           # min/max spread over this neighbourhood of blocks, to keep borders whole
    maskFeedbackRate: float = 0.8   # thresholds are 80% of the minimum seen on a detected contour
    medianFilter: bool = True   # 3x3 median on the masked binary image, to drop pixel-sized objects

    # --- Contour step. Same meaning and the same defaults as cv2.aruco.DetectorParameters. ---
    minMarkerPerimeterRate: float = 0.03
    maxMarkerPerimeterRate: float = 4.0
    polygonalApproxAccuracyRate: float = 0.03
    minCornerDistanceRate: float = 0.05
    minDistanceToBorder: int = 3

    # --- Identify step. ---
    # Where the marker's code is read from. "threshold" is the paper's answer: the binary image the
    # candidate was found in, so a marker is identified on the same evidence that found it.
    # "otsu" keeps the paper's REORDERING (identify inside each pass, merge markers rather than
    # contours) but reads the bits ARUco's way, by warping the original grey image and thresholding
    # it with Otsu's method. That is a deviation, and on this footage it is the one that pays: see
    # README.md. The paper's argument for "threshold" is that Otsu is the step image-improving
    # filters were really helping, which is an argument about a step being redundant, not about it
    # being wrong, and these frames are low-contrast rather than dark.
    identifyFrom: str = "threshold"
    markerBorderBits: int = 1
    errorCorrectionRate: float = 0.6
    maxErroneousBitsInBorderRate: float = 0.35
    perspectiveRemovePixelPerCell: int = 8
    perspectiveRemoveIgnoredMarginPerCell: float = 0.13

    # --- Merge Markers step. ---
    # Two detections of the same id whose corners are within this fraction of the marker's perimeter
    # are the same marker seen in two thresholding passes.
    minMarkerDistanceRate: float = 0.05
    # What to do with those duplicates. "largest" keeps the biggest, which is what ARUco does when
    # it merges contours. "mean" averages the passes instead, on the theory that they are near
    # independent estimates of the same corner. They are not, and it is measurably worse: see
    # README.md. Kept only so that result stays reproducible.
    mergeCorners: str = "largest"

    def replace(self, **overrides):
        """A copy with some fields changed, raising on a name this class does not define.

        setattr on a plain object would accept a typo and produce a detector that quietly ignored
        the setting, which is exactly the failure detector_config.py exists to prevent.
        """
        known = {f for f in self.__dataclass_fields__}
        for key in overrides:
            if key not in known:
                raise SystemExit(f"unknown UWARUco parameter {key!r}; "
                                 f"known parameters: {', '.join(sorted(known))}")
        merged = Params(**{**{f: getattr(self, f) for f in known}, **overrides})
        if merged.mergeCorners not in ("largest", "mean"):
            raise SystemExit(f"mergeCorners is {merged.mergeCorners!r}; use \"largest\" (ARUco's "
                             f"rule) or \"mean\" (average the passes that agree)")
        if merged.identifyFrom not in ("threshold", "otsu"):
            raise SystemExit(f"identifyFrom is {merged.identifyFrom!r}; use \"threshold\" (the "
                             f"paper) or \"otsu\" (ARUco's bit source, kept in the paper's workflow)")
        return merged

    def describe(self):
        """One-line summary, for logs and for the calibration YAML."""
        return ", ".join(f"{f}={getattr(self, f)}" for f in sorted(self.__dataclass_fields__))


# ====== THRESHOLD STEP ======
def threshold(gray, win_size, constant):
    """ARUco's thresholding: adaptive mean, inverted, so marker BLACK becomes 255.

    Everything downstream assumes that polarity: contours are traced around the black parts of the
    marker, and a bit reads as white when the binary image is 0 there.
    """
    win_size = int(win_size) | 1        # adaptiveThreshold requires an odd window
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, max(3, win_size), constant)


# ====== MASK STEP (Algorithm 1) ======
@dataclass
class Mask:
    """The output of Compute Mask: the mask itself, and the images the next frame's thresholds
    are read off (Algorithm 2). maxs and diffs are kept at full image size so that a marker contour,
    which is in image pixels, can index them directly."""
    mask: np.ndarray     # uint8, 0 or 255, full image size; None when masking is disabled
    maxs: np.ndarray     # local maximum intensity, full image size
    diffs: np.ndarray    # local maximum minus local minimum, i.e. edge strength, full image size


def block_extremes(gray, block):
    """Minimum and maximum intensity of every block x block tile, as two images block times smaller."""
    h, w = gray.shape
    rows, cols = h // block, w // block
    tiles = gray[:rows * block, :cols * block].reshape(rows, block, cols, block)
    return tiles.min(axis=(1, 3)), tiles.max(axis=(1, 3))


def compute_mask(gray, brightness_threshold, noise_threshold, p):
    """Algorithm 1: keep only the parts of the image that are both very bright and contain an edge.

    The brightness mask finds the white areas of markers, which are the brightest thing in an
    underwater frame; the noise mask finds strong local contrast, which is a marker's border. A
    marker needs both, so the two are ANDed. The 3x3 spread of the minima and maxima is not
    cosmetic: without it a block that lies wholly inside a white area contains no edge and a block
    that lies wholly inside the black border is not bright, so the mask would cut the border apart
    exactly where the contour has to stay whole.

    Thresholds of zero mean "no evidence yet" (the first frame, or a frame where nothing was found),
    and the paper disables masking in that case; mask is then None.
    """
    mins, maxs = block_extremes(gray, p.maskBlock)
    kernel = np.ones((p.maskBlur, p.maskBlur), np.uint8)
    mins = cv2.erode(mins, kernel)      # minimum over the 3x3 neighbourhood of blocks
    maxs = cv2.dilate(maxs, kernel)     # maximum over the same
    diffs = maxs.astype(np.int16) - mins.astype(np.int16)

    full = (gray.shape[1], gray.shape[0])
    maxs_full = cv2.resize(maxs, full, interpolation=cv2.INTER_NEAREST)
    diffs_full = cv2.resize(diffs.astype(np.int16), full, interpolation=cv2.INTER_NEAREST)

    if brightness_threshold <= 0 and noise_threshold <= 0:
        return Mask(None, maxs_full, diffs_full)

    mask = ((maxs_full >= brightness_threshold) & (diffs_full >= noise_threshold))
    return Mask(mask.astype(np.uint8) * 255, maxs_full, diffs_full)


def feedback_thresholds(mask, corners, p):
    """Algorithm 2: the thresholds for the next frame, read off the markers found in this one.

    The dimmest and least contrasty pixel on any marker border is the weakest evidence the detector
    still managed to use, so anything below it can be masked away. The 20% margin absorbs a change
    in lighting between one frame and the next; with nothing found there is no evidence at all and
    the thresholds go back to zero, which turns masking off.
    """
    if not len(corners):
        return 0.0, 0.0
    outline = np.zeros(mask.maxs.shape, np.uint8)
    cv2.polylines(outline, [c.reshape(4, 2).astype(np.int32) for c in corners], True, 255, 1)
    on_contour = outline > 0
    if not on_contour.any():
        return 0.0, 0.0
    return (p.maskFeedbackRate * float(mask.maxs[on_contour].min()),
            p.maskFeedbackRate * float(mask.diffs[on_contour].min()))


def median_filter(binary):
    """3x3 median of a binary image, which removes pixel-sized objects and closes pixel-sized holes.

    The paper's shortcut: on a binary image the median is the mean thresholded at half, so a 3x3
    box blur and a compare does the same work as medianBlur and is what it describes.
    """
    return ((cv2.blur(binary, (3, 3)) >= 128).astype(np.uint8)) * 255


# ====== CONTOUR STEP ======
def find_candidates(binary, p):
    """Quadrilateral candidates in a binary image, corners counter-clockwise.

    This mirrors ARUco's own contour filtering (OpenCV's _findMarkerContours) so that the only thing
    differing between this detector and ArucoDetector is what the paper changes. The cheap length
    test comes first on purpose: with the threshold constant at zero the image carries thousands of
    noise contours, and approxPolyDP on all of them is what makes the Base version slow.
    """
    h, w = binary.shape
    min_perimeter = p.minMarkerPerimeterRate * max(h, w)
    max_perimeter = p.maxMarkerPerimeterRate * max(h, w)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        n = len(contour)
        if n < min_perimeter or n > max_perimeter:
            continue
        approx = cv2.approxPolyDP(contour, n * p.polygonalApproxAccuracyRate, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        # Corners too close together are a degenerate quad rather than a marker seen edge-on.
        gaps = np.linalg.norm(quad[:, None, :] - quad[None, :, :], axis=2)
        if gaps[np.triu_indices(4, 1)].min() < n * p.minCornerDistanceRate:
            continue
        # A marker touching the image border is cut off, so its corners are not where they look.
        if (quad.min() < p.minDistanceToBorder
                or quad[:, 0].max() >= w - p.minDistanceToBorder
                or quad[:, 1].max() >= h - p.minDistanceToBorder):
            continue

        # Counter-clockwise, which is the order the bit extraction and the dictionary assume.
        (dx1, dy1), (dx2, dy2) = quad[1] - quad[0], quad[2] - quad[0]
        if dx1 * dy2 - dy1 * dx2 < 0:
            quad[[1, 3]] = quad[[3, 1]]
        candidates.append(quad)
    return candidates


# ====== IDENTIFY STEP ======
def extract_bits(binary, gray, quad, p):
    """Read the marker's code, warped square, from whichever image p.identifyFrom names.

    This is the paper's second change. ARUco warps the ORIGINAL grey-scale image here and thresholds
    it again with Otsu's method, which is a second, independent decision about what is black and
    what is white, and it is the one that image-improving filters were really helping. UWARUco reads
    the bits from the binary image the candidate was found in, so a marker that survived the
    threshold is identified on the same evidence that found it. Both are available here because on
    this project's footage they do not come out the same way round; see Params.identifyFrom.
    """
    size = (p.markerBorderBits * 2 + 6) * p.perspectiveRemovePixelPerCell
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    source = binary if p.identifyFrom == "threshold" else gray
    warped = cv2.warpPerspective(source, cv2.getPerspectiveTransform(quad, target), (size, size),
                                 flags=cv2.INTER_NEAREST)
    if p.identifyFrom == "otsu":
        # Otsu over this marker alone, so its own contrast sets the black/white split rather than
        # the whole frame's. This is the step the paper removes.
        warped = cv2.threshold(warped, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cell = p.perspectiveRemovePixelPerCell
    margin = int(p.perspectiveRemoveIgnoredMarginPerCell * cell)
    cells = warped.reshape(size // cell, cell, size // cell, cell)
    inner = cells[:, margin:cell - margin, :, margin:cell - margin].mean(axis=(1, 3))
    # The thresholded image is inverted (marker black is 255), the Otsu one is not, so a mostly-set
    # cell is a BLACK bit in the first case and a WHITE bit in the second.
    bits = inner < 128 if p.identifyFrom == "threshold" else inner >= 128
    return bits.astype(np.uint8)


def border_errors(bits, border_bits):
    """Border cells that came out white. A real marker's border is black all the way round."""
    inner = np.zeros_like(bits, bool)
    inner[border_bits:bits.shape[0] - border_bits, border_bits:bits.shape[1] - border_bits] = True
    return int(bits[~inner].sum())


def identify(binary, gray, quad, dictionary, p):
    """(id, corners) for a candidate, or None if it is not a marker of this dictionary.

    The corners come back rotated to the dictionary's canonical orientation, which is what makes an
    id mean the same four object points in every image.
    """
    bits = extract_bits(binary, gray, quad, p)
    n_bits = bits.shape[0] * bits.shape[1]
    if border_errors(bits, p.markerBorderBits) > n_bits * p.maxErroneousBitsInBorderRate:
        return None

    inner = bits[p.markerBorderBits:bits.shape[0] - p.markerBorderBits,
                 p.markerBorderBits:bits.shape[1] - p.markerBorderBits]
    found, marker_id, rotation = dictionary.identify(np.ascontiguousarray(inner),
                                                     p.errorCorrectionRate)
    if not found:
        return None
    return int(marker_id), np.roll(quad, -((4 - rotation) % 4), axis=0)


# ====== MERGE MARKERS STEP ======
def merge_markers(found, p):
    """One detection per marker, out of up to one per thresholding pass.

    The three passes see the same marker at three window sizes, so the same id arrives up to three
    times with corners a fraction of a pixel apart. ARUco keeps the larger of two near-duplicate
    candidates; the same rule is used here, and two detections of one id that are NOT near each
    other are both kept, because one of them is a misidentification somewhere else in the frame and
    dropping the wrong one silently would cost a real tag.
    """
    # Each entry is [id, best quad, its perimeter, every quad in this group], so that either merge
    # rule can be applied at the end without detecting twice.
    kept = []
    for marker_id, quad in found:
        perimeter = cv2.arcLength(quad.reshape(4, 1, 2).astype(np.float32), True)
        for group in kept:
            if group[0] != marker_id:
                continue
            if np.abs(quad - group[1]).sum() < p.minMarkerDistanceRate * min(perimeter, group[2]) * 4:
                group[3].append(quad)
                if perimeter > group[2]:
                    group[1], group[2] = quad, perimeter
                break
        else:
            kept.append([marker_id, quad, perimeter, [quad]])

    if p.mergeCorners == "mean":
        return [(g[0], np.mean(g[3], axis=0).astype(np.float32)) for g in kept]
    return [(g[0], g[1]) for g in kept]


def as_opencv(markers, rejected):
    """The (corners, ids, rejected) triple that cv2.aruco.ArucoDetector.detectMarkers returns."""
    if not markers:
        return (), None, [r.reshape(1, 4, 2).astype(np.float32) for r in rejected]
    markers = sorted(markers, key=lambda m: m[0])
    corners = tuple(q.reshape(1, 4, 2).astype(np.float32) for _, q in markers)
    ids = np.array([[i] for i, _ in markers], np.int32)
    return corners, ids, [r.reshape(1, 4, 2).astype(np.float32) for r in rejected]


# ====== DETECTORS ======
class BaseDetector:
    """UWARUco, Base version (Figure 2b).

    Threshold at each window size with nothing subtracted from the local mean, find contours,
    identify the markers in that same binary image, then merge the markers from all passes.
    """

    def __init__(self, dictionary, params=None):
        self.dictionary = dictionary
        self.p = params or Params()

    def binaries(self, gray):
        """The thresholded image of each pass. Overridden by the Masked version."""
        return [threshold(gray, w, self.p.adaptiveThreshConstant)
                for w in self.p.adaptiveThreshWinSizes]

    def detectMarkers(self, gray):
        found, rejected = [], []
        for binary in self.binaries(gray):
            for quad in find_candidates(binary, self.p):
                marker = identify(binary, gray, quad, self.dictionary, self.p)
                if marker is None:
                    rejected.append(quad)
                else:
                    found.append(marker)
        return as_opencv(merge_markers(found, self.p), rejected)


class MaskedDetector(BaseDetector):
    """UWARUco, Masked version (Figure 2c).

    The paper computes the mask for frame n from the markers found in frame n-1, because it is
    detecting in a video. Calibration frames are not a video: they are stills chosen to be far apart
    in pose, so carrying a threshold from one to the next would be carrying it from an unrelated
    view. Each image is therefore given its own feedback: a first, unmasked pass finds what it can,
    Algorithm 2 reads the thresholds off those markers, and a second pass re-runs with the mask.
    That is the same loop the paper describes, closed within one image instead of across two, and it
    is the one deviation from the paper that changes what the algorithm does rather than how it is
    written.  Pass feedback="sequential" to get the paper's behaviour instead, which is only
    meaningful on frames that really are consecutive.
    """

    def __init__(self, dictionary, params=None, feedback="self"):
        super().__init__(dictionary, params)
        if feedback not in ("self", "sequential", "none"):
            raise SystemExit(f"unknown feedback mode {feedback!r}; use self, sequential or none")
        self.feedback = feedback
        self.thresholds = (0.0, 0.0)    # brightness, noise; zero disables masking

    def masked_binaries(self, gray, thresholds):
        """The thresholded images with the mask and the median filter applied, plus the mask itself."""
        mask = compute_mask(gray, thresholds[0], thresholds[1], self.p)
        binaries = []
        for win in self.p.adaptiveThreshWinSizes:
            binary = threshold(gray, win, self.p.adaptiveThreshConstant)
            if mask.mask is not None:
                binary = cv2.bitwise_and(binary, mask.mask)
                if self.p.medianFilter:
                    binary = median_filter(binary)
            binaries.append(binary)
        return binaries, mask

    def detect_once(self, gray, thresholds):
        binaries, mask = self.masked_binaries(gray, thresholds)
        found, rejected = [], []
        for binary in binaries:
            for quad in find_candidates(binary, self.p):
                marker = identify(binary, gray, quad, self.dictionary, self.p)
                (rejected if marker is None else found).append(quad if marker is None else marker)
        return merge_markers(found, self.p), rejected, mask

    def detectMarkers(self, gray):
        markers, rejected, mask = self.detect_once(gray, self.thresholds)
        thresholds = feedback_thresholds(mask, [q for _, q in markers], self.p)

        if self.feedback == "self" and thresholds != (0.0, 0.0):
            # Second pass, now that this image has told us how dim and how soft its own markers are.
            # Keep whichever pass found more: masking can only remove evidence, so a mask derived
            # from a lucky first pass must not be allowed to lose tags that pass already had.
            masked_markers, masked_rejected, _ = self.detect_once(gray, thresholds)
            if len(masked_markers) > len(markers):
                markers, rejected = masked_markers, masked_rejected
        elif self.feedback == "sequential":
            self.thresholds = thresholds

        return as_opencv(markers, rejected)


# ====== PRESETS ======
@dataclass
class Presets:
    """The contents of uwaruco_presets.json: named tunings, in data rather than in code.

    Same arrangement as 01_Calibration/detector_presets.json, for the same reason: trying a tuning
    should be an edit to a data file that a run records in its output, not an edit to a script.
    """
    path: str
    default: str
    presets: dict = field(default_factory=dict)

    def params(self, name):
        if name not in self.presets:
            raise SystemExit(f"unknown UWARUco preset {name!r}; "
                             f"{self.path} defines: {', '.join(self.presets)}")
        tuning = {k: v for k, v in self.presets[name].items() if not k.startswith("_")}
        return Params(**board_params()).replace(**tuning)

    def version(self, name):
        """"base" or "masked": which of the paper's two workflows this preset asks for."""
        return self.presets[name].get("_version", "masked") if name in self.presets else "masked"

    def make_detector(self, dictionary, name):
        params = self.params(name)
        if self.version(name) == "base":
            return BaseDetector(dictionary, params)
        return MaskedDetector(dictionary, params, feedback=self.presets[name].get("_feedback", "self"))

    def describe(self, name):
        return f"uwaruco/{self.version(name)}: {self.params(name).describe()}"


def load_presets(path=PRESETS_FILE):
    """Read and check uwaruco_presets.json.

    The checking is deliberately strict about parameter NAMES (Params.replace raises on an unknown
    one) because a preset that sets a misspelled key would otherwise look like it was doing
    something while changing nothing at all.
    """
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    for key in ("default", "presets"):
        if key not in config:
            raise SystemExit(f"{path}: missing required key {key!r}")

    presets = {}
    for name, body in config["presets"].items():
        if not isinstance(body, dict):
            raise SystemExit(f"{path}: preset {name!r} must be a JSON object")
        # "_version" and "_feedback" pick the workflow; other "_" keys are documentation.
        body = {k: v for k, v in body.items() if not k.startswith("_") or k in ("_version", "_feedback")}
        if body.get("_version", "masked") not in ("base", "masked"):
            raise SystemExit(f"{path}: preset {name!r} has _version {body['_version']!r}; "
                             f"use \"base\" or \"masked\"")
        presets[name] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in body.items()}

    result = Presets(path=path, default=config["default"], presets=presets)
    if result.default not in presets:
        raise SystemExit(f"{path}: default {result.default!r} is not one of {', '.join(presets)}")
    for name in presets:                    # fail here, not on the first frame of a long run
        result.params(name)
    return result


def make_detector(dictionary, preset=None, path=PRESETS_FILE):
    """A UWARUco detector for one named preset. The entry point Calibration.py uses."""
    presets = load_presets(path)
    return presets.make_detector(dictionary, preset or presets.default)


# ====== COMPARE DETECTORS ON A FOLDER ======
def aruco_detector(dictionary, preset="wide"):
    """The project's existing ArucoDetector, as the thing UWARUco has to beat."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_Calibration"))
    import Calibration                      # noqa: E402  (imported here so the module stays optional)
    return Calibration.load_presets().make_detector(preset)


def count_tags(detector, files):
    """Tags found per image, and how long the whole folder took."""
    counts = []
    start = cv2.getTickCount()
    for path in files:
        _, ids, _ = detector.detectMarkers(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        counts.append(0 if ids is None else len(ids))
    seconds = (cv2.getTickCount() - start) / cv2.getTickFrequency()
    counts = np.array(counts)
    return counts, seconds


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare UWARUco with ArUco on a folder of frames.")
    ap.add_argument("frames_dir", nargs="?",
                    default=os.path.join(PROJECT_ROOT, "data/Calibration/18_Sep/Rigframes"),
                    help="folder searched recursively for PNGs")
    ap.add_argument("--min-tags", type=int, default=10,
                    help="tags an image needs to be usable for calibration (default 10)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N images (0 = all)")
    ap.add_argument("--aruco-preset", default="wide",
                    help="ArUco preset from 01_Calibration/detector_presets.json to compare against")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.frames_dir, "**", "*.png"), recursive=True))
    if not files:
        raise SystemExit(f"no PNGs under {args.frames_dir}")
    if args.limit:
        files = files[:args.limit]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    presets = load_presets()
    detectors = {f"aruco/{args.aruco_preset}": aruco_detector(dictionary, args.aruco_preset)}
    for name in presets.presets:
        detectors[f"uwaruco/{name}"] = presets.make_detector(dictionary, name)

    print(f"{len(files)} images under {os.path.relpath(args.frames_dir, PROJECT_ROOT)}; "
          f"usable = {args.min_tags} or more tags")
    print(f"{'detector':24s} {'usable':>7s} {'tags':>8s} {'median':>7s} {'s/image':>8s}")
    for name, detector in detectors.items():
        counts, seconds = count_tags(detector, files)
        print(f"{name:24s} {int((counts >= args.min_tags).sum()):7d} {int(counts.sum()):8d} "
              f"{int(np.median(counts)):7d} {seconds / len(files):8.2f}")


if __name__ == "__main__":
    main()
