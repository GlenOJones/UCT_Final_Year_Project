"""Search detector_presets.json parameter space for the settings that find the most tags.

Answers one question: which cv2.aruco.DetectorParameters find the most AprilGrid tags on a set of
frames? Nothing here re-implements detection. The script builds a Presets object in memory and hands
it to Calibration.py's own make_detector() and detect_tags(), so what is measured is exactly what a
calibration run would see, and Calibration.py itself is untouched.

The search is coordinate descent from a starting preset: one axis at a time, every candidate value
tried with the rest held at the current best, keeping whichever value finds the most tags, repeated
until a round changes nothing. That is far cheaper than the full grid (thousands of combinations)
and, because the axes mostly act independently, it lands in the same place.

Score is the number of DISTINCT board tags found, summed over the images: per image, how many of
the ids 0..69 were seen at least once. Counting raw detections instead is what an earlier version
of this script did, and the search exploited it - minMarkerDistanceRate=0 turns off the merging of
overlapping candidates, so one physical tag is reported many times over, and the score went from
4439 to 52822 on 208 images that can only carry 208 * 70 = 14560 tags. A detection that repeats an
id already found in the same image adds nothing to a calibration, so it is not counted.

Two kinds of detection are counted separately as evidence that the decoder has been loosened too
far, and a candidate producing more than --max-bad-rate of them is never selected however many tags
it finds:
    duplicates  the same id twice in one image; at least one of the two is misplaced
    false ids   an id outside 0..69, which the 36h11 dictionary contains but this board cannot
Neither corrupts the fit directly (matchImagePoints keeps the first of a repeated id and ignores
ids the board lacks), but both rise exactly when the decoder starts misreading tags it did see -
a tag read as the wrong id lands in the fit at the wrong board position and is invisible otherwise.

More tags is still not the same as a better calibration: run Calibration.py with the winning preset
and compare the reprojection error against the preset you started from.

Every evaluation is cached to results/analysis/detector_sweep/<frames>_evaluations.jsonl as it finishes, so
a sweep that is interrupted, or one re-run after SWEEP_AXES grows, only computes what is new. Pass
--fresh to throw that cache away.

Use (from the project root):
    src/venv/bin/python src/01_Calibration/sweep_detector.py
    src/venv/bin/python src/01_Calibration/sweep_detector.py --start opencv_default --save-preset best
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Calibration as calib  # noqa: E402
from detector_config import validate  # noqa: E402

PROJECT_ROOT = calib.PROJECT_ROOT

# Candidate values per axis, in the order the axes are searched: thresholding first (it decides
# which shapes exist at all), then contour filtering, then bit decoding. Each list contains the
# value 'wide' already uses, so a round can always choose to change nothing.
#
# The board parameters markerBorderBits and errorCorrectionRate are deliberately absent. They
# describe the printed Kalibr target rather than the tuning, detector_presets.json sets them once
# for every preset, and letting a search move them is how a preset ends up finding zero tags.
SWEEP_AXES = {
    # Size of the local window used to threshold the image, swept from min to max in steps. Small
    # windows suit tags that fill the frame, large ones distant tags; the sweep is the cost.
    "adaptiveThreshWinSizeMin": [3, 5, 7, 9, 13],
    "adaptiveThreshWinSizeMax": [23, 33, 43, 53, 63, 83],
    "adaptiveThreshWinSizeStep": [2, 4, 5, 8, 10, 14],
    # Subtracted from the local mean. Do not expect 0 to win: these tags are low-contrast rather
    # than dark, and 0 collapses detection (see the note in detector_presets.json).
    "adaptiveThreshConstant": [3, 5, 7, 9, 12],
    # A candidate whose greyscale spread is below this is thresholded by Otsu instead. The default
    # 5.0 throws away low-contrast underwater tags.
    "minOtsuStdDev": [0.0, 1.0, 2.0, 3.0, 5.0],
    # Contour perimeter as a fraction of the larger image side: the floor sets how small a tag can
    # be before it is ignored, which matters for the far end of the board.
    #
    # 0.005 and 0.01 are deliberately not offered. Once the thresholding axes have widened (a
    # dense sweep to winSizeMax=83), a floor that low admits an enormous pool of tiny contours:
    # on the 18 Sep set one evaluation at 0.005 took 2054 s against 18 s at 0.03, and was then
    # refused anyway for a 1.16% bad rate. The whole axis was worth +3 tags for ~37 minutes.
    "minMarkerPerimeterRate": [0.02, 0.03],
    "maxMarkerPerimeterRate": [4.0, 8.0],
    # How far a contour may sit from a true quadrilateral. Blur and water distortion round the
    # corners off, so a looser value can recover tags that are genuinely there.
    "polygonalApproxAccuracyRate": [0.01, 0.03, 0.05, 0.08],
    "minCornerDistanceRate": [0.0, 0.05, 0.1],
    "minMarkerDistanceRate": [0.0, 0.05, 0.1],
    # Resolution the tag is un-warped to before its bits are read, and how much of each cell's edge
    # is ignored when deciding the bit. More pixels per cell helps small, soft tags.
    "perspectiveRemovePixelPerCell": [4, 8, 12, 16],
    "perspectiveRemoveIgnoredMarginPerCell": [0.13, 0.2, 0.33],
    # Fraction of border bits allowed to be wrong before the candidate is thrown out.
    "maxErroneousBitsInBorderRate": [0.35, 0.5, 0.75],
}

# The board carries ids 0..69; anything else the 36h11 dictionary reports is a false positive.
BOARD_IDS = range(calib.N_TAGS)


@dataclass(frozen=True)
class Score:
    """What one set of detector settings found across both cameras."""
    board_tags: int       # distinct board ids per image, summed - the thing being maximised
    duplicate_tags: int   # detections repeating an id already found in the same image
    false_tags: int       # detections with an id outside 0..69, i.e. certainly not on this board
    usable_images: int    # images with at least min_tags distinct board tags
    detections: int       # every detection, whatever its id: the denominator of the bad rate
    seconds: float

    @property
    def bad_tags(self):
        """Detections that are certainly wrong: a repeated id, or an id the board does not carry."""
        return self.duplicate_tags + self.false_tags

    @property
    def bad_rate(self):
        """Wrong detections as a fraction of all of them."""
        return self.bad_tags / self.detections if self.detections else 0.0

    @property
    def key(self):
        """Sort key: most board tags, then fewest wrong detections, then most usable images."""
        return (self.board_tags, -self.bad_tags, self.usable_images)

    def __str__(self):
        return (f"{self.board_tags:6d} tags  {self.usable_images:4d} images"
                f"  {self.bad_tags:5d} bad ({self.bad_rate:5.2%})  {self.seconds:5.1f}s")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames-dir", default=os.path.join(PROJECT_ROOT, "data/Calibration/18_Sep/Rigframes"),
                    help="one folder per recording, each with left/ and right/ (default 18_Sep/Rigframes)")
    ap.add_argument("--start", default=None,
                    help="preset in detector_presets.json to start the search from "
                         "(default: that file's own default)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="most passes over the axes; the search stops early when one changes "
                         "nothing (default 4)")
    ap.add_argument("--min-tags", type=int, default=10,
                    help="tags an image needs to count as usable, matching Calibration.py "
                         "(default 10)")
    ap.add_argument("--max-bad-rate", type=float, default=0.005,
                    help="most repeated or off-board ids a candidate may produce, as a fraction of "
                         "its detections, before it is refused however many tags it finds "
                         "(default 0.005)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore and overwrite the cached evaluations from earlier runs, "
                         "recomputing every candidate from scratch")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results/analysis/detector_sweep"),
                    help="where the JSON record of the search goes")
    ap.add_argument("--save-preset", metavar="NAME",
                    help="add the winning settings to detector_presets.json under this name. "
                         "The file is rewritten by json.dump, so its formatting changes.")
    return ap.parse_args(argv)


def make_run(frames_dir, settings, board_params, image_size, board, min_tags):
    """A Calibration.Run configured with one candidate's settings.

    The single-preset Presets built here is what keeps the search honest: make_detector() applies
    the same board_params overlay and the same forced cornerRefinementMethod that a real run gets.
    """
    presets = calib.Presets(path="<sweep>", default="candidate",
                            board_params=board_params, presets={"candidate": dict(settings)})
    return calib.Run(frames_dir=frames_dir, preset="candidate", out_dir="<unused>",
                     min_tags=0,                 # count every image; usable is judged below
                     min_common_tags=min_tags,   # unused here, no stereo fit is done
                     save_detections=False,      # writing 208 images per candidate would dominate
                     image_size=image_size, detector_settings=presets.describe("candidate"),
                     detector=presets.make_detector("candidate"), board=board)


def evaluate(settings, ctx):
    """Detect with these settings over both cameras and score the result.

    detect_tags() prints a line per camera, which would bury the search output, so its stdout is
    captured. Results are memoised because coordinate descent revisits the current best on every
    round and each evaluation re-reads every frame.
    """
    frozen = tuple(sorted(settings.items()))
    if frozen in ctx.seen:
        return ctx.seen[frozen]

    started = time.time()
    board_tags = duplicates = false_tags = usable = total = 0
    for camera in calib.CAMERAS:
        run = make_run(ctx.frames_dir, settings, ctx.board_params, ctx.image_size, ctx.board,
                       ctx.min_tags)
        with contextlib.redirect_stdout(io.StringIO()):
            detections = calib.detect_tags(run, camera)
        for _, ids in detections.values():
            found = [] if ids is None else ids.ravel().tolist()
            on_board = [i for i in found if i in BOARD_IDS]
            distinct = len(set(on_board))
            total += len(found)
            board_tags += distinct
            duplicates += len(on_board) - distinct
            false_tags += len(found) - len(on_board)
            usable += distinct >= ctx.min_tags

    score = Score(board_tags, duplicates, false_tags, usable, total, time.time() - started)
    ctx.seen[frozen] = score
    append_cache(ctx.cache_path, settings, score)
    return score


# ====== EVALUATION CACHE ======
# One evaluation re-reads every frame and can take anything from 4 s to half an hour, so losing a
# run's work to a stopped process is expensive. Each result is appended as it is produced, and a
# later run reloads them, which makes the sweep resumable and makes widening SWEEP_AXES cheap:
# only the genuinely new combinations are computed.
def cache_header(ctx_frames_dir, min_tags, board_params):
    """What a cached score depends on besides the settings themselves.

    A cache built on other frames, another min_tags or other board parameters describes a different
    question, so it is refused rather than silently mixed in.
    """
    return {"frames_dir": calib.rel(ctx_frames_dir), "min_tags": min_tags,
            "board_params": dict(sorted(board_params.items()))}


def load_cache(path, header):
    """Read the cache written by earlier runs. Returns {frozen settings: Score}, empty if unusable."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    if not lines or lines[0] != header:
        print(f"ignoring {calib.rel(path)}: it was built for different frames or settings")
        return {}

    seen = {}
    for entry in lines[1:]:
        settings = entry["settings"]
        seen[tuple(sorted(settings.items()))] = Score(**entry["score"])
    print(f"reusing {len(seen)} evaluations from {calib.rel(path)}")
    return seen


def start_cache(path, header, fresh):
    """Create or keep the cache file, and return what can be reused from it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fresh and os.path.exists(path):
        os.remove(path)
        print(f"discarded {calib.rel(path)} (--fresh)")

    seen = load_cache(path, header)
    if not seen:
        # Rewrite from the header so a cache refused above is replaced rather than appended to.
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
    return seen


def append_cache(path, settings, score):
    """Record one evaluation. Flushed immediately: the point is to survive a killed process."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"settings": dict(sorted(settings.items())),
                             "score": score.__dict__}) + "\n")


@dataclass
class Context:
    """Everything an evaluation needs that does not change between candidates."""
    frames_dir: str
    board_params: dict
    image_size: tuple
    board: object
    min_tags: int
    max_bad_rate: float
    cache_path: str
    seen: dict


def search(ctx, settings, rounds):
    """Coordinate descent over SWEEP_AXES. Returns (best settings, best score, trial log)."""
    best = dict(settings)
    best_score = evaluate(best, ctx)
    print(f"start: {best_score}   {describe(best)}\n")
    trials = [{"settings": dict(best), "score": best_score.__dict__, "axis": "start"}]

    for round_no in range(1, rounds + 1):
        changed = False
        print(f"===== round {round_no} =====")
        for axis, values in SWEEP_AXES.items():
            # The current value is included so the axis can decline to move; it is already memoised.
            current = best.get(axis)
            round_best, round_best_score = best, best_score
            for value in values:
                candidate = {**best, axis: value}
                score = evaluate(candidate, ctx)
                trials.append({"settings": candidate, "score": score.__dict__, "axis": axis})
                # A candidate over the ceiling is printed, so the sweep still shows what it did,
                # but it cannot win: past that point extra tags are being bought with misreads.
                rejected = score.bad_rate > ctx.max_bad_rate
                better = not rejected and score.key > round_best_score.key
                mark = "REJECTED" if rejected else ("*" if better else "")
                print(f"  {axis} = {value!s:<6} {score} {mark}")
                if better:
                    round_best, round_best_score = candidate, score
            if round_best_score.key > best_score.key:
                print(f"  -> {axis}: {current} -> {round_best[axis]} "
                      f"(+{round_best_score.board_tags - best_score.board_tags} tags)")
                best, best_score, changed = round_best, round_best_score, True
        if not changed:
            print(f"round {round_no} changed nothing; stopping\n")
            break

    return best, best_score, trials


def describe(settings):
    """One-line form of a settings dict, matching how Calibration.py records a preset."""
    return ", ".join(f"{k}={v}" for k, v in sorted(settings.items()))


def preset_json(name, settings, start, score):
    """The winning settings as a detector_presets.json preset, ready to paste."""
    why = (f"Coordinate-descent sweep from '{start}' by sweep_detector.py: {score.board_tags} "
           f"distinct board tags over {score.usable_images} usable images, "
           f"{score.bad_rate:.2%} wrong detections.")
    return {name: {"_why": why, **{k: v for k, v in sorted(settings.items())}}}


def save_preset(name, body):
    """Add a preset to detector_presets.json in place, then re-validate the file.

    json.dump loses the file's hand-written layout but keeps its "_comment" keys, since those are
    ordinary members of the object. Validating afterwards means a mistake here fails now rather
    than on the next calibration run.
    """
    path = calib.PRESETS_FILE
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    if name in config["presets"]:
        print(f"  replacing existing preset {name!r} in {calib.rel(path)}")
    config["presets"].update(body)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    validate(json.load(open(path, encoding="utf-8")), path)
    print(f"saved preset {name!r} to {calib.rel(path)}")


def write_log(out_dir, frames_dir, start, max_bad_rate, best, best_score, trials):
    """Record every candidate tried, so a claim about a preset can be traced back to its run."""
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(frames_dir)).replace(" ", "_")
    path = os.path.join(out_dir, f"{name}_sweep.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"frames_dir": calib.rel(frames_dir), "start_preset": start,
                   "max_bad_rate": max_bad_rate, "axes": SWEEP_AXES,
                   "best_settings": best, "best_score": best_score.__dict__,
                   "trials": trials}, fh, indent=2)
    print(f"search log: {calib.rel(path)}")
    return path


def main():
    args = parse_args()
    presets = calib.load_presets()
    start = args.start or presets.default
    if start not in presets.presets:
        raise SystemExit(f"unknown preset {start!r}; {calib.PRESETS_FILE} defines: "
                         f"{', '.join(presets.presets)}")

    for camera in calib.CAMERAS:
        if not calib.find_images(args.frames_dir, camera):
            raise SystemExit(f"no {camera} PNGs under {args.frames_dir}/*/{camera}/")
    n_images = sum(len(calib.find_images(args.frames_dir, c)) for c in calib.CAMERAS)
    image_size = calib.cv2.imread(calib.find_images(args.frames_dir, "left")[0],
                                  calib.cv2.IMREAD_GRAYSCALE).shape[::-1]
    print(f"{n_images} images under {calib.rel(args.frames_dir)}, size {image_size}")
    print(f"starting from preset {start!r}; board params {describe(presets.board_params)}")
    print(f"at most {calib.N_TAGS * n_images} tags are on the board across these images; "
          f"candidates over {args.max_bad_rate:.2%} wrong detections are refused\n")

    run_name = os.path.basename(os.path.normpath(args.frames_dir)).replace(" ", "_")
    cache_path = os.path.join(args.out_dir, f"{run_name}_evaluations.jsonl")
    header = cache_header(args.frames_dir, args.min_tags, presets.board_params)
    seen = start_cache(cache_path, header, args.fresh)

    ctx = Context(frames_dir=args.frames_dir, board_params=presets.board_params,
                  image_size=image_size, board=calib.build_board(), min_tags=args.min_tags,
                  max_bad_rate=args.max_bad_rate, cache_path=cache_path, seen=seen)
    # The starting point is the preset's own tuning only; board_params are merged in by
    # make_detector(), so putting them in the search state as well would let a round move them.
    best, best_score, trials = search(ctx, presets.presets[start], args.rounds)

    print("\n===== best =====")
    print(f"{best_score}")
    print(describe(best))
    baseline = ctx.seen[tuple(sorted(presets.presets[start].items()))]
    print(f"against {start!r}: {best_score.board_tags - baseline.board_tags:+d} tags, "
          f"{best_score.usable_images - baseline.usable_images:+d} usable images")
    print(f"{len(ctx.seen)} distinct settings evaluated")

    name = args.save_preset or "swept"
    body = preset_json(name, best, start, best_score)
    print(f"\nAs a detector_presets.json preset:\n{json.dumps(body, indent=2)}")
    write_log(args.out_dir, args.frames_dir, start, args.max_bad_rate, best, best_score,
              trials)
    if args.save_preset:
        save_preset(args.save_preset, body)
    print(f"\nNow check it calibrates well, not just detects well:\n"
          f"  src/venv/bin/python src/01_Calibration/Calibration.py --preset {name}")


if __name__ == "__main__":
    main()
