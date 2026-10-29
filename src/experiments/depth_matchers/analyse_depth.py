"""Table and figures of the depth experiment, from what depth_frames.py saved (run_depth.py first).

    results/<out_dir>/summary.csv, .md, .tex           per recording and matcher: frames, median seconds per
                                                       frame (matcher alone, and the whole frame), median pixels
                                                       with depth (and as % of the image), median points (1 mm voxels)
    results/<out_dir>/<recording>/<matcher>/depth/     every frame: rectified left image | depth map
    results/<out_dir>/<recording>/matches_<matcher>.jpg  one frame: rectified left above right, a line from each
                                                       grid pixel to where the matcher put it in the right image
    results/<out_dir>/<recording>/matches.jpg          the matches figures side by side, one per matcher
    results/<out_dir>/<recording>/depth_compare.jpg    the same frame's depth map from every matcher
    results/<out_dir>/depth_compare_all.png            figures.combined: one column per recording (its image, then
                                                       every matcher's depth map), one depth scale for all
    results/<out_dir>/depth_crops.png                  figures.crops: a close-up of each recording's object, its
                                                       image then every matcher's depth, each row on its own scale

    src/venv/bin/python src/experiments/depth_matchers/analyse_depth.py [--trial N] [the options run_depth.py took]
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
for _stage in ("03_depth", "04_pose"):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), _stage))
import common  # noqa: E402
import depth_config  # noqa: E402
from dense_stereo import Rectifier  # noqa: E402
from stereo_rig import find_frame_pairs, load_rig  # noqa: E402

NO_DEPTH = (60, 60, 60)        # BGR of pixels with no depth
INK = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
MATCH_FONT_SCALE = 1.4


# ====== INPUTS ======
def load_run(cfg, recording, matcher):
    """(depth_frames.json record, folder) of one run, or (None, folder) if it has not been made."""
    folder = depth_config.depth_dir(cfg, recording, matcher)
    path = os.path.join(folder, "depth_frames.json")
    if not os.path.exists(path):
        return None, folder
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), folder


def read_disparity(folder, name):
    """Disparity (px, float) of one frame; 0 where no depth was kept."""
    raw = cv2.imread(os.path.join(folder, "disparity", os.path.splitext(name)[0] + ".png"), cv2.IMREAD_UNCHANGED)
    return None if raw is None else raw.astype(np.float32) / 16.0


def rectified_colour(rectifier, camera, path):
    """The frame rectified like the matcher saw it, but in colour (the matchers see grey)."""
    return cv2.remap(cv2.imread(path, cv2.IMREAD_COLOR), *rectifier.maps[camera], cv2.INTER_LINEAR)


def depth_of(rectifier, disparity):
    """Camera depth (mm) per pixel, NaN where there is none."""
    z = cv2.reprojectImageTo3D(disparity, rectifier.rig.Q)[..., 2]
    return np.where(disparity > 0, z, np.nan)


# ====== TABLE ======
TOLERANCE_PX = 1.0   # the tuning's tolerance (tune_config.json), so the test scores mean the same thing


def test_truth(cfg, recording, names):
    """{frame name: plane disparity} for the frames with a usable tag (board_truth.py), cached in the work folder."""
    import board_truth
    import tag_poses
    folder = os.path.join(common.project_path(cfg["work_dir"]), recording["name"], "truth")
    os.makedirs(folder, exist_ok=True)
    rectifier, detector = Rectifier(load_rig(common.project_path(recording["calibration"]))), tag_poses.make_detector()
    pairs = {n: (l, r) for n, l, r in find_frame_pairs(common.project_path(recording["frames"]))}
    truth = {}
    for name in names:
        path = os.path.join(folder, os.path.splitext(name)[0] + ".npz")
        if not os.path.exists(path):
            t = board_truth.frame_truth(rectifier, detector, *pairs[name])
            np.savez_compressed(path, disparity=np.zeros(0, np.float32) if t is None else t["disparity"])
        disparity = np.load(path)["disparity"]
        if disparity.size:
            truth[name] = disparity
    return truth


def accuracy(cfg, recording, matcher, truth):
    """Precision, recall, F1 and median error (px) of a run on the tag planes of the test frames, or None."""
    import board_truth
    record, folder = load_run(cfg, recording, matcher)
    counts = []
    for f in record["per_frame"]:
        if f["name"] in truth:
            plane = truth[f["name"]]
            counts.append(board_truth.score(read_disparity(folder, f["name"]), {"region": plane > 0, "disparity": plane},
                                            TOLERANCE_PX))
    return board_truth.summarise(counts, TOLERANCE_PX) if counts else None


def summarise(cfg):
    rows = []
    for recording in cfg["recordings"]:
        runs = {m: load_run(cfg, recording, m)[0] for m in cfg["matchers"]}
        names = sorted({f["name"] for r in runs.values() if r for f in r["per_frame"]})
        truth = test_truth(cfg, recording, names)
        for matcher, record in runs.items():
            if record is None:
                print(f"  {recording['name']} / {matcher}: not run yet")
                continue
            frames = record["per_frame"]
            first = cv2.imread(find_frame_pairs(common.project_path(recording["frames"]))[0][1], cv2.IMREAD_GRAYSCALE)
            pixels = np.median([f["pixels"] for f in frames])
            score = accuracy(cfg, recording, matcher, truth)
            rows.append({"recording": recording["name"], "matcher": matcher, "frames": len(frames),
                         "match_s_median": round(float(np.median([f["match_seconds"] for f in frames])), 2),
                         "frame_s_median": round(float(np.median([f["frame_seconds"] for f in frames])), 2),
                         "pixels_median": int(pixels), "coverage_pct": round(100 * pixels / first.size, 1),
                         "points_median": int(np.median([f["points"] for f in frames])),
                         "tag_frames": score["frames"] if score else 0,
                         "precision": round(score["precision"], 3) if score else None,
                         "recall": round(score["recall"], 3) if score else None,
                         "f1": round(score["f1"], 3) if score else None,
                         "median_error_px": round(score["median_error_px"], 2) if score and score["median_error_px"] is not None else None})
    out = depth_config.result_dir(cfg)
    os.makedirs(out, exist_ok=True)
    if not rows:
        return rows
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dash = lambda v, f: "-" if v is None else format(v, f)  # noqa: E731
    lines = ["| Recording | Matcher | Frames | s/frame (matcher) | s/frame (total) | Pixels with depth | Coverage | "
             "Points (1 mm voxels) | Tag frames | Precision | Recall | F1 | Median error (px) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['recording']} | {r['matcher']} | {r['frames']} | {r['match_s_median']:.2f} | {r['frame_s_median']:.2f} | "
              f"{r['pixels_median']:,} | {r['coverage_pct']:.1f}% | {r['points_median']:,} | {r['tag_frames']} | "
              f"{dash(r['precision'], '.3f')} | {dash(r['recall'], '.3f')} | {dash(r['f1'], '.3f')} | "
              f"{dash(r['median_error_px'], '.2f')} |" for r in rows]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("Medians over frames. Seconds: matcher alone, and the whole frame (rectify, match, points, save).\n"
                 f"Precision, recall, F1, median error: on the tag planes (board_truth.py) of the test frames with a "
                 f"usable tag, tolerance {TOLERANCE_PX:g} px of disparity.\n\n")
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.tex"), "w", encoding="utf-8") as fh:
        fh.write(latex_table(cfg, rows))
    print("\n".join(lines))
    return rows


MATCHER_NAMES = {"sgbm": "SGBM", "sgbm_tuned": "SGBM (tuned)", "aquastereo": "AquaStereo",
                 "foundationstereo": "FoundationStereo"}


def latex_table(cfg, rows):
    """summary rows as a booktabs table (needs booktabs and multirow), one block of rows per recording."""
    titles = cfg["figures"].get("combined", {}).get("titles", {})
    lines = [r"% Made by src/experiments/depth_matchers/analyse_depth.py from depth_frames.py's output; do not edit.",
             r"\begin{table}[H]", r"\centering",
             r"\caption{Depth map generation per matcher, medians over the frames used. Time is the matcher alone. "
             r"Coverage is the share of the image with depth in the recording's depth range; points are those pixels "
             r"merged into 1~mm voxels. Precision, recall and median error are measured against the plane of each tag "
             r"and its white surround, in the frames where a tag was found in both cameras (none in the sun "
             rf"recording): precision is the share of the matcher's depth there within {TOLERANCE_PX:g}~px of "
             r"disparity of the plane, recall the share of the tag regions it covers within that tolerance.}",
             r"\label{tab:depth_matchers}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Scene & Matcher & Time & Coverage & Points & Precision & Recall & Median error \\",
             r" & & (s/frame) & (\%) & ($\times10^3$) & & & (px) \\", r"\midrule"]
    recordings = [r["name"] for r in cfg["recordings"]]
    order = cfg["figures"].get("combined", {}).get("recordings", [])
    recordings = [n for n in order if n in recordings] + [n for n in recordings if n not in order]
    dash = lambda v, f: "--" if v is None else format(v, f)  # noqa: E731
    for i, name in enumerate(recordings):
        block = [r for r in rows if r["recording"] == name]
        if not block:
            continue
        tag_frames = max(r["tag_frames"] for r in block)
        scene = f"{titles.get(name, name).replace('_', r'\_')}"
        detail = f"{block[0]['frames']} frames" + (f", {tag_frames} with tags" if tag_frames else "")
        for j, r in enumerate(block):
            first = scene if j == 0 else (detail if j == 1 else "")
            lines.append(f"{first} & {MATCHER_NAMES.get(r['matcher'], r['matcher'])} & {r['match_s_median']:.2f} & "
                         f"{r['coverage_pct']:.1f} & {r['points_median'] / 1000:,.0f} & {dash(r['precision'], '.2f')} & "
                         f"{dash(r['recall'], '.2f')} & {dash(r['median_error_px'], '.2f')} \\\\")
        if i < len(recordings) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def colour_depth(depth, depth_min, depth_max):
    """Depth as colour: viridis, near = yellow, far = purple; no depth = dark grey."""
    t = np.clip((depth_max - np.nan_to_num(depth, nan=depth_max)) / (depth_max - depth_min), 0, 1)
    image = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    image[np.isnan(depth)] = NO_DEPTH
    return image


def round_ticks(depth_min, depth_max, count=5):
    """Fractions along the key for about count round depths (multiples of 50, 100, 250 or 500 mm) in the range."""
    step = next(s for s in (5, 10, 20, 25, 50, 100, 250, 500, 1000) if (depth_max - depth_min) / s <= count)
    values = np.arange(np.ceil(depth_min / step) * step, depth_max + 1e-6, step)
    return tuple((v - depth_min) / (depth_max - depth_min) for v in values)


def colour_bar(height, depth_min, depth_max, width=90, font=0.45, ticks=(0, 0.5, 1), light=False):
    """A vertical key for colour_depth, labelled in mm."""
    bar_width = int(24 * font / 0.45)
    t = np.linspace(1, 0, height)[:, None].repeat(bar_width, 1)
    bar = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    ink = (40, 40, 40) if light else INK
    key = np.full((height, width, 3), 255 if light else 0, np.uint8)
    key[:, 6:6 + bar_width] = bar
    if light:
        cv2.rectangle(key, (6, 0), (5 + bar_width, height - 1), ink, 1)
    text_x, thickness, pad = 28 + bar_width, max(1, int(round(font * 2))), int(28 * font)
    cv2.putText(key, "mm", (text_x, pad + pad // 3), FONT, font, ink, thickness, cv2.LINE_AA)
    for frac in ticks:
        y = int(np.clip(frac * (height - 1), pad, height - pad // 2))
        if y < 2.5 * pad:   # would collide with the unit at the top
            continue
        label = f"{depth_min + frac * (depth_max - depth_min):.0f}"
        start = 6 + bar_width if light else 6
        cv2.line(key, (start, y), (6 + bar_width + (7 if light else 4), y), ink, thickness)
        cv2.putText(key, label, (text_x, y + pad // 3), FONT, font, ink, thickness, cv2.LINE_AA)
    return key


def label(image, text, scale=0.8):
    thick = max(2, int(round(2.5 * scale)))
    cv2.putText(image, text, (12, int(38 * scale)), FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(image, text, (12, int(38 * scale)), FONT, scale, INK, thick, cv2.LINE_AA)
    return image


def depth_panel(rectifier, recording, left_colour, disparity, title, scale):
    """Rectified left | depth map | key, scaled for saving."""
    depth = colour_depth(depth_of(rectifier, disparity), recording["depth_min_mm"], recording["depth_max_mm"])
    small = [cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for i in (left_colour, depth)]
    panel = np.hstack([small[0], small[1], colour_bar(small[1].shape[0], recording["depth_min_mm"], recording["depth_max_mm"])])
    return label(panel, title)


def depth_images(cfg, recording, matcher, rectifier, pairs):
    """The depth panel of every frame the run made."""
    record, folder = load_run(cfg, recording, matcher)
    out = os.path.join(depth_config.result_dir(cfg, recording), matcher, "depth")
    os.makedirs(out, exist_ok=True)
    for f in record["per_frame"]:
        disparity = read_disparity(folder, f["name"])
        left = rectified_colour(rectifier, "left", pairs[f["name"]][0])
        panel = depth_panel(rectifier, recording, left, disparity,
                            f"{matcher}  {f['name']}  {f['pixels']:,} px with depth  {f['match_seconds']:.2f} s",
                            cfg["figures"]["image_scale"])
        cv2.imwrite(os.path.join(out, os.path.splitext(f["name"])[0] + ".jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  {recording['name']} / {matcher}: {len(record['per_frame'])} depth images -> {common.rel(out)}/")


def grid_pixels(shape, grid):
    """Pixel centres of a cols x rows grid over the image, inset by half a cell."""
    cols, rows = grid
    xs = (np.arange(cols) + 0.5) * shape[1] / cols
    ys = (np.arange(rows) + 0.5) * shape[0] / rows
    return [(int(x), int(y)) for y in ys for x in xs]


def matches_figure(rectifier, recording, left, right, disparity, grid, title):
    """Rectified left above right; a line from each grid pixel to where the matcher found it in the right image,
    coloured by its depth; a grey cross where the matcher kept no depth."""
    height = left.shape[0]
    canvas = np.vstack([left, right])
    depth = depth_of(rectifier, disparity)
    found = 0
    for x, y in grid_pixels(left.shape, grid):
        d = disparity[y, x]
        if d <= 0:
            cv2.drawMarker(canvas, (x, y), (150, 150, 150), cv2.MARKER_TILTED_CROSS, 18, 3, cv2.LINE_AA)
            continue
        found += 1
        colour = tuple(int(c) for c in colour_depth(np.array([[depth[y, x]]]), recording["depth_min_mm"],
                                                    recording["depth_max_mm"])[0, 0])
        end = (int(round(x - d)), y + height)
        cv2.line(canvas, (x, y), end, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(canvas, (x, y), end, colour, 3, cv2.LINE_AA)
        for point in ((x, y), end):
            cv2.circle(canvas, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 5, colour, -1, cv2.LINE_AA)
    label(canvas, f"{title}: {found}/{len(grid_pixels(left.shape, grid))} grid pixels matched", MATCH_FONT_SCALE)
    cv2.putText(canvas, "left (rectified)", (12, height - 16), FONT, MATCH_FONT_SCALE, INK, 3, cv2.LINE_AA)
    cv2.putText(canvas, "right (rectified)", (12, 2 * height - 16), FONT, MATCH_FONT_SCALE, INK, 3, cv2.LINE_AA)
    cv2.line(canvas, (0, height), (canvas.shape[1], height), INK, 2)
    return canvas


def figure_frame(cfg, recording):
    """(frame name, {matcher: run folder}) of the figures' frame: one every matcher made, at frame_index."""
    runs = {m: load_run(cfg, recording, m) for m in cfg["matchers"]}
    runs = {m: folder for m, (record, folder) in runs.items() if record is not None}
    if not runs:
        return None, runs
    names = sorted(set.intersection(*[{f["name"] for f in load_run(cfg, recording, m)[0]["per_frame"]} for m in runs]))
    index = cfg["figures"]["frame_index"]
    return (names[len(names) // 2] if index == "middle" else names[min(int(index), len(names) - 1)]), runs


def frame_figures(cfg, recording, rectifier, pairs):
    """For one frame: every matcher's correspondences, side by side, and every matcher's depth map, stacked."""
    name, runs = figure_frame(cfg, recording)
    if not runs:
        return
    left = rectified_colour(rectifier, "left", pairs[name][0])
    right = rectified_colour(rectifier, "right", pairs[name][1])
    scale, out = cfg["figures"]["image_scale"], depth_config.result_dir(cfg, recording)
    matches, panels = [], []
    for matcher, folder in runs.items():
        disparity = read_disparity(folder, name)
        figure = matches_figure(rectifier, recording, left, right, disparity, cfg["figures"]["grid"], f"{matcher}, {name}")
        figure = cv2.resize(figure, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out, f"matches_{matcher}.jpg"), figure, [cv2.IMWRITE_JPEG_QUALITY, 92])
        matches.append(figure)
        panels.append(depth_panel(rectifier, recording, left, disparity, f"{matcher}  {name}", scale))
    gap = lambda shape, axis: np.full((shape[0], 12, 3) if axis else (12, shape[1], 3), 255, np.uint8)  # noqa: E731
    cv2.imwrite(os.path.join(out, "matches.jpg"),
                np.hstack([x for m in matches for x in (m, gap(m.shape, 1))][:-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(out, "depth_compare.jpg"),
                np.vstack([x for p in panels for x in (p, gap(p.shape, 0))][:-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {recording['name']}: frame {name} -> matches.jpg, depth_compare.jpg")


def combined_figure(cfg, spec):
    """One figure across recordings: a column per recording (in spec's order) holding its rectified left image and then
    every matcher's depth map of its figure frame, all coloured on one depth scale, with one key on the right."""
    recordings = {r["name"]: r for r in cfg["recordings"]}
    columns = [recordings[n] for n in spec["recordings"] if n in recordings]
    if not columns:
        return
    depth_min = spec.get("depth_min_mm", min(r["depth_min_mm"] for r in columns))
    depth_max = spec.get("depth_max_mm", max(r["depth_max_mm"] for r in columns))
    scale, gap = cfg["figures"]["image_scale"], 12
    tiles = []   # per column: [image, depth per matcher]
    for recording in columns:
        name, runs = figure_frame(cfg, recording)
        rectifier = Rectifier(load_rig(common.project_path(recording["calibration"])))
        pairs = {n: (l, r) for n, l, r in find_frame_pairs(common.project_path(recording["frames"]))}
        left = rectified_colour(rectifier, "left", pairs[name][0])
        column = [label(cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                        f"{spec.get('titles', {}).get(recording['name'], recording['name'])}  ({name})")]
        for matcher in cfg["matchers"]:
            if matcher not in runs:
                column.append(None)
                continue
            depth = colour_depth(depth_of(rectifier, read_disparity(runs[matcher], name)), depth_min, depth_max)
            column.append(label(cv2.resize(depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), matcher))
        tiles.append(column)
        print(f"  combined: {recording['name']} frame {name}")
    h, w = tiles[0][0].shape[:2]
    rows = len(tiles[0])
    key_width = 170
    canvas = np.full((rows * h + (rows - 1) * gap, len(tiles) * w + len(tiles) * gap + key_width, 3), 255, np.uint8)
    for c, column in enumerate(tiles):
        for r, tile in enumerate(column):
            y, x = r * (h + gap), c * (w + gap)
            canvas[y:y + h, x:x + w] = NO_DEPTH if tile is None else cv2.resize(tile, (w, h))
    # One key for the depth rows, beside them.
    top = h + gap
    canvas[top:, -key_width:] = colour_bar(canvas.shape[0] - top, depth_min, depth_max, width=key_width, font=1.0,
                                           ticks=round_ticks(depth_min, depth_max))
    out = os.path.join(depth_config.result_dir(cfg), spec.get("name", "depth_compare_all") + ".png")
    cv2.imwrite(out, canvas)
    print(f"  combined figure ({depth_min:.0f}-{depth_max:.0f} mm) -> {common.rel(out)}")


def crop_padded(array, box, fill):
    """array[y0:y1, x0:x1] for box (x0, y0, x1, y1), which may run past the image: the part outside is fill (NaN for
    depth, which is drawn as no depth), so an object at the frame's edge can still be placed anywhere in its tile."""
    x0, y0, x1, y1 = box
    out = np.full((y1 - y0, x1 - x0) + array.shape[2:], fill, dtype=array.dtype if np.isfinite(fill) else np.float32)
    h, w = array.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def crop_figure(cfg, spec):
    """Close-ups for the report: a row per recording (spec["rows"]: recording, box [x0, y0, x1, y1] in the rectified
    left image at full size, title), holding the crop of the image and then every matcher's depth of it."""
    recordings = {r["name"]: r for r in cfg["recordings"]}
    tile, gap, key_width = spec.get("tile_height", 420), 12, 170
    text = spec.get("label_scale", 0.8)
    white = lambda h, w: np.full((h, w, 3), 255, np.uint8)  # noqa: E731
    fit = lambda image: cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)  # noqa: E731

    rows = []   # (row spec, image crop, {matcher: depth crop}, (low, high))
    for row in spec["rows"]:
        recording = recordings.get(row["recording"])
        if recording is None:
            continue
        name, runs = figure_frame(cfg, recording)
        rectifier = Rectifier(load_rig(common.project_path(recording["calibration"])))
        pairs = {n: (l, r) for n, l, r in find_frame_pairs(common.project_path(recording["frames"]))}
        box = row["box"]
        depths = {m: crop_padded(depth_of(rectifier, read_disparity(runs[m], name)), box, np.nan)
                  for m in cfg["matchers"] if m in runs}
        low, high = np.percentile(np.concatenate([d[np.isfinite(d)] for d in depths.values()]), [5, 95])
        image = crop_padded(rectified_colour(rectifier, "left", pairs[name][0]), box, 128)
        # rotate_cw: rotate the row clockwise (multiple of 90) so the recordings line up.
        turns = -(row.get("rotate_cw", 0) // 90)
        if turns:
            image = np.ascontiguousarray(np.rot90(image, turns))
            depths = {m: np.ascontiguousarray(np.rot90(d, turns)) for m, d in depths.items()}
        rows.append((row, image, depths,
                     (low - 0.1 * (high - low), high + 0.1 * (high - low))))
        print(f"  crops: {recording['name']} frame {name}, depth {rows[-1][3][0]:.0f}-{rows[-1][3][1]:.0f} mm")
    if not rows:
        return
    common_scale = spec.get("common_scale", False)
    if common_scale:
        shared = (min(r[3][0] for r in rows), max(r[3][1] for r in rows))

    # Column headings along the top, row names down the left.
    ink, thick = (30, 30, 30), max(2, int(round(2.2 * text)))

    def caption(width, height, words, vertical=False):
        box = white(width, height) if vertical else white(height, width)
        (w, h), _ = cv2.getTextSize(words, FONT, text, thick)
        cv2.putText(box, words, ((box.shape[1] - w) // 2, (box.shape[0] + h) // 2), FONT, text, ink, thick, cv2.LINE_AA)
        return cv2.rotate(box, cv2.ROTATE_90_COUNTERCLOCKWISE) if vertical else box

    header_h, side_w = int(64 * text), int(64 * text)
    strips = []
    for row, image, depths, own in rows:
        low, high = shared if common_scale else own
        tiles = [caption(side_w, tile, row.get("title", row["recording"]), vertical=True), fit(image)]
        for matcher in cfg["matchers"]:
            tiles.append(fit(colour_depth(depths[matcher], low, high)) if matcher in depths else
                         np.full((tile, tile, 3), NO_DEPTH, np.uint8))
        if not common_scale:
            tiles.append(colour_bar(tile, low, high, width=key_width, font=0.8, ticks=round_ticks(low, high), light=True))
        strips.append(np.hstack([x for t in tiles for x in (t, white(tile, gap))][:-1]))
    body = np.vstack([x for strip in strips for x in (strip, white(gap, strip.shape[1]))][:-1])
    headings = [white(header_h, side_w), caption(tile, header_h, "Image")] + \
        [caption(tile, header_h, MATCHER_NAMES.get(m, m)) for m in cfg["matchers"]]
    header = np.hstack([x for t in headings for x in (t, white(header_h, gap))][:-1])
    header = np.hstack([header, white(header_h, body.shape[1] - header.shape[1])])
    figure = np.vstack([header, body])
    if common_scale:
        key = colour_bar(body.shape[0], *shared, width=key_width, font=0.8, ticks=round_ticks(*shared, count=8),
                         light=True)
        figure = np.hstack([figure, white(figure.shape[0], gap), np.vstack([white(header_h, key_width), key])])
        print(f"  crops: one scale, {shared[0]:.0f}-{shared[1]:.0f} mm")
    out = os.path.join(depth_config.result_dir(cfg), spec.get("name", "depth_crops") + ".png")
    cv2.imwrite(out, figure)
    print(f"  crops figure -> {common.rel(out)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    depth_config.add_arguments(ap)
    ap.add_argument("--no-depth-images", action="store_true", help="skip the per-frame depth images (slowest part)")
    args = ap.parse_args(argv)
    cfg = depth_config.resolve(args, ap)
    rows = summarise(cfg)
    for recording in cfg["recordings"]:
        rectifier = Rectifier(load_rig(common.project_path(recording["calibration"])))
        pairs = {name: (left, right) for name, left, right in find_frame_pairs(common.project_path(recording["frames"]))}
        if not args.no_depth_images:
            for matcher in cfg["matchers"]:
                if load_run(cfg, recording, matcher)[0] is not None:
                    depth_images(cfg, recording, matcher, rectifier, pairs)
        frame_figures(cfg, recording, rectifier, pairs)
    if cfg["figures"].get("combined"):
        combined_figure(cfg, cfg["figures"]["combined"])
    if cfg["figures"].get("crops"):
        crop_figure(cfg, cfg["figures"]["crops"])
    for spec in cfg["figures"].get("extra_crops", []):
        crop_figure(cfg, spec)
    print(f"results in {common.rel(depth_config.result_dir(cfg))}/")
    return rows


if __name__ == "__main__":
    main()
