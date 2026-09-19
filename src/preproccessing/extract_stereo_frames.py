"""

Note - this script was generated with the assistance of claude code
It allows for a simple and standardised extraction of chosen frames fram raw YUYV recordngs.

Extract timestamp-matched stereo frame pairs from raw YUYV or MJPEG .mkv recordings.

A recording folder must contain left.mkv, left_timestamps.csv, right.mkv and
right_timestamps.csv. Each right frame is paired with the left frame whose
pts_ns is nearest; pairs further apart than --max-dt-ms are skipped with a
warning. Output goes to <folder>/frames/{left,right}/NNNN.png (lossless, same
name for both halves of a pair) plus <folder>/frames/pairs.csv.

The format is detected from the files. --gray saves 8-bit grayscale luma
instead of colour; use it for calibration and stereo matching, keep colour for
texturing.
    raw YUYV  colour converted by ffmpeg (BT.601, limited range); --gray is the
              recorded Y plane byte-for-byte (levels 16-235), no conversion
    MJPEG     frames decoded by OpenCV (libjpeg); --gray is the JPEG's luma
              channel, full range 0-255. The camera already JPEG-compressed
              these frames, so neither output is lossless sensor data.

Usage (setup: python3 -m venv src/venv && src/venv/bin/pip install -r src/requirements.txt):
    PY=src/venv/bin/python
    $PY src/preproccessing/extract_stereo_frames.py data/Calibration/LabTesting/26Aug all --step 20
    $PY src/preproccessing/extract_stereo_frames.py data/Calibration/LabTesting/26Aug select --gray
    $PY src/preproccessing/extract_stereo_frames.py data/Calibration/18_Sep/C1 select --gray --name C1

Viewer keys (select mode, right camera with matched left alongside):
    a/d or left/right   back/forward 1      s/w or down/up, A/D   back/forward 10
    space               mark/unmark         enter                 save pairs
    q/esc               quit (press twice if there are unsaved changes)
Select mode starts with nothing marked; saving replaces any existing frames/ output.

Labels (select mode with --name PREFIX): marking a frame asks for a label, typed
in the viewer (letters, digits, - _ .; enter confirms, esc cancels, empty uses
the frame number). The pair is saved as left/PREFIX_Left_LABEL.png and
right/PREFIX_Right_LABEL.png, e.g. --name C1 and label 20mm give C1_Left_20mm.png.
Without --name, files are numbered 0000.png, 0001.png, ...
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CAMS = ("left", "right")
KEYS = {  # waitKeyEx codes: ASCII, X11 keysyms, Qt key codes, Windows codes
    -1: (ord("a"), 65361, 0x1000012, 2424832),
    +1: (ord("d"), 65363, 0x1000014, 2555904),
    -10: (ord("s"), ord("A"), 65364, 0x1000015, 2621440),
    +10: (ord("w"), ord("D"), 65362, 0x1000013, 2490368),
}
SAVE_KEYS = (10, 13, 65293, 65421, 0x1000004, 0x1000005)
QUIT_KEYS = (ord("q"), 27)
BACKSPACE_KEYS = (8, 65288, 0x1000003)
LABEL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
WHITE, GREEN, RED, YELLOW = (255, 255, 255), (0, 255, 0), (0, 0, 255), (0, 255, 255)
HELP = "a/d: -1/+1   s/w: -10/+10   space: mark   enter: save   q: quit"


def probe(video):
    """Return (width, height, codec, offsets, sizes) for a raw YUYV or MJPEG .mkv.

    codec is "yuyv" or "mjpeg"; offsets/sizes locate each frame's data in the
    file (the pixels for YUYV, a complete JPEG for MJPEG). ffprobe's packet pos
    points at the Matroska block body: a track-number varint, a 2-byte timecode
    and a flags byte come before the frame data.
    """
    info = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=codec_name,width,height,pix_fmt:packet=pos,size", str(video)],
        capture_output=True, text=True, check=True).stdout)
    s = info["streams"][0]
    w, h = s["width"], s["height"]
    codec = {("rawvideo", "yuyv422"): "yuyv"}.get((s["codec_name"], s["pix_fmt"]),
                                                  "mjpeg" if s["codec_name"] == "mjpeg" else None)
    if codec is None:
        sys.exit(f"{video}: unsupported format {s['codec_name']}/{s['pix_fmt']} (need raw yuyv422 or mjpeg)")
    offsets, sizes = [], []
    with open(video, "rb") as f:
        for p in info["packets"]:
            pos, size = int(p["pos"]), int(p["size"])
            f.seek(pos)
            hdr = f.read(16)
            start = pos + (9 - hdr[0].bit_length()) + 3  # skip varint + timecode + flags
            f.seek(start)
            ok = size == w * h * 2 if codec == "yuyv" else f.read(2) == b"\xff\xd8"  # JPEG start marker
            if not ok or hdr[start - pos - 1] & 0x06:  # 0x06: laced
                sys.exit(f"{video}: unexpected packet layout at byte {pos}")
            offsets.append(start)
            sizes.append(size)
    return w, h, codec, offsets, sizes


def load_recording(folder):
    """Probe both cameras; return {cam: dict(video, w, h, codec, pts, offsets, sizes, mm)}."""
    rec = {}
    for cam in CAMS:
        video = folder / f"{cam}.mkv"
        w, h, codec, offsets, sizes = probe(video)
        with open(folder / f"{cam}_timestamps.csv", newline="") as f:
            pts = np.array([int(r["pts_ns"]) for r in csv.DictReader(f)])
        n = min(len(offsets), len(pts))
        if len(offsets) != len(pts):
            print(f"WARNING: {cam}: {len(offsets)} frames but {len(pts)} timestamps; using first {n}")
        rec[cam] = dict(video=video, w=w, h=h, codec=codec, pts=pts[:n], offsets=offsets[:n],
                        sizes=sizes[:n], mm=np.memmap(video, np.uint8, "r"))
    if rec["left"]["codec"] != rec["right"]["codec"]:
        sys.exit("left and right recordings use different formats")
    return rec


def yuyv(cam, i):
    """Raw frame i as an (h, w, 2) array: [..., 0] is luma Y, [..., 1] alternates Cb/Cr."""
    w, h, off = cam["w"], cam["h"], cam["offsets"][i]
    return cam["mm"][off:off + w * h * 2].reshape(h, w, 2)


def jpeg(cam, i, flags):
    """Decode MJPEG frame i with OpenCV; flags is cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE (luma)."""
    off = cam["offsets"][i]
    return cv2.imdecode(np.asarray(cam["mm"][off:off + cam["sizes"][i]]), flags)


def read_frame(cam, i):
    """Fast random-access BGR frame, for display only."""
    if cam["codec"] == "mjpeg":
        return jpeg(cam, i, cv2.IMREAD_COLOR)
    return cv2.cvtColor(yuyv(cam, i), cv2.COLOR_YUV2BGR_YUYV)  # OpenCV BT.601 limited range


def luma_frames(cam, wanted):
    """Yield (index, luma plane) for the wanted indices: the recorded Y for YUYV, the JPEG luma for MJPEG."""
    for i in sorted(wanted):
        yield i, (jpeg(cam, i, cv2.IMREAD_GRAYSCALE) if cam["codec"] == "mjpeg"
                  else np.ascontiguousarray(yuyv(cam, i)[..., 0]))


def colour_frames(cam, wanted):
    """Yield (index, BGR frame) for saving: ffmpeg for YUYV, OpenCV's JPEG decoder for MJPEG."""
    if cam["codec"] == "yuyv":
        yield from ffmpeg_frames(cam, wanted)
    else:
        for i in sorted(wanted):
            yield i, jpeg(cam, i, cv2.IMREAD_COLOR)


def ffmpeg_frames(cam, wanted):
    """Yield (index, BGR frame) for the wanted indices, decoded by ffmpeg.

    ffmpeg applies the stream's colour tags (BT.601, limited range) and
    interpolates the half-width chroma, so this is the path for saved frames.
    -nostdin stops ffmpeg putting the terminal in raw mode, which it would
    never undo when killed early below.
    """
    w, h, wanted = cam["w"], cam["h"], set(wanted)
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(cam["video"]), "-map", "0:v:0",
           "-fps_mode", "passthrough", "-vf", "scale=flags=accurate_rnd+full_chroma_int",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE) as proc:
        for i in range(max(wanted, default=-1) + 1):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                sys.exit(f"{cam['video']}: ffmpeg ended at frame {i}")
            if i in wanted:
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        proc.kill()


def match_left(rec, right):
    """Return (right_i, left_i, dt_ms) for each right index, using the nearest left pts."""
    lp, rp, r = rec["left"]["pts"], rec["right"]["pts"], np.asarray(right, dtype=int)
    j = np.clip(np.searchsorted(lp, rp[r]), 1, len(lp) - 1)
    l = np.where(np.abs(lp[j] - rp[r]) < np.abs(lp[j - 1] - rp[r]), j, j - 1)
    return [(int(a), int(b), (rp[a] - lp[b]) / 1e6) for a, b in zip(r, l)]


def pair_up(rec, right, max_dt_ms):
    """Matches for the right indices, warning about and dropping gaps over max_dt_ms."""
    matches = match_left(rec, right)
    for r, l, dt in matches:
        if abs(dt) > max_dt_ms:
            print(f"WARNING: skipping right {r}: nearest left {l} is {dt:+.1f} ms away")
    return [m for m in matches if abs(m[2]) <= max_dt_ms]


def file_name(k, cam, prefix, label):
    """PNG name for pair k: PREFIX_Cam_LABEL.png with --name, else the pair index."""
    return f"{prefix}_{cam.capitalize()}_{label}.png" if prefix else f"{k:04d}.png"


def write_pairs(rec, pairs, out, gray=False, prefix=None, labels=None):
    """Replace frames/ output with lossless PNG pairs plus pairs.csv.

    labels maps right frame index -> label; files are named by file_name().
    """
    labels = labels or {}
    frames = luma_frames if gray else colour_frames
    with ThreadPoolExecutor(8) as pool:
        for cam, col in (("right", 0), ("left", 1)):
            (out / cam).mkdir(parents=True, exist_ok=True)
            for p in (out / cam).glob("*.png"):
                p.unlink()
            name = {p[col]: file_name(k, cam, prefix, labels.get(p[0], f"{p[0]:04d}"))
                    for k, p in enumerate(pairs)}
            print(f"writing {cam} ({len(name)} {'gray' if gray else 'colour'} frames)...")
            jobs, ok = deque(), True  # bounded queue: don't hold every frame in memory
            for i, img in frames(rec[cam], name):
                jobs.append(pool.submit(cv2.imwrite, str(out / cam / name[i]), img))
                if len(jobs) > 16:
                    ok &= jobs.popleft().result()
            if not (ok and all(j.result() for j in jobs)):
                sys.exit(f"failed writing PNGs to {out / cam}")
    with open(out / "pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pair", "right_frame", "left_frame", "right_pts_ns", "left_pts_ns", "dt_ms",
                     "label", "right_file", "left_file"])
        for k, (r, l, dt) in enumerate(pairs):
            label = labels.get(r, "")
            wr.writerow([k, r, l, rec["right"]["pts"][r], rec["left"]["pts"][l], f"{dt:.3f}", label,
                         file_name(k, "right", prefix, label or f"{r:04d}"),
                         file_name(k, "left", prefix, label or f"{r:04d}")])
    print(f"wrote {len(pairs)} pairs to {out}")


def put_text(img, lines, x, y, color=WHITE):
    """Draw non-empty text lines on black boxes, top-left at (x, y)."""
    for text in filter(None, lines):
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y), (x + w + 10, y + h + base + 8), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 5, y + h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += h + base + 8


def render(rec, i, marks, max_dt_ms, screen, status, typing=None):
    """Matched left | right frame i, scaled to fit the screen, with overlay.

    marks maps marked right indices to their label (None without --name);
    typing is the label being entered, or None.
    """
    _, l, dt = match_left(rec, [i])[0]
    ok, marked, pts = abs(dt) <= max_dt_ms, i in marks, rec["right"]["pts"]
    left = read_frame(rec["left"], l) // (1 if ok else 3)  # dim a rejected match
    img = np.hstack([left, read_frame(rec["right"], i)])
    s = min(screen[0] / img.shape[1], screen[1] / img.shape[0], 1.0)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    if marked:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), GREEN, 6)
    put_text(img, [f"LEFT {l}  dt={dt:+.2f} ms",
                   "" if ok else f"gap > {max_dt_ms:.0f} ms: pair will be skipped"],
             10, 10, WHITE if ok else RED)
    put_text(img, [f"RIGHT {i}/{len(pts) - 1}  t={(pts[i] - pts[0]) / 1e9:.3f}s"
                   + (f"  [MARKED {marks[i]}]" if marked and marks[i] else "  [MARKED]" if marked else ""),
                   f"{len(marks)} marked", status],
             w // 2 + 10, 10, GREEN if marked else WHITE)
    if typing is not None:
        put_text(img, [f"label: {typing}_", "enter: confirm (empty = frame number)   esc: cancel"],
                 w // 2 + 10, h // 2, YELLOW)
    put_text(img, [HELP], 10, h - 40)
    return img


def window_open(win):
    """True while the window exists (Qt raises once it has been closed with X)."""
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def run_select(rec, out, max_dt_ms, screen, gray, prefix=None):
    """Interactive viewer on the right camera; Enter saves the marked pairs.

    With a prefix, marking a frame first asks for its label (typed in the window).
    """
    n, win = len(rec["right"]["pts"]), "stereo frame select"
    i, marks, saved, status, armed, typing = 0, {}, {}, "", False, None
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(win, render(rec, i, marks, max_dt_ms, screen, status))
    while (k := cv2.waitKeyEx(100)) != -1 or window_open(win):
        if k == -1:
            continue
        quit_armed, armed, status = armed, False, ""
        step = next((s for s, codes in KEYS.items() if k in codes), 0)
        if typing is not None:  # entering a label: keys edit the text instead of navigating
            if k in SAVE_KEYS:
                label = typing or f"{i:04d}"
                if label in marks.values():
                    status = f"label '{label}' already used"
                else:
                    marks[i], typing = label, None
            elif k == 27:
                typing, status = None, "mark cancelled"
            elif k in BACKSPACE_KEYS:
                typing = typing[:-1]
            elif 0 < k < 128 and chr(k) in LABEL_CHARS:
                typing += chr(k)
        elif step:
            i = int(np.clip(i + step, 0, n - 1))
        elif k == ord(" ") and i in marks:
            del marks[i]
        elif k == ord(" "):
            if prefix:
                typing = ""
            else:
                marks[i] = None
        elif k in SAVE_KEYS and not marks:
            status = "nothing marked - existing output left untouched"
        elif k in SAVE_KEYS:
            cv2.imshow(win, render(rec, i, marks, max_dt_ms, screen, "saving..."))
            cv2.waitKey(1)
            write_pairs(rec, pair_up(rec, sorted(marks), max_dt_ms), out, gray, prefix,
                        {r: lab for r, lab in marks.items() if lab})
            saved, status = dict(marks), f"saved {len(marks)} marked frames ({'gray' if gray else 'colour'})"
        elif k in QUIT_KEYS:
            if marks == saved or quit_armed:
                break
            armed, status = True, "unsaved changes - press q again to quit"
        cv2.imshow(win, render(rec, i, marks, max_dt_ms, screen, status, typing))
    if window_open(win):
        cv2.destroyAllWindows()
    if marks != saved:
        print(f"quit without saving ({len(marks)} marked)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="recording folder with left/right .mkv and _timestamps.csv")
    ap.add_argument("mode", choices=["all", "select"])
    ap.add_argument("--step", type=int, default=1, help="all mode: take every Nth right frame")
    ap.add_argument("--max-dt-ms", type=float, help="max left/right time gap (default: 1/4 frame interval)")
    ap.add_argument("--screen", type=lambda s: tuple(map(int, s.split("x"))), default=(1880, 1000),
                    help="viewer size limit WxH (default 1880x1000)")
    ap.add_argument("--overwrite", action="store_true", help="all mode: replace an existing frames/ output")
    ap.add_argument("--gray", action="store_true", help="save the luma plane (grayscale) instead of colour")
    ap.add_argument("--name", help="file name prefix: pairs are saved as NAME_Left_LABEL.png / NAME_Right_LABEL.png, "
                                   "with LABEL typed when marking (select mode) or the frame number (all mode)")
    args = ap.parse_args()
    if args.name and not set(args.name) <= LABEL_CHARS:
        sys.exit(f"--name may only contain letters, digits and - _ . (got {args.name!r})")

    rec, out = load_recording(args.folder), args.folder / "frames"
    n = len(rec["right"]["pts"])
    max_dt_ms = args.max_dt_ms
    if max_dt_ms is None:
        max_dt_ms = 0.25 * np.median(np.diff(rec["right"]["pts"])) / 1e6
    print(f"{n} right / {len(rec['left']['pts'])} left frames, max pair gap {max_dt_ms:.1f} ms")
    if args.mode == "select":
        run_select(rec, out, max_dt_ms, args.screen, args.gray, args.name)
    elif (out / "pairs.csv").exists() and not args.overwrite:
        sys.exit(f"{out} already has output; pass --overwrite to replace it")
    else:
        write_pairs(rec, pair_up(rec, range(0, n, max(args.step, 1)), max_dt_ms), out, args.gray, args.name)


if __name__ == "__main__":
    main()
