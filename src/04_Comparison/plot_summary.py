"""Accuracy against coverage for every reconstruction of a session, from its summary.csv.

Each method folder is one point: how much of the CAD surface it covers (x) against its median
absolute error (y). Variants that only differ by the voting threshold (<method>_v<k>, points kept
if at least k frames saw them) are joined into one curve, which is the trade-off the threshold
controls: stricter voting, fewer but more accurate points.

    src/venv/bin/python src/04_Comparison/plot_summary.py --session Sep24
    src/venv/bin/python src/04_Comparison/plot_summary.py --session Sep24 --fit on_board

Writes results/<session>/accuracy_vs_coverage_<fit>.png.
"""
import argparse
import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import project_paths as paths  # noqa: E402

# Colour = method (fixed order, never cycled); line style = scan. Methods shown by default are the
# main ones; one-off experiments (other RAFT weights, iteration counts) only with --all, in grey.
MAIN_METHODS = ["sgbm", "colmap", "colmap_hq", "raft", "raft_s075"]
COLOURS = ["#8a6d3b", "#2a78d6", "#945ecf", "#e8743b", "#19a979"]
OTHER = "#b5b4af"
LINESTYLES = ["--", "-", ":", "-."]
SURFACE, INK, MUTED = "#fcfcfb", "#2b2b2a", "#8a8a86"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Plot accuracy against coverage from a session's summary.csv.")
    ap.add_argument("--session", required=True)
    ap.add_argument("--fit", default="on_local_board", choices=("on_board", "on_local_board", "shape"))
    ap.add_argument("--all", action="store_true", help="also show the one-off experiments, in grey")
    ap.add_argument("--metric", default="median_abs_mm", choices=("median_abs_mm", "rms_mm", "p95_abs_mm"))
    return ap.parse_args(argv)


def main():
    args = parse_args()
    with open(paths.summary_path(args.session), newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["fit"] == args.fit and r["reliable"] == "True"]
    # Family = scan + method without its voting suffix; k = the voting threshold (None if absent).
    families = {}
    for r in rows:
        match = re.fullmatch(r"(.+)_v(\d+)", r["method"])
        base, k = (match.group(1), int(match.group(2))) if match else (r["method"], None)
        families.setdefault((r["scan"], base), []).append((k, float(r["coverage"]), float(r[args.metric])))

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    scans = sorted({scan for scan, _ in families})
    for (scan, base), points in sorted(families.items(), key=lambda kv: (kv[0][1] not in MAIN_METHODS, kv[0])):
        if base not in MAIN_METHODS and not args.all:
            continue
        points.sort(key=lambda p: (p[0] is not None, p[0] or 0))
        colour = COLOURS[MAIN_METHODS.index(base)] if base in MAIN_METHODS else OTHER
        style = LINESTYLES[scans.index(scan) % len(LINESTYLES)]
        xs, ys = [100 * p[1] for p in points], [p[2] for p in points]
        ax.plot(xs, ys, style, marker="o", color=colour, lw=2, ms=8, mec=SURFACE, mew=2, label=f"{base}  ({scan})")
        for k, x, y in zip((p[0] for p in points), xs, ys):
            if k is not None:
                ax.annotate(f"≥{k}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7, color=MUTED)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.set_xlabel("coverage of the CAD surface (%)")
    ax.set_ylabel({"median_abs_mm": "median absolute error (mm)", "rms_mm": "RMS error (mm)",
                   "p95_abs_mm": "95th percentile absolute error (mm)"}[args.metric])
    ax.set_title(f"{args.session}: accuracy against coverage ({args.fit} fit). Better is right and down; "
                 f"≥k = points seen from at least k frames", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    out = os.path.join(paths.RESULTS_DIR, args.session, f"accuracy_vs_coverage_{args.fit}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {paths.rel(out)}")


if __name__ == "__main__":
    main()
