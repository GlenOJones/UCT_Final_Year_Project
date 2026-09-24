"""Where every input and output of the project lives. All scripts take their paths from here, so the
layout is defined once and cannot drift.

    data/<kind>/<session>/<scan>/            a recording: left.mkv, right.mkv, *_timestamps.csv
                                 frames/     extracted pairs: left/, right/, pairs.csv (inputs)
    results/<session>/
        calibration/<scan>_<preset>_stereo.yaml, <scan>_<preset>_detections/
        <scan>/
            README.md                        what the scan is and how its results were made
            poses/tag_poses.yaml             rig pose per frame, tag layout, board frame
            <method>/                        sgbm, colmap, ...: one folder per reconstruction method
                cloud.ply                    fused cloud, board frame, mm
                cloud_clean.ply              cropped + outliers removed (use for comparison)
                mesh.ply                     Poisson mesh of cloud_clean
                *.png                        renders of the above
            comparison/<method>/             against the CAD model: metrics.json, comparison.png,
                                             distances.ply, cad_aligned.ply
        summary.csv                          every scan x method x fit compared with CAD
    results/analysis/<name>/                 studies that span sessions (detector sweep, ...)
    work/<session>/<scan>/<method>/          bulky intermediate files (COLMAP workspaces): safe to
                                             delete once the results above exist

A session is a day of recording (Sep24, 18_Sep); a scan is one recording within it (mjpg_pyr2).
"""
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WORK_DIR = os.path.join(PROJECT_ROOT, "work")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# Fixed file names inside a method folder: the folder says which scan and method, the name says
# which product.
CLOUD = "cloud.ply"
CLOUD_CLEAN = "cloud_clean.ply"
MESH = "mesh.ply"
SPARSE = "sparse.ply"


def rel(path):
    """Project-relative form of a path, for messages and for what is recorded in output files."""
    return os.path.relpath(path, PROJECT_ROOT)


# ====== INPUTS ======
def recording_dir(session, scan):
    """data/<kind>/<session>/<scan>, whichever kind of data (3DRecon, Calibration) it is under."""
    matches = [p for p in glob.glob(os.path.join(DATA_DIR, "*", session, scan)) if os.path.isdir(p)]
    if len(matches) != 1:
        found = "none" if not matches else ", ".join(rel(m) for m in matches)
        raise SystemExit(f"expected one recording data/*/{session}/{scan}, found {found}")
    return matches[0]


def frames_dir(session, scan):
    """Extracted frame pairs of a scan: <recording>/frames/{left,right}/ (extract_stereo_frames.py's
    default output)."""
    return os.path.join(recording_dir(session, scan), "frames")


# ====== RESULTS ======
def calibration_dir(session):
    return os.path.join(RESULTS_DIR, session, "calibration")


def default_calibration(session):
    """The session's calibration YAML, if there is exactly one; otherwise ask for --calibration."""
    found = sorted(glob.glob(os.path.join(calibration_dir(session), "*_stereo.yaml")))
    if len(found) != 1:
        listing = "none" if not found else ", ".join(os.path.basename(f) for f in found)
        raise SystemExit(f"{rel(calibration_dir(session))} has {len(found)} calibrations ({listing}); "
                         f"pass --calibration")
    return found[0]


def scan_dir(session, scan):
    return os.path.join(RESULTS_DIR, session, scan)


def poses_path(session, scan):
    return os.path.join(scan_dir(session, scan), "poses", "tag_poses.yaml")


def method_dir(session, scan, method):
    return os.path.join(scan_dir(session, scan), method)


def comparison_dir(session, scan, method):
    return os.path.join(scan_dir(session, scan), "comparison", method)


def summary_path(session):
    return os.path.join(RESULTS_DIR, session, "summary.csv")


def work_dir(session, scan, method):
    return os.path.join(WORK_DIR, session, scan, method)


def locate(path):
    """(session, scan, method) of a file under results/<session>/<scan>/<method>/, else None.
    Lets a script given just a cloud path find the scan's poses and where its outputs go."""
    parts = os.path.relpath(os.path.abspath(path), RESULTS_DIR).split(os.sep)
    if len(parts) >= 4 and not parts[0].startswith("..") and parts[0] != "analysis":
        return parts[0], parts[1], parts[2]
    return None


def add_scan_arguments(parser, method=None):
    """--session and --scan (and --method, if the script works on one method's output)."""
    parser.add_argument("--session", help="recording session, e.g. Sep24")
    parser.add_argument("--scan", help="scan within the session, e.g. mjpg_pyr_lights_2")
    if method is not None:
        parser.add_argument("--method", default=method,
                            help=f"reconstruction method folder, e.g. sgbm or colmap (default {method})")


def require_scan(args, parser):
    if not (args.session and args.scan):
        parser.error("--session and --scan are required")


def add_cloud_arguments(parser, method="sgbm", default_file=CLOUD):
    """For scripts that work on one saved cloud: pick it by --session/--scan/--method (and --file),
    or give its path with --cloud. --poses is only needed for a cloud outside results/."""
    add_scan_arguments(parser, method=method)
    parser.add_argument("--file", default=default_file,
                        help=f"which file in the method folder (default {default_file})")
    parser.add_argument("--cloud", help="a cloud by path instead of --session/--scan/--method/--file")
    parser.add_argument("--poses", help="tag_poses.yaml for a --cloud outside results/<session>/<scan>/")


def resolve_cloud(args, parser):
    """(cloud path, poses path) from the arguments add_cloud_arguments defines."""
    if args.cloud:
        where = locate(args.cloud)
        if args.poses:
            return args.cloud, args.poses
        if where is None:
            parser.error(f"{args.cloud} is not under results/<session>/<scan>/: pass --poses")
        return args.cloud, poses_path(where[0], where[1])
    require_scan(args, parser)
    return os.path.join(method_dir(args.session, args.scan, args.method), args.file), poses_path(args.session, args.scan)
