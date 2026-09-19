""" '

This script calibrates the stereo camera using OpenCV 
Camera intrinsics are outputted to X (what type of file to store it in? I want it to be readable by python and human)

I want to do camera calibration + stereo calibration

Run this from the 4022 parent folder





"""


# ====== IMPORTS ======
import os
import re
import glob
import numpy as np
import cv2
import datetime



# Calibratoin board config used: kalibr kalibr_create_target_pdf   --type apriltag --nx 10 --ny 7   --tsize 0.050 --tspace 0.3   /data/10x7.pdf
# ====== BOARD GEOMETRY ======
PATTERN = "aprilmesh"
CHECKERBOARD = (9, 6)   # Internal corners - (col,row), (9,6) for 10x7 board
TAG_SIZE = 35.36       # mm
TAG_GAP = TAG_SIZE * 0.3  # mm
PITCH = TAG_SIZE + TAG_GAP  # mm



# ====== SINGLE-IMAGE DETECTION TEST (disabled, kept for reference) ======
'''

IMAGE_PATH = "data/Calibration/36h11_aprilmesh_10x7_50mm.png"   # For testing, using single image
# Importing image
gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE) # Reads the image in grayscale
print(gray.shape)   # Outputting the shape of the image (height, width)


# Setting up detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)   # Tag family
params = cv2.aruco.DetectorParameters() # params holds the detecror parameters. 
params.markerBorderBits = 2 # Kalibr uses 2 bits for the boarders
detector = cv2.aruco.ArucoDetector(dictionary, params) # Generating detector object

# detecting markers in the image
corners, ids, rejected = detector.detectMarkers(gray)  #  Detection
# Corners are the coordinates of the detected markers
# IDs are the id of the detected markers
# Rjected are the shapes that looked like tags but didnt decide they were tags
print(len(ids), "tags found")


# ====== DISPLAY DETECTIONS ======
# Shrink the image to fit the screen
scale = min(1800 / gray.shape[1], 1000 / gray.shape[0])
small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# Draw the detected tags onimage
vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
cv2.aruco.drawDetectedMarkers(vis, [c * scale for c in corners], ids)

# Show until a key is pressed
cv2.imshow("AprilGrid detections", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
# ====== INPUT FRAMES ======
FRAMES_DIR = "data/Calibration/18_Sep/C1/frames"
CAMERAS = ("left", "right")

first_image = sorted(glob.glob(os.path.join(FRAMES_DIR, "left", "*.png")))[0]
image_size = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE).shape[::-1]   # (width, height)
print("image size:", image_size)

# ====== APRILTAG DETECTOR ======
# Detector: AprilTag 36h11
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.markerBorderBits = 2                                     # Kalibr tags have a 2-bit black border
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE    # refined below with cv2.cornerSubPix (see refine_corners)
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 53                            # big close-up tags need a wider threshold window
params.adaptiveThreshWinSizeStep = 5
params.errorCorrectionRate = 1.0                                # accept blurred tags; 36h11 codes are far apart
detector = cv2.aruco.ArucoDetector(dictionary, params)

# Corner refinement. On the 18 Sep underwater frames the built-in options were poor:
#   CORNER_REFINE_APRILTAG  accurate corners (~0.3 px) but drops ~75% of the tags
#   CORNER_REFINE_SUBPIX    keeps the tags but corners are off by ~1.5 px (OpenCV 5.0.0)
# Calling cv2.cornerSubPix directly keeps every tag and gives ~0.3 px corners.
SUBPIX_WIN = (5, 5)   # half-size of the search window, px
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

def refine_corners(gray, corners):
    """Sub-pixel refine every tag corner on the original image."""
    return tuple(cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)
                 .reshape(1, 4, 2) for c in corners)

# ====== BOARD MODEL ======
# Board: physical position (mm) of every tag corner
tag_corners = []
for tag_id in range(70):
    x, y = (tag_id % 10) * PITCH, (tag_id // 10) * PITCH   # id 0 bottom-left, ids run right then up
    # Kalibr prints tags rotated 180 deg, so OpenCV reports: bottom-right, bottom-left, top-left, top-right
    tag_corners.append(np.array([[x + TAG_SIZE, y, 0], [x, y, 0],
                                 [x, y + TAG_SIZE, 0], [x + TAG_SIZE, y + TAG_SIZE, 0]], np.float32))
board = cv2.aruco.Board(tag_corners, dictionary, np.arange(70))


# ====== DETECT TAGS IN EVERY IMAGE ======
MIN_TAGS = 10   # ignore images where too few tags were found. Useful for too blurry, ocluded or far away
SAVE_DETECTIONS = True   # save every image with its detections drawn on, to check which tags are seen

OUTPUT_DIR = "results/calibration"
recording = os.path.basename(os.path.dirname(FRAMES_DIR))            # "15 September"
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, recording.replace(" ", "_") + "_detections")


def save_detections(camera, name, gray, corners, ids, rejected):
    """Save a copy of the image with detected tags (green, with id) and rejected candidates (red)."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if rejected:
        cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids, borderColor=(0, 255, 0))
    n_tags = 0 if ids is None else len(ids)
    missing = sorted(set(range(70)) - set([] if ids is None else ids.ravel().tolist()))
    colour = (0, 255, 0) if n_tags >= MIN_TAGS else (0, 0, 255)
    lines = [f"{camera} {name}: {n_tags}/70 tags" + ("" if n_tags >= MIN_TAGS else "  SKIPPED"),
             f"{len(rejected)} rejected candidates (red)"]
    if 0 < len(missing) <= 20:
        lines.append("missing ids: " + " ".join(map(str, missing)))
    for k, text in enumerate(lines):
        cv2.putText(vis, text, (10, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    os.makedirs(os.path.join(DETECTIONS_DIR, camera), exist_ok=True)
    cv2.imwrite(os.path.join(DETECTIONS_DIR, camera, name), vis)


def detect_tags(camera):
    """Detect tags in every image of one camera.
    Returns {filename: (corners, ids)} for the usable images only."""
    image_files = sorted(glob.glob(os.path.join(FRAMES_DIR, camera, "*.png")))
    detections = {}
    for path in image_files:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        corners, ids, rejected = detector.detectMarkers(gray)
        corners = refine_corners(gray, corners)
        if SAVE_DETECTIONS:
            save_detections(camera, os.path.basename(path), gray, corners, ids, rejected)

        n_tags = 0 if ids is None else len(ids)
        if n_tags < MIN_TAGS:
            print(f"  {camera}: skip {os.path.basename(path)}: {n_tags} tags")
            continue

        detections[os.path.basename(path)] = (corners, ids)

    print(f"{camera}: {len(detections)} of {len(image_files)} images usable")
    return detections


# ====== CAMERA CALIBRATION ======
def calibrate_camera(camera, detections):
    """Calibrate one camera from its detections.
    Returns a dict: camera_matrix, dist_coeffs, rms, std_intrinsics, images, per_image_errors."""
    names = sorted(detections)
    all_obj_points, all_img_points = [], []
    for name in names:
        corners, ids = detections[name]
        obj_points, img_points = board.matchImagePoints(corners, ids)
        all_obj_points.append(obj_points)
        all_img_points.append(img_points)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view_errors = \
        cv2.calibrateCameraExtended(all_obj_points, all_img_points, image_size, None, None)

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"\n{camera.upper()} CAMERA")
    print(f"RMS reprojection error: {rms:.3f} px")
    print(f"focal length   fx = {fx:.1f} px   fy = {fy:.1f} px")
    print(f"principal point cx = {cx:.1f} px   cy = {cy:.1f} px   (image centre {image_size[0]/2:.0f}, {image_size[1]/2:.0f})")
    print("distortion (k1, k2, p1, p2, k3):", ", ".join(f"{v:.4f}" for v in dist_coeffs.ravel()))

    # Error per image, worst first
    per_view_errors = per_view_errors.ravel()
    print("error per image:")
    for name, err in sorted(zip(names, per_view_errors), key=lambda t: -t[1]):
        flag = "  <-- high" if err > 2 * np.median(per_view_errors) else ""
        print(f"  {name}: {err:.3f} px{flag}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rms": rms,
        "std_intrinsics": std_intrinsics.ravel()[:9],   # 1-sigma of fx, fy, cx, cy, k1, k2, p1, p2, k3
        "images": names,
        "per_image_errors": per_view_errors,
    }


# ====== RUN: BOTH CAMERAS ======
detections = {}
intrinsics = {}
for camera in CAMERAS:
    detections[camera] = detect_tags(camera)
    intrinsics[camera] = calibrate_camera(camera, detections[camera])

# ====== SAVE CALIBRATION (YAML) ======
output_path = os.path.join(OUTPUT_DIR, recording.replace(" ", "_") + "_stereo.yaml")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_camera(fs, camera, result):
    """Write one camera's intrinsics, uncertainty and quality as a commented YAML section."""
    K, D = result["camera_matrix"], result["dist_coeffs"].ravel()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    std = result["std_intrinsics"]
    names = ["k1", "k2", "p1", "p2", "k3"]

    fs.startWriteStruct(f"{camera}_camera", cv2.FileNode_MAP)

    fs.writeComment("---- Intrinsics: pinhole model + lens distortion ----")
    fs.writeComment("camera_matrix K = [ fx   0  cx ]")
    fs.writeComment("                  [  0  fy  cy ]")
    fs.writeComment("                  [  0   0   1 ]")
    fs.writeComment(f"fx = {fx:.2f} px, fy = {fy:.2f} px (focal length); "
                    f"cx = {cx:.2f} px, cy = {cy:.2f} px (principal point)")
    fs.write("camera_matrix", K)

    fs.writeComment("distortion_coefficients = [k1, k2, p1, p2, k3] (OpenCV standard model)")
    fs.writeComment("k1, k2, k3: radial distortion (negative k1 = barrel); p1, p2: tangential (lens tilt)")
    fs.writeComment(", ".join(f"{n} = {v:.5f}" for n, v in zip(names, D)))
    fs.write("distortion_coefficients", result["dist_coeffs"])

    fs.writeComment("Readable copies of the values above")
    fs.write("focal_length_x_px", fx)
    fs.write("focal_length_y_px", fy)
    fs.write("principal_point_x_px", cx)
    fs.write("principal_point_y_px", cy)
    fs.write("field_of_view_horizontal_deg", np.degrees(2 * np.arctan(image_size[0] / (2 * fx))))
    fs.write("field_of_view_vertical_deg", np.degrees(2 * np.arctan(image_size[1] / (2 * fy))))

    fs.writeComment("---- Uncertainty: 1 standard deviation from the calibration fit ----")
    fs.writeComment(f"fx +- {std[0]:.2f} px, fy +- {std[1]:.2f} px, cx +- {std[2]:.2f} px, cy +- {std[3]:.2f} px")
    fs.startWriteStruct("uncertainty_1sigma", cv2.FileNode_MAP)
    for key, value in zip(["fx_px", "fy_px", "cx_px", "cy_px"] + names, std):
        fs.write(key, value)
    fs.endWriteStruct()

    fs.writeComment("---- Quality ----")
    fs.writeComment("RMS distance between detected corners and where the model puts them; below ~0.5 px is good")
    fs.writeComment(f"rms = {result['rms']:.3f} px over {len(result['images'])} images")
    fs.write("rms_reprojection_error_px", result["rms"])
    fs.write("images_used", len(result["images"]))
    fs.writeComment("Per-image RMS error (px), worst first:")
    for name, err in sorted(zip(result["images"], result["per_image_errors"]), key=lambda t: -t[1]):
        fs.writeComment(f"  {name}  {err:.3f}")
    fs.writeComment("Same data as lists: images[i] has error per_image_error_px[i]")
    fs.startWriteStruct("images", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for name in result["images"]:
        fs.write("", name)
    fs.endWriteStruct()
    fs.startWriteStruct("per_image_error_px", cv2.FileNode_SEQ | cv2.FileNode_FLOW)
    for err in result["per_image_errors"]:
        fs.write("", err)
    fs.endWriteStruct()

    fs.endWriteStruct()


fs = cv2.FileStorage(output_path, cv2.FILE_STORAGE_WRITE)
fs.writeComment("=====================================================================")
fs.writeComment("STEREO CAMERA CALIBRATION")
fs.writeComment("Generated by src/01_Calibration/Calibration.py - re-run the script rather than editing by hand.")
fs.writeComment("Units: px = pixels (image quantities), mm = millimetres (board sizes).")
fs.writeComment("Numbers are stored at full precision; comments show rounded values for reading.")
fs.writeComment("Load in Python:")
fs.writeComment('  fs = cv2.FileStorage("' + output_path + '", cv2.FILE_STORAGE_READ)')
fs.writeComment('  K_left = fs.getNode("left_camera").getNode("camera_matrix").mat()')
fs.writeComment("=====================================================================")

fs.startWriteStruct("metadata", cv2.FileNode_MAP)
fs.write("created", datetime.datetime.now().isoformat(timespec="seconds"))
fs.write("script", "src/01_Calibration/Calibration.py")
fs.write("opencv_version", cv2.__version__)
fs.write("frames_dir", FRAMES_DIR)
fs.endWriteStruct()

fs.writeComment("Image size the calibration was made at; it is only valid at this resolution")
fs.startWriteStruct("image", cv2.FileNode_MAP)
fs.write("width_px", image_size[0])
fs.write("height_px", image_size[1])
fs.endWriteStruct()

fs.writeComment("Calibration target: Kalibr AprilGrid (kalibr_create_target_pdf --type apriltag --nx 10 --ny 7)")
fs.startWriteStruct("calibration_board", cv2.FileNode_MAP)
fs.write("type", "Kalibr AprilGrid")
fs.write("tag_family", "AprilTag 36h11")
fs.write("tags_x", 10)
fs.write("tags_y", 7)
fs.writeComment(f"tag size {TAG_SIZE:.2f} mm (outer black square, as printed), gap {TAG_GAP:.2f} mm")
fs.write("tag_size_mm", TAG_SIZE)
fs.write("tag_gap_mm", TAG_GAP)
fs.write("corner_refinement", "APRILTAG")
fs.write("min_tags_per_image", MIN_TAGS)
fs.endWriteStruct()

for camera in CAMERAS:
    write_camera(fs, camera, intrinsics[camera])

fs.writeComment("Stereo extrinsics (rotation R, translation T = baseline) - not computed yet")
fs.startWriteStruct("stereo", cv2.FileNode_MAP)
fs.write("status", "not computed")
fs.endWriteStruct()

fs.release()
print(f"\nsaved calibration to {output_path}")


"""
================================================================================
OPTIONS FOR IMPROVING CALIBRATION QUALITY  -- CLAUDE FEEDBACK
================================================================================

------------------------------ A. CAPTURE / PROCESS ----------------------------
A1. Number and variety of views. Use 20-40 views per camera, not more of the
    same. What matters is variety: tilt the board +-30-45 deg about both axes,
    vary the distance, and rotate it in plane. A stack of fronto-parallel views
    leaves focal length and distortion poorly determined.
A2. Cover the whole image, especially the corners. Distortion is strongest at
    the edges, so views where the board sits near or crosses the frame border
    are what constrain it. Gaps in coverage are where undistortion goes wrong.
A3. Fill the frame. A board covering a small part of the image contributes
    little; get it close enough that it spans most of the frame in some views.
A4. Avoid motion blur. The rig records at 5 fps, so a moving board smears.
    Hold each pose still, or select sharp frames (variance of the Laplacian is
    a simple sharpness score) using the select mode of extract_stereo_frames.py.
A5. Both cameras must see the board. Stereo calibration only uses views where
    the board is detected in both, so aim the board at the overlap region.

------------------------------ B. IMAGE PIPELINE -------------------------------
B1. Use the recorded luma (extract_stereo_frames.py --gray). It is exactly what
    the sensor produced; grayscale rebuilt from colour differs by up to 39 grey
    levels where an RGB channel clips (measured on the 26 Aug recording).
B2. Never use JPEG for calibration frames. Measured on this rig, JPEG q95
    already shifts corners by ~0.02 px and raises reprojection error; q75 is
    clearly worse. PNG is lossless and costs only disk space.
B3. Do not resize or crop before detection. Calibration parameters are in
    pixels of the original image, and resampling both moves and blurs corners.
B4. Sub-pixel corner refinement. params.cornerRefinementMethod =
    CORNER_REFINE_SUBPIX (or CORNER_REFINE_APRILTAG). Tune
    cornerRefinementWinSize to the tag size in pixels: too large a window
    crosses neighbouring features, too small is noise-sensitive.
B5. If tags are missed, tune the detector rather than accepting fewer points:
    adaptiveThreshWinSizeMin/Max/Step for uneven lighting, and
    minMarkerPerimeterRate when the board is far away and tags are small.

------------------------------ C. THE FIT ITSELF -------------------------------
C1. Distortion model. The lenses are wide (~96 deg horizontal), so compare the
    standard 5-coefficient model, CALIB_RATIONAL_MODEL (8 coefficients) and the
    fisheye model, and choose on held-out error, not on in-sample RMS. In water
    a flat port adds refraction, which no pinhole distortion model describes
    exactly - see E3.
C2. Judge the fit on held-out data. In-sample RMS always improves with more
    parameters. Hold out whole views (k-fold, blocked in time so near-duplicate
    frames do not leak) and compare reprojection error on unseen views. See
    src/analysis/gray_vs_rgb_calibration.py for an implementation.
C3. Inspect per-view error. cv2.calibrateCameraExtended returns perViewErrors;
    a view far above the rest usually means blur or a misdetection. Drop it and
    refit, but never drop views just because they are the hardest.
C4. Report uncertainty, not a single number. Bootstrap over views to get the
    spread of focal length, principal point and baseline. A parameter whose
    bootstrap spread is large is not pinned down by the data you collected.
C5. Constrain the model when the data is weak. CALIB_FIX_ASPECT_RATIO or
    CALIB_FIX_PRINCIPAL_POINT reduce the number of free parameters; a well
    covered dataset should not need them.
C6. Stereo: calibrate each camera first, then run stereoCalibrate with
    CALIB_FIX_INTRINSIC so that a bad view cannot corrupt the intrinsics, and
    check the rectified epipolar error (matched corners should share a row).
C7. Sanity-check against physics: focal length in px should match the lens
    spec, the principal point should be near the image centre, and the baseline
    should match the measured distance between the lenses.

------------------------------ D. BOARD HARDWARE -------------------------------
D1. Flatness is the biggest board error. Mount the print on a rigid flat
    backing (aluminium composite, foam PVC, glass). Paper taped to a wall or
    held by hand bows by millimetres, and the fit absorbs that as distortion.
D2. Measure the printed tag, do not trust the PDF. Printers scale. Measure
    several tags with calipers, average, and put that in TAG_SIZE. A 1% scale
    error puts a 1% error straight into the baseline and all reconstruction.
D3. Matte finish, no glare. A glossy print reflects lights, and a blown-out
    highlight destroys the corners under it. Matte laminate also waterproofs.
D4. Board size to suit the working distance. Underwater, the board must still
    fill a good part of the frame at the stand-off distance you will use.
D5. Keep it clean and dry, and re-measure if it has been rolled or soaked.

------------------------------ E. CAMERA / RIG HARDWARE ------------------------
E1. Lock every automatic setting: fixed focus, manual exposure, manual gain,
    manual white balance. Autofocus changes the focal length between views,
    which invalidates a single intrinsic model.
E2. Expose so white does not clip. On the 26 Aug recording ~7% of pixels had a
    clipped RGB channel, concentrated on the board's white paper. Clipping
    flattens the very edges corners are measured from. Expose for the
    highlights, even if the image looks dark.
E3. Port choice for underwater. A flat port refracts and behaves like an extra
    lens whose effect depends on distance; a dome port, correctly centred on
    the entrance pupil, largely removes it. With a flat port, calibrate in
    water and use a refractive model (Pinax, already in Correction/) rather
    than assuming pinhole plus radial distortion.
E4. Calibrate in the medium you will use. Air calibration does not transfer to
    water: refraction changes the effective focal length and distortion.
E5. Rigid stereo mounting. Any flex between the cameras changes the baseline
    and rotation, so the calibration expires. Use a stiff bar, check it after
    transport, and recalibrate after anything mechanical changes.
E6. Hardware sync (a shared trigger) beats software timestamps. The current
    left/right offset is under 5 ms, which is fine for a static board but not
    for a moving rig under ice.
E7. Global shutter, or keep everything still. A rolling shutter skews moving
    objects and biases corner positions.
E8. Even, diffuse lighting. Avoid a single hard light, which gives glare on one
    side and noise in the shadows. Underwater, keep lights off-axis to reduce
    backscatter.
E9. Thermal and mechanical stability: let the cameras warm up before
    calibrating, since focus and mounting shift slightly as they heat.
================================================================================
"""
