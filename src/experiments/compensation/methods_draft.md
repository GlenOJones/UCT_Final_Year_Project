# Methods draft: comparison of underwater calibration methods

*Draft for the thesis. Numbers are from `results/experiments/compensation/water_final/`. Items in [brackets] still need filling in.*

## Data

All four techniques are scored on the same underwater observations of a Kalibr AprilGrid (10 × 7 AprilTag 36h11
tags; tag edge 35.36 mm, measured with callipers on the print; gap 10.61 mm). The board was filmed in
[fresh tank / sea water, temperature, salinity] by both StellarHD cameras at 1600 × 1200 px, MJPG. Ten
recordings were made, each a sweep of the board along the rail from about 0.23 m to 1.45 m at one fixed board
tilt between 8° and 49°. Eleven or twelve frames were extracted from each recording, 115 per camera.

Tags were detected and refined by the project's calibration pipeline (`Calibration.py`): OpenCV
`ArucoDetector` with the `water_final` preset on the grey image, corners refined with `cv2.cornerSubPix` in a
window of 15 % of the tag side, images with fewer than 10 tags discarded, then an iterative outlier rejection
that drops tags whose worst corner misses by more than max(2 px, 5 × median) and images whose corner RMS exceeds
max(1 px, 3 × median). The two steepest recordings (C7, L7) were rejected entirely. The comparison starts from
exactly the images and tags this pipeline kept: 84 left images (18 772 corners) and 80 right images (18 436
corners) from 8 recordings, and 77 stereo pairs with at least 10 tags seen by both cameras.

Technique 4 also needs an in-air calibration of the same rig. The board was filmed in air in nine recordings
(`data/Calibration/FinalComparison/Air`, 106 frames per camera) and calibrated by the same pipeline and preset,
keeping 59 left and 65 right images: $f_x = 718.6$ and $719.2$ px, RMS 0.214 and 0.228 px, stereo RMS 0.284 px.

## Techniques

Techniques 1–3 are fitted by Levenberg–Marquardt minimisation of the pixel reprojection error over the
intrinsic parameters and one board pose per image. They differ only in the projection model or in the training
data. Technique 4 fits nothing underwater.

**1. Brown–Conrady (baseline).** A pinhole camera with the Brown–Conrady polynomial distortion, as in
Zhang's method: $f_x, f_y, c_x, c_y$, radial $k_1, k_2, k_3$ and tangential $p_1, p_2$ (9 intrinsic
parameters). Fitted with `cv2.calibrateCamera` (flags 0), which initialises from Zhang's closed-form solution
from the board homographies.

**2. Kannala–Brandt (wide-angle model).** The distortion is a polynomial in the angle of incidence $\theta$,
$\theta_d = \theta(1 + k_1\theta^2 + k_2\theta^4 + k_3\theta^6 + k_4\theta^8)$, with $f_x, f_y, c_x, c_y$ and no
tangential terms (8 intrinsic parameters). Fitted with `cv2.fisheye.calibrate` with
`CALIB_RECOMPUTE_EXTRINSIC` and `CALIB_FIX_SKEW`. Its initialisation is different: the principal point starts
at the image centre, the focal length is estimated by least squares from the homographies, and the distortion
starts at zero.

**3. Distance transfer.** Brown–Conrady as in technique 1, but trained only on frames with the board 0.2–0.8 m
from the camera (38 left and 31 right frames when all recordings are used), then scored at every distance and
in particular at 1.0–1.5 m. A camera that behaves as a single-viewpoint (pinhole) camera should transfer
from near to far without loss. Refraction the model cannot absorb would make the error grow with distance.

**4. Pinax (flat-port physics).** The model of Łuczyński, Pfingsthorn and Birk (2017) for a camera behind a
flat port: the in-air Brown–Conrady calibration maps a pixel to its ray in air, and Snell's law at a flat
interface, $\sin\theta_\text{air} = n_w \sin\theta_\text{water}$, maps that ray into water ($n_w = 1.333$); the
glass index cancels out of the direction. Projection runs the chain backwards. No intrinsic parameter is fitted
to the underwater data; the 9 in-air parameters are used as calibrated. The rays are taken to leave from Pinax's
single virtual centre on the optical axis. Its position (and how far real rays miss it) depends on the
lens-to-port distance and the port thickness, which have no physical value for a water-contact lens; the
position is a constant shift along the axis that every board pose absorbs, and the miss distance is not
modelled. The test asks whether the Aquagon behaves as an in-air lens behind a flat port.

A plain rescaling of an in-air calibration by the refractive index (the "1.333 rule") was not tested
separately: Pinax contains it as its paraxial limit and adds the full angle dependence of refraction, so it is
the stronger form of the same idea. The in-air and in-water focal lengths are 718.6/719.2 px and 952.7/956.0 px,
a ratio of 1.326–1.329, close to the refractive index of water, so near the optical axis the flat-port picture
holds.

## Evaluation protocol

**Held-out recordings.** Frames from one recording are near-duplicates, so whole recordings are held out. Every
choice of 2 of the 8 recordings is used (28 splits, each recording held out 7 times). Each technique is
fitted on the other 6 recordings and scored on the 2 held out.

**Reprojection error.** Each held-out frame is posed with the technique's intrinsics fixed by minimising the
pixel reprojection error: `cv2.solvePnP` (iterative) for Brown–Conrady, and `cv2.fisheye.solvePnP` followed by
a pixel-space Levenberg–Marquardt refinement for Kannala–Brandt. The refinement is needed because
`cv2.fisheye.solvePnP` minimises the error in undistorted normalised coordinates, which would weight
points differently from the Brown–Conrady pose. For Pinax the start is `cv2.solvePnP` on the refracted
normalised rays, followed by the same pixel-space refinement. The residual of each corner is its detected
position minus its reprojection.

**Metrics.** From the held-out residuals of each split:
- overall RMS reprojection error;
- RMS in three zones by radius $r$ from the principal point, normalised so that $r = 1$ at the farthest image
  corner: centre $r < 0.33$, mid $0.33 \le r < 0.66$, outer $r \ge 0.66$. Also the ratios outer/centre and
  mid/centre;
- RMS against $r$ in bins of 0.05, and against board distance in bins of 100 mm;
- RMS in the near (0.2–0.8 m) and far (1.0–1.5 m) bands and their ratio. The ratio is also computed within
  the centre zone only, because far boards cover only the middle of the image;
- an 8 × 6 image grid coloured by mean residual length with the mean residual vector overlaid (cells with
  fewer than 5 points greyed out).

For Pinax, the underwater ray angle each model assigns to the same pixel is also compared directly against the
in-water Brown–Conrady calibration (all recordings), with no board pose involved, at pixels within 30 px of an
observed corner.

**Scale.** The rotation and translation between the cameras are fitted on the training pairs with both
cameras' intrinsics fixed: `cv2.stereoCalibrate` with `CALIB_FIX_INTRINSIC` for Brown–Conrady. In OpenCV
5.0.0, `cv2.fisheye.stereoCalibrate` fails when the pairs have different numbers of points, so for
Kannala–Brandt (and for Pinax) the same pixel-error minimisation was implemented directly. On Brown–Conrady it reproduces
`cv2.stereoCalibrate` to $10^{-7}$ px. Board corners of each held-out pair are undistorted to normalised
coordinates with the technique's model and triangulated (`cv2.triangulatePoints`). Every pair of corners whose
true separation is 250–450 mm serves as a scale bar, and its triangulated length is compared with the board
geometry. Each stereo pair is summarised by its mean (signed) and RMS length error. The bars within one pair
share a board pose and are not independent.

**Intervals and noise.** Each metric is reported as its mean over the 28 splits with the 2.5–97.5 percentile
range of the per-split values. Two techniques are compared per split on identical training and held-out data.
Their difference counts as within noise when the percentile range of the per-split difference contains zero.

## Limitations

1. **The bottom 29 % of the image is never observed.** The board stood on the rail with its lower edge at
   about $v = 856$ px in every frame, so no corner below that row was detected in any of the 230 images. Every
   model extrapolates its distortion there, and the outer zone is measured only along the top and sides.
2. **The distance range is 0.23–1.45 m.** The planned 2–3 m validation was not possible. The AprilGrid sweeps
   end at about 1.45 m. The separate 145 mm reference tags (`WaterComparison`, filmed at 0.5–2.5 m) were
   detected only at 0.5 and 1.0 m, plus one tag at 1.5 m with the merged channels. The distance-transfer test
   therefore covers about a factor of two in range. It cannot speak for the working distances of later
   surveys.
3. **The scale bars are the calibration board itself.** An error in the measured tag size scales every model's
   lengths equally and cannot be detected here. Length bias is relative to the board as measured.
4. **Held-out reprojection error is a lower bound.** Each held-out frame's pose (6 parameters) is fitted to the
   corners it is scored on and absorbs part of any model error. The triangulated lengths use no board pose and
   do not share this bias.
5. **The intervals are not formal confidence intervals.** The 28 splits share training data and are not
   independent resamples. The percentile range describes how much a result depends on which recordings were
   held out.
6. **Tilt and recording are confounded.** Each recording has one tilt, so a held-out recording is also an
   unseen tilt. Frames in the steepest recordings (C7, L7) were rejected, so tilts above about 49° are not
   tested.
7. **The close-range technique trains on fewer frames** (about 40 % of the others). A loss in transfer would
   mix the effect of distance with the effect of less data. No loss was found, so this does not change the
   conclusion here.
8. **One water condition and one session.** The freshwater-to-seawater alternative was not tested. The
   detector preset (`water_final`) is not yet the adopted one, and the comparison should be rerun once it is
   final.
9. **Capture.** The cameras are not hardware-synchronised and record MJPG. Stereo pairs are matched by
   timestamp, so any board motion between the two exposures enters the triangulated lengths.
10. **Pinax depends on inputs not measured here.** The water's refractive index is assumed (1.333, fresh; sea
    water is about 1.339), and the in-air calibration itself varies with the detector preset ($f_x$ 712.9 vs
    718.6 px left between `wide` and `water_final`). Both were varied: held-out RMS stays at 1.0–2.3 px against
    0.27–0.29 px for the fitted models, so the conclusion holds, but the exact Pinax numbers do not. The in-air
    frames come from the 5 October air session (`Air_Oct5`); Pinax assumes the lenses and focus were unchanged
    between that session and the underwater one [confirm].
11. **Pinax in its far-field form.** The spread of real rays around the virtual centre is not modelled, since
    the lens-to-port distance and port thickness have no physical value for a water-contact lens. This error
    is largest at close range and would add to, not explain away, Pinax's error.
