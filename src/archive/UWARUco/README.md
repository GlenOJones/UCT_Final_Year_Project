# UWARUco

A re-implementation of the underwater square-marker detector of

> J. Čejka, F. Bruno, D. Skarlatos, F. Liarokapis, **"Detecting Square Markers in Underwater
> Environments"**, *Remote Sensing* 11(4):459, 2019. [doi:10.3390/rs11040459](https://doi.org/10.3390/rs11040459)

built so that it can be dropped into this project's calibration in place of OpenCV's
`cv2.aruco.ArucoDetector`, and so that the two can be compared on the thing that actually matters
here: the calibration that comes out the other end.

```
src/venv/bin/python src/01_Calibration/Calibration.py --detector uwaruco --preset tuned
src/venv/bin/python src/01_Calibration/compare_calibrations.py --run wide uwaruco:tuned
src/venv/bin/python src/UWARUco/uwaruco.py            # tag counts only, no calibration
```

## Why a re-implementation and not a set of parameters

The paper's central change is an **ordering** change, not a parameter change, and OpenCV exposes no
switch for it.

Standard ARUco thresholds the image at three window sizes, finds contours in each, **merges the
contours**, and only then identifies each marker — by warping the *original grey-scale* image and
thresholding it again with Otsu's method. So the image that finds a marker and the image that reads
its code are two different images, produced by two independent decisions about what is black.

UWARUco moves the Identify step **inside** each thresholding pass, reads the code from that pass's
own binary image, and merges **markers** rather than contours. A marker that survives the threshold
is then identified on the same evidence that found it.

```
ARUco      Threshold(3,7)  -> Contours -.
           Threshold(13,7) -> Contours --> Merge Contours -> Identify (Otsu on grey) -> Pose
           Threshold(23,7) -> Contours -'

UWARUco    Threshold(10,0) -> Contours -> Identify -.
(Base)     Threshold(20,0) -> Contours -> Identify --> Merge Markers -> Pose
           Threshold(40,0) -> Contours -> Identify -'

UWARUco    Compute Mask -\
(Masked)   Threshold(10,0) -> Mask -> 3x3 median -> Contours -> Identify -.
           Threshold(20,0) -> Mask -> 3x3 median -> Contours -> Identify --> Merge -> Pose
           Threshold(40,0) -> Mask -> 3x3 median -> Contours -> Identify -'
                                                                    \-> Mask Feedback
```

Both versions are in [uwaruco.py](uwaruco.py) as `BaseDetector` and `MaskedDetector`, each with the
same `detectMarkers(gray) -> (corners, ids, rejected)` interface as `ArucoDetector`.

The other two pieces of the paper are implemented as described:

* **Algorithm 1, the mask.** 4×4 block minima and maxima, each spread over a 3×3 neighbourhood of
  blocks, give a brightness mask (`maxs >= brightness_threshold`, the white parts of markers) and a
  noise mask (`maxs - mins >= noise_threshold`, strong local edges). Their AND is applied to each
  thresholded image. The 3×3 spread is load-bearing: without it the mask cuts a marker's border
  apart at exactly the blocks that are all-white or all-black.
* **Algorithm 2, the feedback.** Both thresholds are 80% of the minimum value seen on the contour of
  a marker that was actually found, i.e. the weakest evidence the detector still managed to use.
  Nothing found means thresholds of zero, which disables masking.

## What was deliberately changed, and why

Two deviations. Both are switchable, and both are recorded in the calibration YAML of any run that
uses them.

**1. Mask feedback within one image instead of across two.** The paper detects in video, so frame
*n*'s mask comes from frame *n−1*'s markers. Calibration frames are not video — they are stills
chosen to be as far apart in pose as possible — so a threshold carried from the previous one is a
threshold from an unrelated view. `MaskedDetector` therefore defaults to `_feedback: "self"`: an
unmasked first pass, Algorithm 2 on what it found, then a second pass with the mask, keeping
whichever pass found more tags. `_feedback: "sequential"` gives the paper's behaviour, and is only
meaningful on frames that really are consecutive.

**2. `identifyFrom`, which image the bits are read from.** `"threshold"` is the paper's answer.
`"otsu"` keeps the paper's re-ordering but reads the bits ARUco's way. On this project's footage
`"otsu"` is decisively better, and the measurements below are the reason it is the default in the
recommended preset. The paper's argument for dropping Otsu is that it is the step the image-improving
filters were really helping — an argument that the step is *redundant* once the threshold is right,
not that it is wrong. On 1280×720 frames whose whole intensity range is about 119–236, the marker
interior thresholded against a broad local mean loses its bits, while Otsu over the warped marker
alone still separates them because it rescales to that marker's own contrast.

## What the paper's numbers do on this footage

Measured on every fifth frame of `data/Calibration/18_Sep/Rigframes` (42 images), against the
project's existing `wide` ArUco preset. "usable" is images with at least 10 tags, which is the bar
`Calibration.py` sets for an image to enter the fit.

| detector | usable | tags | median tags |
|---|---|---|---|
| `aruco/wide` (the incumbent) | 17/42 | 808 | 4 |
| UWARUco Base, paper's `adaptiveThreshConstant = 0` | 14/42 | 516 | 3 |
| UWARUco Masked, paper's `adaptiveThreshConstant = 0` | 15/42 | 545 | 5 |
| UWARUco Masked, `constant = 4`, bits from threshold | 16/42 | 594 | 5 |
| **UWARUco Masked, `constant = 4`, bits from Otsu** | **22/42** | **926** | **11** |

Two findings, and they point in opposite directions.

**The paper's parameter choice does not transfer.** `adaptiveThreshConstant = 0` — the paper's
headline setting, which keeps a dim marker's border connected — collapses detection here, and
`detector_presets.json` had already recorded the same effect for the ArUco detector (403 tags down
to 43). The reason is that these tags are **low-contrast rather than dark**: against a local mean
that is itself close to the tag's own brightness, subtracting nothing thresholds most of the tag to
white. The paper's markers are dark objects in dim water; ours are washed out. `constant = 4` sits
between ARUco's 7 and the paper's 0 and is the best value measured.

**The paper's workflow change does transfer, and is worth a lot.** With the constant put back to
something sensible and the bits read from Otsu, the paper's re-ordering alone lifts usable images
from 17 to 22 (+29%) and total tags from 808 to 926 (+15%), on the same frames. The masked version
beats the base version in every pairing measured, which is the paper's own claim about the mask.

The tags gained are genuinely new: across the same 42 images the two detectors between them found
202 tags that ArUco alone missed. They are concentrated in the frames ArUco does worst on, which is
what turns those frames from unusable into usable — on the frames ArUco already handles well it
still finds more tags than UWARUco does (see the next section).

Cost: about 0.5 s per image against ArUco's 0.02 s, because leaving the threshold constant low
creates thousands of noise contours and the contour filtering is in Python here rather than in
OpenCV's C++. For a calibration that runs once over a few hundred stills this is irrelevant; for
anything real-time it would not be.

## What it does to the calibration

More tags is not the same as a better calibration, and a detector that pulls in harder frames looks
*worse* on headline RMS for having attempted more. So the comparison that counts is made on the
YAMLs `Calibration.py` writes: `compare_calibrations.py` reads the per-image errors out of them and
reports both the all-images RMS and the RMS over **only the images every run used**.

Both runs over all 208 frames of `18_Sep/Rigframes`:

```
src/venv/bin/python src/01_Calibration/compare_calibrations.py --run wide uwaruco:tuned
```

| | images used (L / R) | RMS over the 45 common images (L / R) | fx (L) ± 1σ |
|---|---|---|---|
| `aruco/wide` | 45 / 46 | **2.014 / 1.369 px** | 777.1 ± 4.86 |
| `uwaruco:tuned` | **57 / 56** | 2.694 / 1.727 px | 776.0 ± 6.65 |

The two agree on the lens: fx differs by 1.0 px (left) and 2.1 px (right), inside either run's own
1σ. So this is not a question of which answer is right, but of what each one costs.

**UWARUco buys coverage, and the coverage costs something in the fit.** It brings twelve more left
frames and ten more right frames in — and those extra frames are not junk: their RMS (2.033 px left)
is better than the common set's. But on the frames both detectors use it is about 0.7 px worse.

Where that 0.7 px does *not* come from is worth stating, because the obvious explanations are both
measurable and both turn out to be wrong:

* **Not worse corners.** On the 190 tags both detectors find in the same frames, their corners
  disagree by 0.81 px before refinement, and after `cv2.cornerSubPix` the two agree to within 0.03
  px on average (mean correction 1.59 px for ArUco, 1.62 px for UWARUco). No corner from either
  detector started further out than the 5 px refinement window. Whatever else differs, the corner
  positions that reach the calibration do not.
* **Not extra marginal tags dragged into the easy frames.** On the images *both* detectors could
  use, ArUco actually finds more tags than UWARUco (205 against 186), and UWARUco's few exclusive
  tags there are larger, not smaller, than the shared ones (median side 42 px against 33 px).

UWARUco's entire gain is on frames ArUco fails on **outright** — frames that had fewer than ten tags
and were dropped. So the higher common-subset RMS is a property of the fit rather than of the
detections: `uwaruco:tuned`'s intrinsics are fitted over a set that includes twelve hard, oblique,
far-away views, and intrinsics fitted to that broader set reproject slightly worse on the easy
frames that both runs share. Whether that is a worse calibration or merely a differently-weighted
one is not something reprojection error on the easy frames can settle; the ruler check on the
baseline can. For what it is worth the three runs agree there too: 150.29 mm for `aruco/wide`,
150.47 mm for `uwaruco:tuned`, a 0.18 mm (0.1%) difference on a 150 mm baseline.

### One thing that was tried and did not work

Since the same marker is found by several window sizes, averaging those passes' corners looks like a
free noise reduction. It is not: `mergeCorners: "mean"` made the calibration distinctly worse, on
identical detections.

| | images used (L / R) | RMS over the 45 common images (L / R) |
|---|---|---|
| `uwaruco:tuned` (keep the largest quad, ARUco's rule) | 57 / 56 | 2.694 / 1.727 px |
| `uwaruco:tuned_mean` (average the passes) | 57 / 56 | 3.367 / 2.503 px |

The passes are not independent estimates of the same corner. Each window size places the border at
its own systematic offset, so averaging them mixes several biases rather than cancelling one noise
term, and hands `cornerSubPix` a starting point that belongs to none of them. The option stays in
`Params` so the result is reproducible, but `"largest"` is the default and should remain it.

So which to use depends on what is short. With 104 frame pairs and 45 already usable, `aruco/wide`
is still the right default for this recording, and it stays the default in `Calibration.py`. On a
recording where too few frames clear the ten-tag bar to calibrate at all, `uwaruco:tuned` is the one
that gets a calibration out — and that is a real capability this project did not have before.

## Files

| file | what it is |
|---|---|
| [uwaruco.py](uwaruco.py) | the detector: threshold, mask, contours, identify, merge, feedback |
| [uwaruco_presets.json](uwaruco_presets.json) | named tunings, in data rather than code |

`markerBorderBits` and `errorCorrectionRate` are **not** set here. They describe the printed target
rather than the tuning, and are read from `01_Calibration/detector_presets.json` so that the two
detectors cannot come to disagree about what is on the board.
