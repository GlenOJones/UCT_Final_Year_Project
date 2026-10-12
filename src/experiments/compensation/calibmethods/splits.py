"""Train/validation splits that hold out whole recordings.

Each recording (Water_C1_frames, ...) is one video swept from about 0.25 m to 1.4 m at a single board
tilt, and its frames are near neighbours of each other. Holding out single frames would leave their
neighbours in training and score a model on almost the data it was fitted to. Holding out recordings
means the held-out frames come from tilts the model never saw, which is the question a calibration
has to answer. Every combination of k held-out recordings is used (k = 2 of 8 gives 28 splits, each
recording held out 7 times), so the result does not depend on one lucky or unlucky choice.
"""
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Split:
    index: int
    train: tuple
    test: tuple


def leave_groups_out(groups, k):
    groups = sorted(groups)
    if not 0 < k < len(groups):
        raise ValueError(f"cannot hold out {k} of {len(groups)} recordings")
    return [Split(i, tuple(g for g in groups if g not in held), held)
            for i, held in enumerate(combinations(groups, k))]
