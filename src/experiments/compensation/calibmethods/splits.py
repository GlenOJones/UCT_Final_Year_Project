"""Train/validation splits that hold out whole recordings."""
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
