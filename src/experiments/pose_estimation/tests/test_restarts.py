"""The restart rule (restarts.py), checked on synthetic trajectories whose failures are known exactly, and on a fake
estimator that only works from a given frame on.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import restarts  # noqa: E402

CRITERIA = {"min_posed_fraction": 0.8, "max_jump_mm": 40.0, "max_jump_fraction": 0.05,
            "start_step": 10, "max_start_fraction": 0.5}


def write(path, text):
    with open(path, "w") as fh:
        fh.write(text)


def read(path):
    with open(path) as fh:
        return fh.read()


def names(n, first=0):
    return [f"{i:04d}.png" for i in range(first, first + n)]


def straight(frame_names, step_mm=5.0):
    """Poses moving step_mm along x per frame, none lost."""
    poses = {}
    for n in frame_names:
        T = np.eye(4)
        T[0, 3] = restarts.frame_number(n) * step_mm
        poses[n] = T
    return poses, {n: {"lost": 0} for n in frame_names}


class Assess(unittest.TestCase):
    def test_clean_run_passes(self):
        frames = names(100)
        verdict = restarts.assess(frames, *straight(frames), CRITERIA)
        self.assertFalse(verdict["failed"])
        self.assertEqual((verdict["posed"], verdict["jumps"]), (100, 0))

    def test_lost_cascade_fails(self):
        # ICP that loses frame 1 and never recovers: 1 of 100 posed.
        frames = names(100)
        poses, diag = straight(frames)
        for n in frames[1:]:
            diag[n]["lost"] = 1
        verdict = restarts.assess(frames, poses, diag, CRITERIA)
        self.assertTrue(verdict["failed"])
        self.assertEqual(verdict["posed"], 1)

    def test_posed_fraction_boundary(self):
        # Exactly 80 % posed passes; 79 % fails.
        frames = names(100)
        poses, diag = straight(frames)
        for n in frames[80:]:
            diag[n]["lost"] = 1
        self.assertFalse(restarts.assess(frames, poses, diag, CRITERIA)["failed"])
        diag[frames[79]]["lost"] = 1
        self.assertTrue(restarts.assess(frames, poses, diag, CRITERIA)["failed"])

    def test_jumps_fail_even_when_every_frame_is_posed(self):
        # COLMAP-style failure: everything "posed", but 10 of 99 steps are 300 mm jumps.
        frames = names(100)
        poses, diag = straight(frames)
        for n in frames[10:20]:
            poses[n] = poses[n].copy()
            poses[n][1, 3] = 300.0 * (restarts.frame_number(n) % 2)
        verdict = restarts.assess(frames, poses, diag, CRITERIA)
        self.assertEqual(verdict["posed"], 100)
        self.assertTrue(verdict["failed"])
        self.assertGreater(verdict["jumps"], 4)

    def test_few_jumps_tolerated(self):
        # One jump in 99 steps is under 5 %.
        frames = names(100)
        poses, diag = straight(frames)
        for n in frames[50:]:
            poses[n] = poses[n].copy()
            poses[n][1, 3] = 100.0
        verdict = restarts.assess(frames, poses, diag, CRITERIA)
        self.assertEqual(verdict["jumps"], 1)
        self.assertFalse(verdict["failed"])


class Starts(unittest.TestCase):
    def test_candidates(self):
        self.assertEqual(restarts.candidate_starts(151, 10, 0.5), [10, 20, 30, 40, 50, 60, 70])
        self.assertEqual(restarts.candidate_starts(20, 10, 0.5), [10])
        self.assertEqual(restarts.candidate_starts(15, 10, 0.5), [])


class Subsets(unittest.TestCase):
    def test_links_and_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
            os.makedirs(os.path.join(src, "frames"))
            for n in ("0000.npz", "0001.npz", "0002.npz"):
                write(os.path.join(src, "frames", n), n)
            write(os.path.join(src, "depth_frames.json"), "{}")
            restarts.link_subset(src, dst, ["0001.npz", "0002.npz"], ["frames"], ["depth_frames.json"])
            self.assertEqual(sorted(os.listdir(os.path.join(dst, "frames"))), ["0001.npz", "0002.npz"])
            self.assertTrue(os.path.islink(os.path.join(dst, "frames", "0001.npz")))
            self.assertEqual(read(os.path.join(dst, "frames", "0002.npz")), "0002.npz")
            self.assertFalse(os.path.islink(os.path.join(dst, "depth_frames.json")))
            # Re-linking replaces the old subset.
            restarts.link_subset(src, dst, ["0002.npz"], ["frames"])
            self.assertEqual(os.listdir(os.path.join(dst, "frames")), ["0002.npz"])

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "src", "left"))
            with self.assertRaises(FileNotFoundError):
                restarts.link_subset(os.path.join(tmp, "src"), os.path.join(tmp, "dst"), ["0000.png"], ["left"])


class RunWithRestarts(unittest.TestCase):
    def fake_estimator(self, frames, works_from):
        """An estimator that loses every frame after the first unless it starts at or after works_from."""
        runs = {}

        def run_from(start):
            subset = [n for n in frames if restarts.frame_number(n) >= start]
            poses, diag = straight(subset)
            if start < works_from:
                for n in subset[1:]:
                    diag[n]["lost"] = 1
            runs[f"run{start}"] = (poses, diag)
            return f"run{start}"
        return run_from, runs.__getitem__

    def test_full_run_passes_no_restart(self):
        frames = names(151)
        run_from, read = self.fake_estimator(frames, works_from=0)
        attempts, kept = restarts.run_with_restarts(frames, run_from, read, CRITERIA)
        self.assertEqual([a["start"] for a in attempts], [0])
        self.assertIsNone(kept)

    def test_first_passing_start_is_kept(self):
        # Works only from frame 25: starts 0, 10, 20 fail, 30 passes and is kept; nothing later is tried.
        frames = names(151)
        run_from, read = self.fake_estimator(frames, works_from=25)
        attempts, kept = restarts.run_with_restarts(frames, run_from, read, CRITERIA)
        self.assertEqual([a["start"] for a in attempts], [0, 10, 20, 30])
        self.assertEqual([a["failed"] for a in attempts], [True, True, True, False])
        self.assertEqual(kept["start"], 30)
        self.assertEqual(kept["frames"], 121)        # judged on the frames it was given

    def test_no_start_passes(self):
        frames = names(151)
        run_from, read = self.fake_estimator(frames, works_from=200)
        attempts, kept = restarts.run_with_restarts(frames, run_from, read, CRITERIA)
        self.assertEqual(attempts[-1]["start"], 70)   # up to half the pass, then gives up
        self.assertIsNone(kept)
        self.assertTrue(all(a["failed"] for a in attempts))


if __name__ == "__main__":
    unittest.main()
