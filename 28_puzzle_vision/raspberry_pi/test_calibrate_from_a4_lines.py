#!/usr/bin/env python3
"""Synthetic regression tests for the one-frame A4 plumb-line fallback."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from calibrate_from_a4_lines import (
    DEFAULT_OUTPUT,
    CalibrationRejected,
    estimate_radial_coefficients,
    evaluate_candidate,
    save_candidate,
    split_groups,
    synthetic_groups,
)


class A4LineCalibrationTests(unittest.TestCase):
    def test_distorted_lines_pass_independent_holdout_gate(self) -> None:
        groups, _expected_matrix = synthetic_groups(k1=-0.31, k2=0.09)
        matrix, distortion = estimate_radial_coefficients(
            split_groups(groups, 0), (1280, 720), 0.55
        )
        report = evaluate_candidate(groups, matrix, distortion, (1280, 720))
        self.assertTrue(report["accepted"], report["reasons"])
        self.assertLess(float(report["validation_ratio"]), 0.20)
        self.assertLess(abs(float(distortion[0]) - (-0.31)), 0.08)

    def test_already_straight_lines_are_rejected(self) -> None:
        groups, _matrix = synthetic_groups(k1=0.0, k2=0.0, noise_px=0.08)
        matrix, distortion = estimate_radial_coefficients(
            split_groups(groups, 0), (1280, 720), 0.55
        )
        report = evaluate_candidate(groups, matrix, distortion, (1280, 720))
        self.assertFalse(report["accepted"])
        self.assertTrue(
            any("already too straight" in str(reason) for reason in report["reasons"]),
            report["reasons"],
        )

    def test_writer_refuses_formal_live_filename(self) -> None:
        groups, matrix = synthetic_groups()
        distortion = np.asarray([-0.31, 0.09, 0.0, 0.0, 0.0], np.float64)
        report = evaluate_candidate(groups, matrix, distortion, (1280, 720))
        names = ["outer_edge_1", "outer_edge_2", "outer_edge_3", "outer_edge_4", "divider"]
        diagnostics = {
            name: {"samples": float(len(group)), "median_contrast": 100.0}
            for name, group in zip(names, groups)
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CalibrationRejected):
                save_candidate(
                    Path(directory) / "camera_calibration.npz",
                    matrix,
                    distortion,
                    (1280, 720),
                    report,
                    names,
                    diagnostics,
                    Path("synthetic.png"),
                    0.55,
                )
            candidate = Path(directory) / DEFAULT_OUTPUT
            save_candidate(
                candidate, matrix, distortion, (1280, 720), report,
                names, diagnostics, Path("synthetic.png"), 0.55,
            )
            with np.load(str(candidate), allow_pickle=False) as saved:
                self.assertTrue(bool(saved["quality_gate_passed"]))
                self.assertEqual(tuple(saved["image_size"]), (1280, 720))
            # The candidate intentionally uses the exact minimal schema loaded
            # by the production remapper, without enabling that remapper here.
            from puzzle_vision import CameraUndistorter

            loader = CameraUndistorter(str(candidate), 1280, 720)
            self.assertTrue(loader.enabled, loader.error)
            self.assertIsNotNone(loader.map1)
            self.assertIsNotNone(loader.map2)


if __name__ == "__main__":
    unittest.main()
