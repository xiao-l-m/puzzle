#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from puzzle_vision import CameraUndistorter, PuzzleVisionApp


class CameraCalibrationGuardTests(unittest.TestCase):
    @staticmethod
    def _write_calibration(
        path: Path,
        *,
        quality: bool = True,
        corrected_rms: float = 0.6,
        validation_ratio: float = 0.15,
    ) -> None:
        np.savez_compressed(
            str(path),
            camera_matrix=np.asarray(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
                np.float64,
            ),
            dist_coeffs=np.asarray([-0.25, 0.06, 0.0, 0.0, 0.0], np.float64),
            image_size=np.asarray([640, 480], np.int32),
            calibration_method=np.asarray(
                "single_frame_a4_plumb_line_candidate"
            ),
            quality_gate_passed=np.asarray(quality, np.bool_),
            validation_rms_ratio=np.asarray(validation_ratio, np.float64),
            mean_reprojection_error=np.asarray(corrected_rms, np.float64),
        )

    def test_valid_candidate_exposes_hash_and_quality_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_calibration.npz"
            self._write_calibration(path)
            loader = CameraUndistorter(str(path), 640, 480)
            self.assertTrue(loader.enabled, loader.error)
            status = loader.status()
            self.assertEqual(len(status["file_sha256"]), 64)
            self.assertEqual(
                status["calibration_method"],
                "single_frame_a4_plumb_line_candidate",
            )
            self.assertEqual(status["mean_reprojection_error_px"], 0.6)

    def test_candidate_with_failed_quality_gate_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera_calibration.npz"
            self._write_calibration(path, quality=False)
            loader = CameraUndistorter(str(path), 640, 480)
            self.assertFalse(loader.enabled)
            self.assertIn("quality gate", loader.error)

    def test_a4_corners_must_match_exact_calibration_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "camera_calibration.npz"
            self._write_calibration(calibration_path)
            loader = CameraUndistorter(str(calibration_path), 640, 480)
            self.assertTrue(loader.enabled, loader.error)

            a4_path = root / "a4.json"
            document = {
                "physical_a4_corners_px": [
                    [80.0, 60.0],
                    [560.0, 60.0],
                    [560.0, 420.0],
                    [80.0, 420.0],
                ],
                "order": "A4_TOP_LEFT,TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT",
                "capture_size_px": [640, 480],
                "camera_undistortion_enabled": True,
                "camera_calibration_sha256": loader.file_sha256,
            }
            a4_path.write_text(json.dumps(document), encoding="utf-8")
            app = PuzzleVisionApp.__new__(PuzzleVisionApp)
            app.a4_calibration_file = a4_path
            app.undistorter = loader
            corners = PuzzleVisionApp._load_a4_calibration(app)
            self.assertIsNotNone(corners)

            document["camera_calibration_sha256"] = "0" * 64
            a4_path.write_text(json.dumps(document), encoding="utf-8")
            self.assertIsNone(PuzzleVisionApp._load_a4_calibration(app))


if __name__ == "__main__":
    unittest.main()
