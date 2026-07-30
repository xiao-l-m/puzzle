#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from calibrate_from_a4_lines import CalibrationRejected
from validate_a4_calibration_candidate import (
    guard_output_path,
    load_candidate,
    remap_roundtrip_errors,
    transform_points,
)


class CandidateA4ConversionTests(unittest.TestCase):
    def test_corner_transform_matches_known_normalized_points(self) -> None:
        size = (1280, 720)
        matrix = np.asarray(
            [[704.0, 0.0, 640.0], [0.0, 704.0, 360.0], [0.0, 0.0, 1.0]],
            np.float64,
        )
        distortion = np.asarray([-0.285, 0.077, 0.0, 0.0, 0.0], np.float64)
        new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            matrix, distortion, size, 1.0, size
        )
        normalized = np.asarray(
            [[-0.65, 0.42], [-0.63, -0.41], [0.38, -0.40], [0.40, 0.43]],
            np.float64,
        )
        radius2 = np.sum(np.square(normalized), axis=1)
        radial = 1.0 + distortion[0] * radius2 + distortion[1] * np.square(radius2)
        observed = np.column_stack(
            [
                matrix[0, 2] + matrix[0, 0] * normalized[:, 0] * radial,
                matrix[1, 2] + matrix[1, 1] * normalized[:, 1] * radial,
            ]
        )
        expected = np.column_stack(
            [
                new_matrix[0, 2] + new_matrix[0, 0] * normalized[:, 0],
                new_matrix[1, 2] + new_matrix[1, 1] * normalized[:, 1],
            ]
        )
        candidate = {
            "camera_matrix": matrix,
            "dist_coeffs": distortion,
            "image_size": size,
        }
        converted = transform_points(observed, candidate, new_matrix)
        np.testing.assert_allclose(converted, expected, atol=0.002)
        self.assertLess(
            max(remap_roundtrip_errors(observed, converted, candidate, new_matrix)),
            0.15,
        )

    def test_guard_refuses_formal_or_input_a4_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaises(CalibrationRejected):
                guard_output_path(Path(directory) / "a4_calibration.json", source)
            with self.assertRaises(CalibrationRejected):
                guard_output_path(source, source)

    def test_schema_rejects_false_quality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez_compressed(
                str(path),
                camera_matrix=np.eye(3),
                dist_coeffs=np.zeros(5),
                image_size=np.asarray([1280, 720]),
                calibration_method=np.asarray("single_frame_a4_plumb_line_candidate"),
                quality_gate_passed=np.asarray(False),
                validation_line_rms_raw=np.asarray(5.0),
                validation_line_rms_corrected=np.asarray(0.8),
                validation_rms_ratio=np.asarray(0.16),
                per_line_rms_raw=np.ones(5) * 5.0,
                per_line_rms_corrected=np.ones(5) * 0.8,
                line_names=np.asarray(
                    ["outer_edge_1", "outer_edge_2", "outer_edge_3", "outer_edge_4", "divider"]
                ),
                line_sample_counts=np.ones(5, np.int32) * 100,
                line_median_contrast=np.ones(5) * 150.0,
                focal_scale=np.asarray(0.55),
                principal_point_fixed=np.asarray(True),
            )
            with self.assertRaises(CalibrationRejected):
                load_candidate(path)


if __name__ == "__main__":
    unittest.main()
