#!/usr/bin/env python3

from __future__ import annotations

import unittest

import cv2
import numpy as np

from puzzle_vision import (
    DetectorConfig,
    PuzzleDetector,
    WARP_HEIGHT,
    WARP_PX_PER_MM,
    WARP_WIDTH,
)


class Project21PieceRecognitionTests(unittest.TestCase):
    @staticmethod
    def _piece(vertices: np.ndarray, center: tuple[float, float]) -> dict:
        return {
            "id": "W1",
            "center_mm": list(center),
            "vertices_mm": np.asarray(vertices, np.float64).tolist(),
            "vertex_count": len(vertices),
            "area_mm2": 400.0,
            "polygon_area_error_ratio": 0.01,
            "angle_deg": 0.0,
        }

    @staticmethod
    def _scene(background: int, white: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        vertical = np.linspace(0.0, 18.0, WARP_HEIGHT, dtype=np.float32)
        gray = np.full((WARP_HEIGHT, WARP_WIDTH), background, np.float32)
        gray += vertical[:, None]
        gray += rng.normal(0.0, 2.0, gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        polygons = [
            np.asarray([[75, 700], [245, 715], [150, 850]], np.int32),
            np.asarray([[315, 690], [505, 720], [475, 875], [300, 840]], np.int32),
            np.asarray([[590, 705], [745, 730], [765, 810], [690, 885], [570, 820]], np.int32),
        ]
        for polygon in polygons:
            cv2.fillPoly(gray, [polygon], int(white))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_project21_otsu_keeps_three_piece_shapes_across_lighting(self) -> None:
        detector = PuzzleDetector(DetectorConfig(stable_frames=3))
        observations = []
        divider_y = int(round(151.8 * WARP_PX_PER_MM))
        for index, (background, white) in enumerate(
            ((28, 175), (55, 215), (82, 245))
        ):
            frame = self._scene(background, white, seed=100 + index)
            mask, _background_luma, _threshold = (
                detector.segment_arbitrary_white_pieces(frame, divider_y)
            )
            pieces, _contours = detector.extract_arbitrary_white_pieces(mask)
            observations.append(pieces)
            self.assertEqual(len(pieces), 3, pieces)
            self.assertEqual(
                [piece["vertex_count"] for piece in pieces],
                [3, 4, 5],
            )
            self.assertTrue(
                all(
                    piece["detection_source"]
                    == "PROJECT21_OTSU_WHITE_MASK"
                    for piece in pieces
                )
            )
            self.assertTrue(
                all(
                    piece["polygon_fit_source"]
                    == "PROJECT21_FIXED_EPSILON_POLYGON"
                    for piece in pieces
                )
            )

        centers = np.asarray(
            [
                [piece["center_mm"] for piece in pieces]
                for pieces in observations
            ],
            np.float64,
        )
        median = np.median(centers, axis=0)
        spread = float(
            np.max(np.linalg.norm(centers - median[None, :, :], axis=2))
        )
        self.assertLessEqual(spread, 0.4)

    def test_five_frame_stationary_latch_removes_alternating_corner(self) -> None:
        detector = PuzzleDetector(DetectorConfig(stable_frames=3))
        base = np.asarray(
            [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]],
            np.float64,
        )
        outputs = []
        stability = []
        for offset in (
            0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0, 8.0, 0.0
        ):
            observed = base.copy()
            observed[3, 0] += offset
            stabilized = detector.stabilize_arbitrary_polygons(
                [self._piece(observed, (50.0, 50.0))]
            )
            outputs.append(np.asarray(stabilized[0]["vertices_mm"]))
            stability.append(detector.arbitrary_piece_stability(stabilized))

        self.assertFalse(any(item["stable"] for item in stability[:8]))
        self.assertTrue(stability[8]["stable"])
        for output in outputs[5:]:
            np.testing.assert_allclose(output, outputs[4], atol=0.01)
        self.assertEqual(
            stabilized[0]["geometry_source"],
            "RECENT_5_FRAME_MEDIAN_POLYGON",
        )

    def test_stationary_latch_clears_after_two_mm_movement(self) -> None:
        detector = PuzzleDetector(DetectorConfig(stable_frames=3))
        base = np.asarray(
            [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]],
            np.float64,
        )
        for _index in range(5):
            detector.stabilize_arbitrary_polygons(
                [self._piece(base, (50.0, 50.0))]
            )

        moved = base + np.asarray([3.0, 0.0])
        moved[3, 0] += 5.0
        output = detector.stabilize_arbitrary_polygons(
            [self._piece(moved, (53.0, 50.0))]
        )[0]
        expected = detector._canonical_detected_polygon(moved)
        np.testing.assert_allclose(
            np.asarray(output["vertices_mm"]), expected, atol=0.01
        )
        self.assertEqual(output["geometry_samples"], 1)


if __name__ == "__main__":
    unittest.main()
