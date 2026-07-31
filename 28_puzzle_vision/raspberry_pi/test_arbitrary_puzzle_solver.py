#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

import numpy as np

from arbitrary_puzzle_solver import (
    point_strictly_inside_polygon,
    polygon_centroid,
    polygons_overlap_with_area,
    reconstruct_rectangle,
    solve_arbitrary_puzzle,
)
from puzzle_motion import StageCalibration, build_task_plan


def observed_piece(
    identifier: str,
    target_polygon: list[list[float]],
    rotation_deg: float,
    observed_center: tuple[float, float],
) -> dict:
    polygon = np.asarray(target_polygon, np.float64)
    local_center = np.mean(polygon, axis=0)
    radians = math.radians(rotation_deg)
    matrix = np.asarray(
        [[math.cos(radians), -math.sin(radians)],
         [math.sin(radians), math.cos(radians)]],
        np.float64,
    )
    observed = (
        (polygon - local_center) @ matrix.T
        + np.asarray(observed_center, np.float64)
    )
    return {
        "id": identifier,
        "vertices_mm": observed.tolist(),
        "center_mm": list(observed_center),
        "pick_point_mm": list(observed_center),
        "pick_method": "AREA_CENTROID",
    }


class ArbitraryPuzzleSolverTests(unittest.TestCase):
    def assert_valid_solution(self, result: dict, count: int) -> None:
        self.assertTrue(result.get("ready"), result)
        self.assertEqual(result["piece_count"], count)
        self.assertEqual(len(result["moves"]), count)
        width, height = result["target_rectangle_mm"]["size"]
        long_side, short_side = max(width, height), min(width, height)
        self.assertGreaterEqual(long_side, 90.0)
        self.assertLessEqual(long_side, 120.0)
        self.assertGreaterEqual(short_side, 50.0)
        self.assertLessEqual(short_side, 90.0)
        self.assertLessEqual(result["maximum_adjacent_vertex_gap_mm"], 15.0)
        polygons = []
        for move in result["moves"]:
            pick = np.asarray(move["place_a4_mm"], np.float64)
            polygon = np.asarray(move["target_vertices_mm"], np.float64)
            self.assertTrue(point_strictly_inside_polygon(pick, polygon), move)
            self.assertLessEqual(float(np.max(polygon[:, 1])), 140.5 + 1.0e-6)
            polygons.append(polygon)
        for first in range(len(polygons)):
            for second in range(first + 1, len(polygons)):
                self.assertFalse(
                    polygons_overlap_with_area(polygons[first], polygons[second])
                )

    def test_one_piece_rectangle(self) -> None:
        pieces = [
            observed_piece(
                "W1",
                [[0, 0], [100, 0], [100, 60], [0, 60]],
                17.0,
                (90.0, 200.0),
            )
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 1)

    def test_two_rotated_triangles(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [100, 0], [0, 60]], 23.0, (65.0, 190.0)
            ),
            observed_piece(
                "W2", [[100, 0], [100, 60], [0, 60]], -31.0, (150.0, 205.0)
            ),
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 2)

    def test_three_piece_reconstruction(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [40, 0], [40, 60], [0, 60]], 10.0, (45.0, 190.0)
            ),
            observed_piece(
                "W2", [[40, 0], [100, 0], [40, 60]], -12.0, (105.0, 200.0)
            ),
            observed_piece(
                "W3", [[100, 0], [100, 60], [40, 60]], 25.0, (165.0, 210.0)
            ),
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 3)

    def test_four_piece_reconstruction(self) -> None:
        target = [
            [[0, 0], [45, 0], [45, 25], [0, 25]],
            [[45, 0], [100, 0], [100, 25], [45, 25]],
            [[0, 25], [45, 25], [45, 60], [0, 60]],
            [[45, 25], [100, 25], [100, 60], [45, 60]],
        ]
        pieces = [
            observed_piece("W1", target[0], 17.0, (35.0, 190.0)),
            observed_piece("W2", target[1], -13.0, (90.0, 195.0)),
            observed_piece("W3", target[2], 29.0, (55.0, 220.0)),
            observed_piece("W4", target[3], -31.0, (150.0, 225.0)),
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 4)

    def test_four_piece_three_segment_composite_seam(self) -> None:
        """One triangle edge is shared by three consecutive piece edges."""
        target = [
            [[0, 0], [0, 30], [76, 18], [100, 0]],
            [[20, 60], [100, 0], [100, 60]],
            [[0, 40], [0, 30], [76, 18], [36, 48]],
            [[0, 60], [0, 40], [36, 48], [20, 60]],
        ]
        pieces = [
            observed_piece("W1", target[0], 31.0, (170.0, 220.0)),
            observed_piece("W2", target[1], -18.0, (125.0, 235.0)),
            observed_piece("W3", target[2], 47.0, (75.0, 220.0)),
            observed_piece("W4", target[3], -36.0, (25.0, 220.0)),
        ]
        for piece in pieces:
            piece["polygon_area_error_ratio"] = 0.0
        result = solve_arbitrary_puzzle(pieces)
        self.assert_valid_solution(result, 4)
        width, height = result["target_rectangle_mm"]["size"]
        # The motion target deliberately opens every internal seam by up to
        # 1 mm; the accumulated outer box may therefore grow several mm.
        self.assertAlmostEqual(max(width, height), 100.0, delta=7.0)
        self.assertAlmostEqual(min(width, height), 60.0, delta=8.0)

    def test_live_four_piece_t_junction_reconstruction(self) -> None:
        """The July field set has two long seams split by centre pieces."""
        pieces = [
            {
                "id": "W1",
                "vertices_mm": [
                    [48.0, 32.25], [83.5, 47.5],
                    [81.25, 57.5], [4.25, 52.25],
                ],
                "center_mm": [49.52, 47.42],
                "pick_point_mm": [49.52, 47.42],
                "polygon_area_error_ratio": 0.0258,
            },
            {
                "id": "W2",
                "vertices_mm": [
                    [128.0, 7.75], [146.25, 67.0],
                    [69.75, 90.0], [94.0, 52.75],
                ],
                "center_mm": [114.18, 54.8],
                "pick_point_mm": [114.18, 54.8],
                "polygon_area_error_ratio": 0.0181,
            },
            {
                "id": "W3",
                "vertices_mm": [
                    [161.5, 39.0], [188.75, 53.25],
                    [147.0, 137.5], [141.75, 113.0],
                ],
                "center_mm": [162.28, 83.35],
                "pick_point_mm": [162.28, 83.35],
                "polygon_area_error_ratio": 0.1,
            },
            {
                "id": "W4",
                "vertices_mm": [
                    [80.0, 112.0], [115.5, 116.0],
                    [102.0, 129.75], [80.75, 130.75],
                ],
                "center_mm": [94.32, 121.31],
                "pick_point_mm": [94.32, 121.31],
                "polygon_area_error_ratio": 0.0849,
            },
        ]
        result = reconstruct_rectangle(pieces)
        self.assertTrue(result.get("ready"), result)
        self.assertEqual(result["piece_count"], 4)
        self.assertGreaterEqual(result["coverage_ratio"], 0.86)
        target_polygons = []
        for move in result["moves"]:
            target_polygon = np.asarray(
                move["target_vertices_local_mm"], np.float64
            )
            target_pick = np.asarray(
                move["target_pick_local_mm"], np.float64
            )
            self.assertTrue(
                point_strictly_inside_polygon(target_pick, target_polygon),
                move,
            )
            target_polygons.append(target_polygon)
        for first in range(len(target_polygons)):
            for second in range(first + 1, len(target_polygons)):
                self.assertFalse(
                    polygons_overlap_with_area(
                        target_polygons[first], target_polygons[second]
                    )
                )
        placed = solve_arbitrary_puzzle(pieces)
        self.assertTrue(placed.get("geometry_ready"), placed)
        width, height = placed["target_rectangle_mm"]["size"]
        self.assertAlmostEqual(placed["target_rectangle_mm"]["center"][0], 105.0)
        self.assertEqual(
            placed["target_horizontal_alignment"], "A4_DIVIDER_CENTER"
        )
        self.assertGreaterEqual(max(width, height), 90.0)
        self.assertLessEqual(max(width, height), 120.0)
        self.assertGreaterEqual(min(width, height), 50.0)
        self.assertLessEqual(min(width, height), 90.0)
        target_centers = {
            move["piece_id"]: polygon_centroid(
                np.asarray(move["target_vertices_mm"], np.float64)
            )
            for move in placed["moves"]
        }
        self.assertGreater(target_centers["W4"][1], target_centers["W2"][1])

    def test_100_by_90_field_set_with_calibration_scale_error(self) -> None:
        """A nominal 100x90 set must survive the observed 2..5% scale error."""
        pieces = [
            {"id": "W1", "center_mm": [125.05, 176.38], "pick_point_mm": [125.05, 176.38], "area_mm2": 2763.31, "polygon_area_error_ratio": 0.0007, "vertices_mm": [[63.03, 158.32], [165.42, 160.1], [165.41, 179.01], [147.7, 197.94], [134.79, 207.8]]},
            {"id": "W2", "center_mm": [63.05, 216.85], "pick_point_mm": [63.05, 216.85], "area_mm2": 3985.84, "polygon_area_error_ratio": 0.0013, "vertices_mm": [[27.32, 171.76], [108.18, 203.54], [75.35, 254.55], [46.29, 262.22]]},
            {"id": "W3", "center_mm": [162.66, 238.82], "pick_point_mm": [162.66, 238.82], "area_mm2": 2265.78, "polygon_area_error_ratio": 0.0018, "vertices_mm": [[205.17, 205.74], [194.86, 233.96], [158.86, 265.75], [110.9, 246.91]]},
            {"id": "W4", "center_mm": [120.25, 266.6], "pick_point_mm": [120.25, 266.6], "area_mm2": 489.06, "polygon_area_error_ratio": 0.0006, "vertices_mm": [[106.09, 251.06], [148.59, 274.58], [106.09, 274.12]]},
        ]
        result = solve_arbitrary_puzzle(pieces, divider_y_mm=151.8)
        self.assertTrue(result.get("ready"), result)
        self.assertEqual(result.get("solver"), "outer_boundary_rectangle_pack_v1")
        width, height = result["target_rectangle_mm"]["size"]
        self.assertLessEqual(abs(max(width, height) - 100.0), 7.0, result)
        self.assertLessEqual(abs(min(width, height) - 90.0), 6.0, result)
        self.assertLessEqual(result["maximum_adjacent_vertex_gap_mm"], 20.0)
        polygons = [
            np.asarray(move["target_vertices_mm"], np.float64)
            for move in result["moves"]
        ]
        for first in range(len(polygons)):
            for second in range(first + 1, len(polygons)):
                self.assertFalse(
                    polygons_overlap_with_area(polygons[first], polygons[second])
                )

    def test_visual_lower_pick_uses_direct_stage_y_and_is_reachable(self) -> None:
        piece = observed_piece(
            "W1", [[0, 0], [100, 0], [100, 60], [0, 60]], 0.0, (90.0, 200.0)
        )
        result = solve_arbitrary_puzzle([piece])
        self.assertTrue(result["ready"], result)
        status = {
            "config": {"a4_locked": True},
            "arbitrary_stability": {"stable": True, "samples": 5},
            "arbitrary_plan": result,
        }
        plan = build_task_plan(3, status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(
            plan["moves"][0]["pick_controller_a4_mm"], [90.0, 200.0]
        )

    def test_visual_upper_pick_is_rejected_as_wrong_source_half(self) -> None:
        piece = observed_piece(
            "W1", [[0, 0], [100, 0], [100, 60], [0, 60]], 0.0, (90.0, 5.0)
        )
        result = solve_arbitrary_puzzle([piece])
        self.assertFalse(result["ready"])
        self.assertEqual(
            result["error"], "SOURCE_PICK_POINT_UNREACHABLE_OR_NOT_LOWER_HALF"
        )

    def test_invalid_piece_counts_and_target_size_are_rejected(self) -> None:
        self.assertEqual(
            solve_arbitrary_puzzle([])["error"], "NEED_ONE_TO_FOUR_PIECES"
        )
        tiny = observed_piece(
            "W1", [[0, 0], [100, 0], [100, 10], [0, 10]], 0.0, (90.0, 95.0)
        )
        self.assertEqual(
            solve_arbitrary_puzzle([tiny])["error"],
            "PIECE_EDGE_CLEARLY_SHORTER_THAN_20MM",
        )
        oversized = observed_piece(
            "W1", [[0, 0], [130, 0], [130, 60], [0, 60]], 0.0, (100.0, 95.0)
        )
        self.assertEqual(
            solve_arbitrary_puzzle([oversized])["error"],
            "NO_RECTANGULAR_EDGE_MATCH_SOLUTION",
        )

    def test_best_rejected_candidate_is_returned_when_quality_filters_all(self) -> None:
        sparse = {
            "id": "W1",
            "vertices_mm": [[0, 0], [100, 0], [100, 20], [40, 20], [0, 60]],
            "center_mm": [48, 20],
            "pick_point_mm": [50, 10],
            "polygon_area_error_ratio": 0.0,
        }
        result = reconstruct_rectangle([sparse])
        self.assertTrue(result.get("ready"), result)
        self.assertTrue(result.get("best_effort_forced"), result)
        self.assertEqual(
            result.get("search_path"),
            "FORCED_BEST_REJECTED_CANDIDATE",
        )
        self.assertIn("COVERAGE", result.get("forced_rejection_reasons", []))

    def test_mode3_motion_plan_preserves_variable_count(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [100, 0], [0, 60]], 12.0, (60.0, 195.0)
            ),
            observed_piece(
                "W2", [[100, 0], [100, 60], [0, 60]], -8.0, (150.0, 210.0)
            ),
        ]
        solved = solve_arbitrary_puzzle(pieces)
        status = {
            "config": {"a4_locked": True},
            "arbitrary_stability": {"stable": True, "samples": 5},
            "arbitrary_plan": solved,
        }
        plan = build_task_plan(3, status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(plan["piece_count"], 2)
        self.assertEqual(len(plan["moves"]), 2)
        for move in plan["moves"]:
            self.assertAlmostEqual(
                move["pick_controller_a4_mm"][1], move["pick_a4_mm"][1]
            )
            self.assertAlmostEqual(
                move["place_controller_a4_mm"][1], move["place_a4_mm"][1]
            )
            self.assertGreaterEqual(move["pick_controller_a4_mm"][1], 62.0)
            self.assertLessEqual(move["pick_controller_a4_mm"][1], 287.0)
            self.assertGreaterEqual(move["place_controller_a4_mm"][1], 62.0)
            self.assertLessEqual(move["place_controller_a4_mm"][1], 287.0)

        corrected = build_task_plan(
            3,
            status,
            StageCalibration(
                mode3_place_offset_x_mm=1.5,
                mode3_place_offset_y_mm=-2.0,
            ),
        )
        self.assertTrue(corrected["ready"], corrected)
        for move in corrected["moves"]:
            self.assertAlmostEqual(
                move["place_controller_a4_mm"][0], move["place_a4_mm"][0] + 1.5
            )
            self.assertAlmostEqual(
                move["place_controller_a4_mm"][1], move["place_a4_mm"][1] - 2.0
            )


if __name__ == "__main__":
    unittest.main()
