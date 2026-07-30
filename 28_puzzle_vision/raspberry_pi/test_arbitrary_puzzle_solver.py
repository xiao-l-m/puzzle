#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest

import numpy as np

from arbitrary_puzzle_solver import (
    point_strictly_inside_polygon,
    polygons_overlap_with_area,
    solve_arbitrary_puzzle,
)
from puzzle_motion import build_task_plan


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
            self.assertGreaterEqual(float(np.min(polygon[:, 1])), 156.5)
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
                (90.0, 95.0),
            )
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 1)

    def test_two_rotated_triangles(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [100, 0], [0, 60]], 23.0, (65.0, 92.0)
            ),
            observed_piece(
                "W2", [[100, 0], [100, 60], [0, 60]], -31.0, (150.0, 103.0)
            ),
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 2)

    def test_three_piece_reconstruction(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [40, 0], [40, 60], [0, 60]], 10.0, (45.0, 95.0)
            ),
            observed_piece(
                "W2", [[40, 0], [100, 0], [40, 60]], -12.0, (105.0, 90.0)
            ),
            observed_piece(
                "W3", [[100, 0], [100, 60], [40, 60]], 25.0, (165.0, 100.0)
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
            observed_piece("W1", target[0], 17.0, (35.0, 80.0)),
            observed_piece("W2", target[1], -13.0, (90.0, 82.0)),
            observed_piece("W3", target[2], 29.0, (55.0, 120.0)),
            observed_piece("W4", target[3], -31.0, (150.0, 115.0)),
        ]
        self.assert_valid_solution(solve_arbitrary_puzzle(pieces), 4)

    def test_unreachable_upper_pick_is_rejected(self) -> None:
        piece = observed_piece(
            "W1", [[0, 0], [100, 0], [100, 60], [0, 60]], 0.0, (90.0, 45.0)
        )
        result = solve_arbitrary_puzzle([piece])
        self.assertFalse(result["ready"])
        self.assertEqual(
            result["error"], "SOURCE_PICK_POINT_UNREACHABLE_OR_NOT_UPPER_HALF"
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

    def test_mode3_motion_plan_preserves_variable_count(self) -> None:
        pieces = [
            observed_piece(
                "W1", [[0, 0], [100, 0], [0, 60]], 12.0, (60.0, 95.0)
            ),
            observed_piece(
                "W2", [[100, 0], [100, 60], [0, 60]], -8.0, (150.0, 100.0)
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


if __name__ == "__main__":
    unittest.main()
