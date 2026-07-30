#!/usr/bin/env python3

import ast
import math
from pathlib import Path
import random
import unittest

from puzzle_motion import (
    _profile_duration_seconds,
    Mode1TargetLatch,
    MotionPlanStabilityGate,
    StageCalibration,
    build_assembly_motion_plan,
    build_transfer_plan,
    estimate_plan_seconds,
)


PIECES = [
    {
        "id": "P1",
        "center_mm": [179.0, 73.7],
        "vertices_mm": [[205.5, 46.0], [163.0, 40.5], [156.5, 126.0]],
    },
    {
        "id": "P2",
        "center_mm": [19.8, 63.9],
        "vertices_mm": [[42.8, 22.8], [12.8, 24.5], [4.2, 124.2]],
    },
    {
        "id": "P3",
        "center_mm": [67.9, 32.0],
        "vertices_mm": [[89.5, 4.0], [35.8, 66.8], [82.2, 38.0]],
    },
    {
        "id": "P4",
        "center_mm": [193.7, 125.4],
        "vertices_mm": [[205.5, 111.2], [177.8, 121.2], [205.8, 130.8]],
    },
]

# Four lower-region pieces whose overall vertical span cannot use one common
# translation, but whose unchanged bounding boxes fit safely in two rows.
LOWER_PACKING_PIECES = [
    {
        "id": "L1",
        "center_mm": [35.0, 185.0],
        "vertices_mm": [[10.0, 175.0], [60.0, 175.0], [35.0, 205.0]],
    },
    {
        "id": "L2",
        "center_mm": [90.0, 213.33],
        "vertices_mm": [[70.0, 205.0], [110.0, 205.0], [90.0, 230.0]],
    },
    {
        "id": "L3",
        "center_mm": [37.5, 250.0],
        "vertices_mm": [[10.0, 240.0], [65.0, 240.0], [65.0, 260.0], [10.0, 260.0]],
    },
    {
        "id": "L4",
        "center_mm": [160.0, 277.5],
        "vertices_mm": [[130.0, 265.0], [190.0, 265.0], [190.0, 290.0], [130.0, 290.0]],
    },
]


LOWER_COMMON_PIECES = [
    {
        "id": "C1",
        "center_mm": [30.0, 200.0],
        "vertices_mm": [[20.0, 190.0], [40.0, 190.0], [30.0, 210.0]],
    },
    {
        "id": "C2",
        "center_mm": [80.0, 205.0],
        "vertices_mm": [[70.0, 195.0], [90.0, 195.0], [80.0, 215.0]],
    },
    {
        "id": "C3",
        "center_mm": [130.0, 230.0],
        "vertices_mm": [[120.0, 220.0], [140.0, 220.0], [130.0, 240.0]],
    },
    {
        "id": "C4",
        "center_mm": [180.0, 235.0],
        "vertices_mm": [[170.0, 225.0], [190.0, 225.0], [180.0, 245.0]],
    },
]


# Representative fixed-camera capture from the real lower-half setup.  These
# four measured orientations fit above the divider without rotating motor 5.
CURRENT_LOWER_PIECES = [
    {
        "id": "P1",
        "center_mm": [63.0, 225.4],
        "vertices_mm": [[77.5, 171.8], [26.2, 255.5], [86.8, 248.8]],
    },
    {
        "id": "P2",
        "center_mm": [180.2, 223.8],
        "vertices_mm": [
            [196.8, 166.8], [177.5, 188.5], [158.8, 261.5], [188.0, 262.8]
        ],
    },
    {
        "id": "P3",
        "center_mm": [123.1, 220.7],
        "vertices_mm": [
            [151.2, 186.2], [110.2, 213.0], [105.5, 249.0], [115.5, 251.0]
        ],
    },
    {
        "id": "P4",
        "center_mm": [34.2, 176.6],
        "vertices_mm": [[41.8, 162.2], [24.2, 171.5], [16.8, 190.0], [51.0, 179.5]],
    },
]


def rotate_a4_measurement_180(pieces):
    rotated = []
    for piece in pieces:
        rotated.append(
            {
                "id": piece["id"],
                "center_mm": [
                    210.0 - piece["center_mm"][0],
                    297.0 - piece["center_mm"][1],
                ],
                "vertices_mm": [
                    [210.0 - point[0], 297.0 - point[1]]
                    for point in piece["vertices_mm"]
                ],
            }
        )
    return rotated


FLIPPED_MEASURED_PIECES = rotate_a4_measurement_180(PIECES)


def load_production_target_templates():
    """Read the literal templates without importing OpenCV on test hosts."""
    source_path = Path(__file__).with_name("puzzle_vision.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    value = None
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TARGET_PIECE_TEMPLATES"
        ):
            value = node.value
            break
    if value is None or not isinstance(value, (ast.Tuple, ast.List)):
        raise AssertionError("TARGET_PIECE_TEMPLATES literal was not found")

    templates = []
    for entry in value.elts:
        if not isinstance(entry, (ast.Tuple, ast.List)) or len(entry.elts) != 2:
            raise AssertionError("unexpected target template structure")
        name = ast.literal_eval(entry.elts[0])
        array_call = entry.elts[1]
        if not isinstance(array_call, ast.Call) or not array_call.args:
            raise AssertionError("target vertices are not an array literal")
        vertices = ast.literal_eval(array_call.args[0])
        templates.append((name, [[float(x), float(y)] for x, y in vertices]))
    return templates


def signed_polygon_area(vertices):
    area2 = 0.0
    for index, point in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]
        area2 += point[0] * following[1] - following[0] * point[1]
    return 0.5 * area2


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def convex_intersection(subject, clip):
    """Sutherland-Hodgman clipping for the convex figure-2 pieces."""
    output = [list(point) for point in subject]
    if signed_polygon_area(clip) < 0.0:
        clip = list(reversed(clip))
    for edge_index, edge_start in enumerate(clip):
        edge_end = clip[(edge_index + 1) % len(clip)]
        edge = [edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]]
        input_vertices = output
        output = []
        if not input_vertices:
            break

        previous = input_vertices[-1]
        previous_inside = cross(
            edge,
            [previous[0] - edge_start[0], previous[1] - edge_start[1]],
        ) >= -1.0e-9
        for current in input_vertices:
            current_inside = cross(
                edge,
                [current[0] - edge_start[0], current[1] - edge_start[1]],
            ) >= -1.0e-9
            if current_inside != previous_inside:
                direction = [
                    current[0] - previous[0],
                    current[1] - previous[1],
                ]
                denominator = cross(edge, direction)
                if abs(denominator) > 1.0e-12:
                    offset = [
                        edge_start[0] - previous[0],
                        edge_start[1] - previous[1],
                    ]
                    fraction = cross(edge, offset) / denominator
                    output.append(
                        [
                            previous[0] + fraction * direction[0],
                            previous[1] + fraction * direction[1],
                        ]
                    )
            if current_inside:
                output.append(list(current))
            previous = current
            previous_inside = current_inside
    return output


class PuzzleMotionTests(unittest.TestCase):
    def test_acceleration_profile_handles_short_and_long_moves(self):
        # 10 mm cannot reach 600 RPM and therefore uses a triangular curve.
        short_seconds = _profile_duration_seconds(10.0 / 4.0, 600, 220, 1.3)
        self.assertAlmostEqual(short_seconds, 2.339, places=3)

        # 200 mm reaches 600 RPM and therefore uses a trapezoidal curve.
        long_seconds = _profile_duration_seconds(200.0 / 4.0, 600, 220, 1.3)
        self.assertAlmostEqual(long_seconds, 7.38, places=2)
        old_seconds = _profile_duration_seconds(200.0 / 4.0, 300, 10, 0.8)
        self.assertGreater(old_seconds / long_seconds, 1.9)

    def setUp(self):
        self.status = {
            "config": {"a4_locked": True},
            "a4_found": True,
            "divider_found": True,
            "divider_y_mm": 150.2,
            "source_region": "upper",
            "stable_four_pieces": True,
            "pieces": PIECES,
        }

    def test_stage_signs(self):
        calibration = StageCalibration()
        self.assertEqual(calibration.a4_to_stage_mm([5, 61]), [0.0, 0.0])
        self.assertEqual(
            calibration.a4_delta_to_motor_cm([5, 61], [205, 286]),
            [-20.0, 22.5],
        )

    def test_submillimetre_vision_boundary_noise_is_clamped(self):
        calibration = StageCalibration()
        point = calibration.clamp_visual_boundary_noise([159.6, 286.5])
        self.assertTrue(calibration.point_is_reachable(point))
        self.assertEqual(
            calibration.a4_to_stage_mm(point),
            [154.6, 225.0],
        )
        far_point = calibration.clamp_visual_boundary_noise([159.6, 287.1])
        self.assertFalse(calibration.point_is_reachable(far_point))

    def test_motion_is_blocked_until_physical_a4_orientation_is_locked(self):
        unlocked = dict(self.status)
        unlocked["config"] = {"a4_locked": False}
        mode1 = build_transfer_plan(unlocked)
        mode2 = build_assembly_motion_plan(unlocked)
        self.assertFalse(mode1["ready"])
        self.assertFalse(mode2["ready"])
        self.assertEqual(mode1["error"], "A4_PHYSICAL_ORIENTATION_NOT_LOCKED")
        self.assertEqual(mode2["error"], "A4_PHYSICAL_ORIENTATION_NOT_LOCKED")

    def test_mode1_preserves_geometry_and_detects_unreachable_pick(self):
        plan = build_transfer_plan(self.status)
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["error"], "STAGE_POINT_UNREACHABLE")
        self.assertEqual(len(plan["moves"]), 4)
        self.assertTrue(
            any(
                item["piece_id"] == "P3" and item["operation"] == "PICK"
                for item in plan["unreachable"]
            )
        )
        shifts = {
            round(move["place_a4_mm"][1] - move["pick_a4_mm"][1], 3)
            for move in plan["moves"]
        }
        self.assertEqual(len(shifts), 1)
        self.assertGreater(next(iter(shifts)), 0.0)

    def test_mode1_ready_with_full_pick_coverage(self):
        calibration = StageCalibration(y_min_a4_mm=0.0)
        plan = build_transfer_plan(self.status, calibration)
        self.assertTrue(plan["ready"], plan)
        self.assertLessEqual(estimate_plan_seconds(plan, calibration), 120.0)

    def test_mode1_targets_are_lower_and_preserve_relative_geometry(self):
        calibration = StageCalibration(y_min_a4_mm=0.0)
        plan = build_transfer_plan(self.status, calibration)
        self.assertTrue(plan["ready"], plan)
        divider = float(self.status["divider_y_mm"])
        margin = float(plan["margin_mm"])
        sources = {piece["id"]: piece for piece in PIECES}

        moves_by_id = {move["piece_id"]: move for move in plan["moves"]}
        self.assertEqual(set(moves_by_id), set(sources))
        for piece_id, move in moves_by_id.items():
            source_vertices = sources[piece_id]["vertices_mm"]
            target_vertices = move["target_vertices_mm"]
            self.assertEqual(len(source_vertices), len(target_vertices))
            for source_vertex, target_vertex in zip(
                source_vertices, target_vertices
            ):
                self.assertAlmostEqual(target_vertex[0], source_vertex[0], places=6)
                self.assertAlmostEqual(
                    target_vertex[1] - source_vertex[1],
                    plan["common_translation_mm"][1],
                    places=6,
                )
                self.assertGreaterEqual(target_vertex[1], divider + margin)
                self.assertLessEqual(target_vertex[1], 297.0 - margin)
                self.assertGreaterEqual(target_vertex[0], 0.0)
                self.assertLessEqual(target_vertex[0], 210.0)
            self.assertEqual(move["rotate_deg_clockwise"], 0.0)

        piece_ids = sorted(sources)
        for first_index, first_id in enumerate(piece_ids):
            for second_id in piece_ids[first_index + 1:]:
                source_delta = [
                    sources[second_id]["center_mm"][axis]
                    - sources[first_id]["center_mm"][axis]
                    for axis in (0, 1)
                ]
                target_delta = [
                    moves_by_id[second_id]["place_a4_mm"][axis]
                    - moves_by_id[first_id]["place_a4_mm"][axis]
                    for axis in (0, 1)
                ]
                self.assertAlmostEqual(target_delta[0], source_delta[0], places=6)
                self.assertAlmostEqual(target_delta[1], source_delta[1], places=6)

    def test_mode1_lower_prefers_common_translation_when_it_fits(self):
        status = {
            "config": {"a4_locked": True},
            "a4_found": True,
            "divider_found": True,
            "divider_y_mm": 170.0,
            "source_region": "lower",
            "stable_four_pieces": True,
            "pieces": LOWER_COMMON_PIECES,
        }
        plan = build_transfer_plan(status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(plan["strategy"], "COMMON_TRANSLATION")
        self.assertLess(plan["common_translation_mm"][1], 0.0)
        self.assertAlmostEqual(
            max(
                point[1]
                for move in plan["moves"]
                for point in move["target_vertices_mm"]
            ),
            165.0,
            places=6,
        )

        by_id = {move["piece_id"]: move for move in plan["moves"]}
        piece_ids = sorted(by_id)
        source_by_id = {piece["id"]: piece for piece in LOWER_COMMON_PIECES}
        for first_index, first_id in enumerate(piece_ids):
            for second_id in piece_ids[first_index + 1:]:
                for axis in (0, 1):
                    source_delta = (
                        source_by_id[second_id]["center_mm"][axis]
                        - source_by_id[first_id]["center_mm"][axis]
                    )
                    target_delta = (
                        by_id[second_id]["place_a4_mm"][axis]
                        - by_id[first_id]["place_a4_mm"][axis]
                    )
                    self.assertAlmostEqual(target_delta, source_delta, places=6)

    def test_mode1_lower_bbox_fallback_is_upper_reachable_and_nonoverlapping(self):
        divider = 170.0
        status = {
            "config": {"a4_locked": True},
            "a4_found": True,
            "divider_found": True,
            "divider_y_mm": divider,
            "source_region": "lower",
            "stable_four_pieces": True,
            "pieces": LOWER_PACKING_PIECES,
        }
        plan = build_transfer_plan(status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(plan["strategy"], "BBOX_ROW_PACKING")
        self.assertIsNone(plan["common_translation_mm"])
        self.assertEqual(len(plan["moves"]), 4)
        target_min, target_max = plan["target_y_range_mm"]
        self.assertEqual(target_min, 5.0)
        self.assertEqual(target_max, divider - 5.0)

        source_by_id = {piece["id"]: piece for piece in LOWER_PACKING_PIECES}
        for move in plan["moves"]:
            self.assertEqual(move["rotate_deg_clockwise"], 0.0)
            self.assertEqual(move["motor5_rotate_deg"], 0.0)
            self.assertTrue(
                StageCalibration().point_is_reachable(move["place_a4_mm"])
            )
            source_vertices = source_by_id[move["piece_id"]]["vertices_mm"]
            target_vertices = move["target_vertices_mm"]
            shifts = {
                (
                    round(target[0] - source[0], 6),
                    round(target[1] - source[1], 6),
                )
                for source, target in zip(source_vertices, target_vertices)
            }
            self.assertEqual(len(shifts), 1)
            for x, y in target_vertices:
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 210.0)
                self.assertGreaterEqual(y, target_min)
                self.assertLessEqual(y, target_max)

        for first_index, first in enumerate(plan["moves"]):
            for second in plan["moves"][first_index + 1:]:
                overlap = convex_intersection(
                    first["target_vertices_mm"], second["target_vertices_mm"]
                )
                overlap_area = (
                    0.0
                    if len(overlap) < 3
                    else abs(signed_polygon_area(overlap))
                )
                self.assertAlmostEqual(overlap_area, 0.0, places=7)

        # Bottom-up packing keeps a 3 mm contour-noise guard above the divider.
        self.assertAlmostEqual(
            max(
                point[1]
                for move in plan["moves"]
                for point in move["target_vertices_mm"]
            ),
            target_max - 3.0,
            places=6,
        )

    def test_mode1_measured_capture_is_repacked_without_rotation(self):
        status = {
            "config": {"a4_locked": True},
            "a4_found": True,
            "divider_found": True,
            "divider_y_mm": 150.2,
            "source_region": "lower",
            "stable_four_pieces": True,
            "pieces": FLIPPED_MEASURED_PIECES,
        }
        plan = build_transfer_plan(status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(plan["packing"]["rotation_total_deg"], 0.0)
        self.assertTrue(
            all(move["rotate_deg_clockwise"] == 0.0 for move in plan["moves"])
        )
        self.assertTrue(
            all(move["motor5_rotate_deg"] == 0.0 for move in plan["moves"])
        )

    def test_polygon_grid_targets_are_deterministic_under_submillimetre_noise(self):
        rng = random.Random(20260730)
        expected_targets = None
        expected_rotations = None
        gate = MotionPlanStabilityGate(
            required_samples=3,
            position_tolerance_mm=0.8,
            angle_tolerance_deg=1.5,
        )
        latch = Mode1TargetLatch(clearance_mm=3.0)
        last_stability = None

        for frame_index in range(24):
            pieces = []
            for original in CURRENT_LOWER_PIECES:
                pieces.append(
                    {
                        "id": original["id"],
                        "center_mm": [
                            original["center_mm"][0] + rng.uniform(-0.25, 0.25),
                            original["center_mm"][1] + rng.uniform(-0.25, 0.25),
                        ],
                        "vertices_mm": [
                            [
                                point[0] + rng.uniform(-0.35, 0.35),
                                point[1] + rng.uniform(-0.35, 0.35),
                            ]
                            for point in original["vertices_mm"]
                        ],
                    }
                )
            status = {
                "config": {"a4_locked": True},
                "a4_found": True,
                "divider_found": True,
                "divider_y_mm": 150.2,
                "source_region": "lower",
                "stable_four_pieces": True,
                "pieces": pieces,
            }
            plan = build_transfer_plan(status)
            plan = latch.update(plan, status, StageCalibration())
            self.assertTrue(plan["ready"], (frame_index, plan))
            self.assertIn(
                plan["strategy"], {"BBOX_ROW_PACKING", "POLYGON_GRID_PACKING"}
            )
            self.assertTrue(plan["packing"]["target_latched"])
            self.assertEqual(plan["packing"]["rotation_total_deg"], 0.0)
            self.assertLessEqual(estimate_plan_seconds(plan), 120.0)

            by_id = {move["piece_id"]: move for move in plan["moves"]}
            targets = {
                piece_id: tuple(move["place_a4_mm"])
                for piece_id, move in by_id.items()
            }
            rotations = {
                piece_id: move["rotate_deg_clockwise"]
                for piece_id, move in by_id.items()
            }
            if expected_targets is None:
                expected_targets = targets
                expected_rotations = rotations
            self.assertEqual(rotations, expected_rotations, frame_index)
            self.assertEqual(targets, expected_targets, frame_index)
            self.assertTrue(all(rotation == 0.0 for rotation in rotations.values()))
            self.assertTrue(
                all(move["motor5_rotate_deg"] == 0.0 for move in plan["moves"])
            )

            target_min, target_max = plan["target_y_range_mm"]
            for move in plan["moves"]:
                self.assertTrue(
                    StageCalibration().point_is_reachable(move["place_a4_mm"])
                )
                for x, y in move["target_vertices_mm"]:
                    self.assertGreaterEqual(x, 0.0)
                    self.assertLessEqual(x, 210.0)
                    self.assertGreaterEqual(y, target_min)
                    self.assertLessEqual(y, target_max)
            for first_index, first in enumerate(plan["moves"]):
                for second in plan["moves"][first_index + 1:]:
                    overlap = convex_intersection(
                        first["target_vertices_mm"],
                        second["target_vertices_mm"],
                    )
                    overlap_area = (
                        0.0
                        if len(overlap) < 3
                        else abs(signed_polygon_area(overlap))
                    )
                    self.assertAlmostEqual(overlap_area, 0.0, places=7)

            last_stability = gate.update(1, plan)
            if frame_index >= 2:
                self.assertTrue(last_stability["stable"], last_stability)

        self.assertIsNotNone(last_stability)
        self.assertLessEqual(last_stability["position_spread_mm"], 0.8)

    def test_mode2_freezes_vision_solution_and_checks_range(self):
        assembly_moves = []
        targets = [[128.3, 213.6], [95.0, 242.0], [68.7, 202.6], [88.1, 220.4]]
        for index, (piece, target) in enumerate(zip(PIECES, targets), start=1):
            assembly_moves.append(
                {
                    "order": index,
                    "piece_id": piece["id"],
                    "target_piece": "G%d" % index,
                    "current_centroid_mm": piece["center_mm"],
                    "target_centroid_mm": target,
                    "rotate_deg_clockwise": 15.0 * index,
                    "target_vertices_mm": [],
                    "shape_residual": 0.05,
                }
            )
        status = dict(self.status)
        status["assembly_plan"] = {
            "ready": True,
            "target_rectangle_mm": {
                "origin": [55.0, 193.6],
                "size": [100.0, 60.0],
            },
            "moves": assembly_moves,
        }
        blocked = build_assembly_motion_plan(status)
        self.assertFalse(blocked["ready"])
        ready = build_assembly_motion_plan(
            status, StageCalibration(y_min_a4_mm=0.0)
        )
        self.assertTrue(ready["ready"], ready)
        self.assertEqual([move["order"] for move in ready["moves"]], [1, 2, 3, 4])

    def test_mode2_keeps_vision_supplied_upper_targets_unchanged(self):
        upper_targets = [
            [128.3, 95.0],
            [95.0, 123.4],
            [68.7, 84.0],
            [88.1, 102.0],
        ]
        assembly_moves = []
        for index, (piece, target) in enumerate(
            zip(LOWER_PACKING_PIECES, upper_targets), start=1
        ):
            assembly_moves.append(
                {
                    "order": index,
                    "piece_id": piece["id"],
                    "target_piece": "G%d" % index,
                    "current_centroid_mm": piece["center_mm"],
                    "target_centroid_mm": target,
                    "rotate_deg_clockwise": -12.5 * index,
                    "target_vertices_mm": [],
                    "shape_residual": 0.05,
                }
            )
        status = {
            "config": {"a4_locked": True},
            "stable_four_pieces": True,
            "source_region": "lower",
            "assembly_plan": {
                "ready": True,
                "target_rectangle_mm": {
                    "origin": [55.0, 75.0],
                    "size": [100.0, 60.0],
                },
                "moves": assembly_moves,
            },
        }
        plan = build_assembly_motion_plan(status)
        self.assertTrue(plan["ready"], plan)
        self.assertEqual(
            [move["place_a4_mm"] for move in plan["moves"]], upper_targets
        )
        self.assertTrue(
            all(61.0 <= point[1] < 150.0 for point in upper_targets)
        )


class Figure2TemplateGeometryTests(unittest.TestCase):
    def test_templates_tile_one_100_by_60_rectangle_without_area_overlap(self):
        templates = load_production_target_templates()
        self.assertEqual(len(templates), 4)
        self.assertEqual(len({name for name, _vertices in templates}), 4)

        areas = []
        all_vertices = []
        for name, vertices in templates:
            self.assertGreaterEqual(len(vertices), 3, name)
            area = abs(signed_polygon_area(vertices))
            self.assertGreater(area, 0.0, name)
            areas.append(area)
            all_vertices.extend(vertices)
            for x, y in vertices:
                self.assertGreaterEqual(x, 0.0, name)
                self.assertLessEqual(x, 100.0, name)
                self.assertGreaterEqual(y, 0.0, name)
                self.assertLessEqual(y, 60.0, name)

        self.assertAlmostEqual(min(point[0] for point in all_vertices), 0.0)
        self.assertAlmostEqual(max(point[0] for point in all_vertices), 100.0)
        self.assertAlmostEqual(min(point[1] for point in all_vertices), 0.0)
        self.assertAlmostEqual(max(point[1] for point in all_vertices), 60.0)
        self.assertAlmostEqual(sum(areas), 100.0 * 60.0, places=6)

        for first_index, (first_name, first) in enumerate(templates):
            for second_name, second in templates[first_index + 1:]:
                overlap = convex_intersection(first, second)
                overlap_area = (
                    0.0
                    if len(overlap) < 3
                    else abs(signed_polygon_area(overlap))
                )
                self.assertTrue(
                    math.isclose(overlap_area, 0.0, abs_tol=1.0e-7),
                    "%s and %s overlap by %.9f mm^2"
                    % (first_name, second_name, overlap_area),
                )


if __name__ == "__main__":
    unittest.main()
