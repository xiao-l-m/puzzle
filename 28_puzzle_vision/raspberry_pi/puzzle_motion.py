#!/usr/bin/env python3
"""Pure motion planning for E-question task 1 modes 1 and 2.

The vision process uses A4 coordinates: top-left is (0, 0), +X points right,
and +Y points down.  This module deliberately has no OpenCV or serial
dependency so plans can be unit-tested away from the Raspberry Pi.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import permutations
import math
from typing import Any, Iterable


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
TRANSFER_EDGE_MARGIN_MM = 5.0
TRANSFER_PACKING_GAP_MM = 5.0


@dataclass(frozen=True)
class StageCalibration:
    """Fixed relationship between the A4 sheet and the slide start position."""

    # Current measured start: magnet centre is 5 mm from the A4 left edge and
    # 61 mm from the A4 top edge.
    home_x_a4_mm: float = 5.0
    home_y_a4_mm: float = 61.0

    # Measured usable travel: X=200 mm, Y=225 mm.
    x_min_a4_mm: float = 5.0
    x_max_a4_mm: float = 205.0
    y_min_a4_mm: float = 61.0
    y_max_a4_mm: float = 286.0

    z_pick_drop_mm: float = 8.0
    rotation_sign: float = 1.0
    vision_boundary_tolerance_mm: float = 1.0

    def point_is_reachable(self, point: Iterable[float]) -> bool:
        x, y = (float(value) for value in point)
        return (
            self.x_min_a4_mm <= x <= self.x_max_a4_mm
            and self.y_min_a4_mm <= y <= self.y_max_a4_mm
        )

    def clamp_visual_boundary_noise(self, point: Iterable[float]) -> list[float]:
        """Clamp only a submillimetre camera error at a calibrated limit."""
        x, y = (float(value) for value in point)
        tolerance = self.vision_boundary_tolerance_mm
        if self.x_min_a4_mm - tolerance <= x < self.x_min_a4_mm:
            x = self.x_min_a4_mm
        elif self.x_max_a4_mm < x <= self.x_max_a4_mm + tolerance:
            x = self.x_max_a4_mm
        if self.y_min_a4_mm - tolerance <= y < self.y_min_a4_mm:
            y = self.y_min_a4_mm
        elif self.y_max_a4_mm < y <= self.y_max_a4_mm + tolerance:
            y = self.y_max_a4_mm
        return [x, y]

    def a4_to_stage_mm(self, point: Iterable[float]) -> list[float]:
        """Return logical slide displacement from the fixed start position."""
        x, y = (float(value) for value in point)
        return [
            round(x - self.home_x_a4_mm, 3),
            round(y - self.home_y_a4_mm, 3),
        ]

    def a4_delta_to_motor_cm(
        self, current: Iterable[float], target: Iterable[float]
    ) -> list[float]:
        """Convert A4 displacement to motor command signs.

        Motor 1 positive moves left, so A4 +X requires a negative command.
        Motors 2 and 3 positive move down, matching A4 +Y.
        """
        current_x, current_y = (float(value) for value in current)
        target_x, target_y = (float(value) for value in target)
        return [
            round(-(target_x - current_x) / 10.0, 4),
            round((target_y - current_y) / 10.0, 4),
        ]


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _angle_delta_deg(value: float, reference: float) -> float:
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


class MotionPlanStabilityGate:
    """Require independently generated plans to agree across camera frames.

    Piece-centre stability alone is insufficient for mode 1 because polygon
    approximation can still change the selected packing rotation or target.
    This gate checks categorical assignment plus every pick/place point and
    rotation angle before a plan may be sent to the MCU.
    """

    def __init__(
        self,
        required_samples: int = 3,
        position_tolerance_mm: float = 0.8,
        angle_tolerance_deg: float = 1.5,
    ) -> None:
        if required_samples < 2:
            raise ValueError("required_samples must be at least 2")
        self.required_samples = int(required_samples)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self._history: dict[int, deque[dict[str, Any]]] = {
            1: deque(maxlen=self.required_samples),
            2: deque(maxlen=self.required_samples),
        }

    def clear(self, mode: int | None = None) -> None:
        if mode is None:
            for history in self._history.values():
                history.clear()
            return
        self._history.setdefault(int(mode), deque(maxlen=self.required_samples)).clear()

    @staticmethod
    def _snapshot(mode: int, plan: dict[str, Any]) -> dict[str, Any] | None:
        moves = list(plan.get("moves") or [])
        if not plan.get("ready", False) or len(moves) != 4:
            return None
        moves.sort(key=lambda move: int(move.get("order", 0)))
        categories: list[tuple[Any, ...]] = []
        points: list[tuple[float, float, float, float]] = []
        angles: list[float] = []
        try:
            for move in moves:
                pick = move["pick_a4_mm"]
                place = move["place_a4_mm"]
                categories.append(
                    (
                        int(move["order"]),
                        str(move["piece_id"]),
                        str(move.get("target_piece", "")),
                    )
                )
                points.append(
                    (
                        float(pick[0]),
                        float(pick[1]),
                        float(place[0]),
                        float(place[1]),
                    )
                )
                angles.append(float(move.get("motor5_rotate_deg", 0.0)))
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        return {
            "category": (
                int(mode),
                str(plan.get("strategy", "")),
                tuple(categories),
            ),
            "points": tuple(points),
            "angles": tuple(angles),
        }

    def update(self, mode: int, plan: dict[str, Any]) -> dict[str, Any]:
        mode = int(mode)
        history = self._history.setdefault(
            mode, deque(maxlen=self.required_samples)
        )
        snapshot = self._snapshot(mode, plan)
        if snapshot is None:
            history.clear()
            return {
                "stable": False,
                "samples": 0,
                "required_samples": self.required_samples,
                "position_spread_mm": None,
                "angle_spread_deg": None,
                "reason": "PLAN_NOT_READY_OR_INVALID",
            }
        if history and history[-1]["category"] != snapshot["category"]:
            history.clear()
        history.append(snapshot)

        position_spread = 0.0
        angle_spread = 0.0
        if len(history) >= 2:
            for move_index in range(4):
                coordinates = [sample["points"][move_index] for sample in history]
                medians = [
                    _median(point[coordinate] for point in coordinates)
                    for coordinate in range(4)
                ]
                for point in coordinates:
                    position_spread = max(
                        position_spread,
                        math.hypot(point[0] - medians[0], point[1] - medians[1]),
                        math.hypot(point[2] - medians[2], point[3] - medians[3]),
                    )
                reference = float(history[0]["angles"][move_index])
                deltas = [
                    _angle_delta_deg(sample["angles"][move_index], reference)
                    for sample in history
                ]
                centre_delta = _median(deltas)
                angle_spread = max(
                    angle_spread,
                    max(abs(delta - centre_delta) for delta in deltas),
                )
        stable = bool(
            len(history) >= self.required_samples
            and position_spread <= self.position_tolerance_mm
            and angle_spread <= self.angle_tolerance_deg
        )
        return {
            "stable": stable,
            "samples": len(history),
            "required_samples": self.required_samples,
            "position_spread_mm": round(position_spread, 3),
            "angle_spread_deg": round(angle_spread, 3),
            "reason": None if stable else "WAITING_FOR_CONSISTENT_PLAN",
        }


def _not_ready(mode: int, error: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": mode,
        "ready": False,
        "error": error,
        "moves": [],
    }
    result.update(extra)
    return result


def _piece_vertices(piece: dict[str, Any]) -> list[list[float]]:
    vertices = piece.get("vertices_mm") or []
    if vertices:
        return [[float(point[0]), float(point[1])] for point in vertices]
    center = piece.get("center_mm")
    if center is None:
        return []
    return [[float(center[0]), float(center[1])]]


def _check_point(
    calibration: StageCalibration,
    point: Iterable[float],
    piece_id: str,
    operation: str,
) -> dict[str, Any] | None:
    position = [round(float(value), 2) for value in point]
    if calibration.point_is_reachable(position):
        return None
    return {
        "piece_id": piece_id,
        "operation": operation,
        "point_a4_mm": position,
        "allowed_a4_mm": {
            "x": [calibration.x_min_a4_mm, calibration.x_max_a4_mm],
            "y": [calibration.y_min_a4_mm, calibration.y_max_a4_mm],
        },
    }


def _move_time_metric(a: Iterable[float], b: Iterable[float]) -> float:
    """Approximate XY time when both axes move at the same linear speed."""
    ax, ay = (float(value) for value in a)
    bx, by = (float(value) for value in b)
    return max(abs(bx - ax), abs(by - ay))


def _nearest_pick_order(
    moves: list[dict[str, Any]], start: Iterable[float]
) -> list[dict[str, Any]]:
    # Four pieces means all 4! orders are cheap to evaluate.  Include the
    # return to HOME because the MCU restores the fixed start pose when done.
    origin = [float(value) for value in start]

    def route_metric(route: tuple[dict[str, Any], ...]) -> float:
        current = origin
        total = 0.0
        for move in route:
            total += _move_time_metric(current, move["pick_a4_mm"])
            total += _move_time_metric(
                move["pick_a4_mm"], move["place_a4_mm"]
            )
            current = move["place_a4_mm"]
        return total + _move_time_metric(current, origin)

    ordered = list(min(permutations(moves), key=route_metric))
    for index, move in enumerate(ordered, start=1):
        move["order"] = index
    return ordered


def _point_on_segment(point: list[float], start: list[float], end: list[float]) -> bool:
    cross = ((end[0] - start[0]) * (point[1] - start[1])
             - (end[1] - start[1]) * (point[0] - start[0]))
    if abs(cross) > 1.0e-7:
        return False
    return (
        min(start[0], end[0]) - 1.0e-7 <= point[0]
        <= max(start[0], end[0]) + 1.0e-7
        and min(start[1], end[1]) - 1.0e-7 <= point[1]
        <= max(start[1], end[1]) + 1.0e-7
    )


def _point_strictly_inside_polygon(
    point: list[float], polygon: list[list[float]]
) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return False
        if (current[1] > point[1]) != (previous[1] > point[1]):
            crossing_x = (
                (previous[0] - current[0])
                * (point[1] - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if point[0] < crossing_x:
                inside = not inside
        previous = current
    return inside


def _segments_cross_strictly(
    first_start: list[float],
    first_end: list[float],
    second_start: list[float],
    second_end: list[float],
) -> bool:
    def orient(a: list[float], b: list[float], c: list[float]) -> float:
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    first_a = orient(first_start, first_end, second_start)
    first_b = orient(first_start, first_end, second_end)
    second_a = orient(second_start, second_end, first_start)
    second_b = orient(second_start, second_end, first_end)
    return (
        first_a * first_b < -1.0e-12
        and second_a * second_b < -1.0e-12
    )


def _polygons_overlap_with_area(
    first: list[list[float]], second: list[list[float]]
) -> bool:
    """Return true for positive-area overlap; touching boundaries are safe."""
    if len(first) < 3 or len(second) < 3:
        return False
    first_box = (
        min(point[0] for point in first),
        max(point[0] for point in first),
        min(point[1] for point in first),
        max(point[1] for point in first),
    )
    second_box = (
        min(point[0] for point in second),
        max(point[0] for point in second),
        min(point[1] for point in second),
        max(point[1] for point in second),
    )
    if (
        min(first_box[1], second_box[1])
        <= max(first_box[0], second_box[0]) + 1.0e-7
        or min(first_box[3], second_box[3])
        <= max(first_box[2], second_box[2]) + 1.0e-7
    ):
        return False
    for first_index, first_start in enumerate(first):
        first_end = first[(first_index + 1) % len(first)]
        for second_index, second_start in enumerate(second):
            second_end = second[(second_index + 1) % len(second)]
            if _segments_cross_strictly(
                first_start, first_end, second_start, second_end
            ):
                return True
    if any(_point_strictly_inside_polygon(point, second) for point in first):
        return True
    if any(_point_strictly_inside_polygon(point, first) for point in second):
        return True
    first_center = [
        sum(point[axis] for point in first) / len(first) for axis in (0, 1)
    ]
    second_center = [
        sum(point[axis] for point in second) / len(second) for axis in (0, 1)
    ]
    return (
        _point_strictly_inside_polygon(first_center, second)
        or _point_strictly_inside_polygon(second_center, first)
    )


def _translated_vertices(
    vertices: list[list[float]], shift_x: float, shift_y: float
) -> list[list[float]]:
    return [
        [float(point[0]) + shift_x, float(point[1]) + shift_y]
        for point in vertices
    ]


def _rotate_vertices_clockwise(
    vertices: list[list[float]],
    center: list[float],
    angle_deg: float,
) -> list[list[float]]:
    """Rotate in A4 image coordinates, where +Y points down."""
    if abs(angle_deg) < 1.0e-9:
        return [[float(point[0]), float(point[1])] for point in vertices]
    radians = math.radians(angle_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotated = []
    for point in vertices:
        delta_x = float(point[0]) - center[0]
        delta_y = float(point[1]) - center[1]
        rotated.append(
            [
                center[0] + cosine * delta_x - sine * delta_y,
                center[1] + sine * delta_x + cosine * delta_y,
            ]
        )
    return rotated


def _make_transfer_move(
    piece: dict[str, Any],
    shift_x: float,
    shift_y: float,
    calibration: StageCalibration,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    center = [float(value) for value in piece["center_mm"]]
    target = [center[0] + shift_x, center[1] + shift_y]
    source_vertices = _piece_vertices(piece)
    rotated_vertices = _rotate_vertices_clockwise(
        source_vertices, center, rotation_deg
    )
    target_vertices = _translated_vertices(rotated_vertices, shift_x, shift_y)
    return {
        "piece_id": str(piece.get("id", "?")),
        "pick_a4_mm": [round(value, 2) for value in center],
        "place_a4_mm": [round(value, 2) for value in target],
        "pick_stage_mm": calibration.a4_to_stage_mm(center),
        "place_stage_mm": calibration.a4_to_stage_mm(target),
        "rotate_deg_clockwise": round(rotation_deg, 2),
        "motor5_rotate_deg": round(
            calibration.rotation_sign * rotation_deg, 2
        ),
        "source_vertices_mm": source_vertices,
        "target_vertices_mm": [
            [round(value, 2) for value in point] for point in target_vertices
        ],
    }


def _validate_transfer_moves(
    moves: list[dict[str, Any]],
    calibration: StageCalibration,
    target_y_min: float,
    target_y_max: float,
) -> tuple[list[dict[str, Any]], str | None]:
    problems: list[dict[str, Any]] = []
    for move in moves:
        piece_id = str(move["piece_id"])
        source_problem = _check_point(
            calibration, move["pick_a4_mm"], piece_id, "PICK"
        )
        target_problem = _check_point(
            calibration, move["place_a4_mm"], piece_id, "PLACE"
        )
        if source_problem is not None:
            problems.append(source_problem)
        if target_problem is not None:
            problems.append(target_problem)
        for point in move["target_vertices_mm"]:
            if (
                point[0] < -1.0e-7
                or point[0] > A4_WIDTH_MM + 1.0e-7
                or point[1] < target_y_min - 1.0e-7
                or point[1] > target_y_max + 1.0e-7
            ):
                problems.append(
                    {
                        "piece_id": piece_id,
                        "operation": "TARGET_VERTEX",
                        "point_a4_mm": point,
                        "allowed_a4_mm": {
                            "x": [0.0, A4_WIDTH_MM],
                            "y": [target_y_min, target_y_max],
                        },
                    }
                )
                break
    for first_index, first in enumerate(moves):
        for second in moves[first_index + 1:]:
            if _polygons_overlap_with_area(
                first["target_vertices_mm"], second["target_vertices_mm"]
            ):
                problems.append(
                    {
                        "piece_id": "%s,%s"
                        % (first["piece_id"], second["piece_id"]),
                        "operation": "TARGET_OVERLAP",
                    }
                )
    if not problems:
        return problems, None
    if any(item["operation"] == "TARGET_OVERLAP" for item in problems):
        return problems, "TARGET_PIECES_OVERLAP"
    if any(item["operation"] == "TARGET_VERTEX" for item in problems):
        return problems, "TARGET_VERTEX_OUT_OF_REGION"
    return problems, "STAGE_POINT_UNREACHABLE"


def _common_translation_moves(
    pieces: list[dict[str, Any]],
    calibration: StageCalibration,
    target_y_min: float,
    target_y_max: float,
    place_near_lower_edge: bool,
) -> tuple[list[dict[str, Any]] | None, float | None, dict[str, float]]:
    vertices = [
        point for piece in pieces for point in _piece_vertices(piece)
    ]
    min_source_y = min(point[1] for point in vertices)
    max_source_y = max(point[1] for point in vertices)
    shift_min = target_y_min - min_source_y
    shift_max = target_y_max - max_source_y
    diagnostics = {
        "minimum_shift_mm": round(shift_min, 2),
        "maximum_shift_mm": round(shift_max, 2),
        "source_span_mm": round(max_source_y - min_source_y, 2),
        "target_height_mm": round(target_y_max - target_y_min, 2),
    }
    if shift_min > shift_max + 1.0e-7:
        return None, None, diagnostics
    # Upper destination: put the lowest vertex just above the divider.
    # Lower destination: put the highest source group just below it.
    shift_y = shift_max if place_near_lower_edge else shift_min
    moves = [
        _make_transfer_move(piece, 0.0, shift_y, calibration)
        for piece in pieces
    ]
    problems, validation_error = _validate_transfer_moves(
        moves, calibration, target_y_min, target_y_max
    )
    # Keep a geometrically valid plan even when a source centroid is outside
    # the calibrated slide range.  The caller then returns the useful
    # STAGE_POINT_UNREACHABLE diagnostics instead of misreporting a packing
    # failure.  Geometry failures still make common translation unusable.
    if problems and validation_error != "STAGE_POINT_UNREACHABLE":
        return None, None, diagnostics
    return moves, shift_y, diagnostics


def _row_groups(
    ordered: tuple[dict[str, Any], ...]
) -> Iterable[list[list[dict[str, Any]]]]:
    if not ordered:
        return
    for break_mask in range(1 << (len(ordered) - 1)):
        rows: list[list[dict[str, Any]]] = [[ordered[0]]]
        for index, piece in enumerate(ordered[1:]):
            if break_mask & (1 << index):
                rows.append([])
            rows[-1].append(piece)
        yield rows


def _rotated_piece_geometry(
    piece: dict[str, Any], angle_deg: float
) -> dict[str, Any]:
    center = [float(value) for value in piece["center_mm"]]
    vertices = _rotate_vertices_clockwise(
        _piece_vertices(piece), center, angle_deg
    )
    min_x = min(point[0] for point in vertices)
    max_x = max(point[0] for point in vertices)
    min_y = min(point[1] for point in vertices)
    max_y = max(point[1] for point in vertices)
    return {
        "piece": piece,
        "angle_deg": float(angle_deg),
        "vertices": vertices,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def _fixed_lattice_candidates(
    minimum: float,
    maximum: float,
    step: float,
) -> list[float]:
    """Return points on an A4-fixed lattice inside an interval.

    The old grid started at ``minimum`` (a measured polygon edge), so a
    0.2 mm contour change translated every candidate and could also make the
    first feasible packing jump by one whole 4 mm cell.  Anchoring the grid at
    the A4 origin makes target centroids independent of camera-frame noise.
    """
    if maximum < minimum - 1.0e-7 or step <= 0.0:
        return []
    first_index = int(math.ceil((minimum - 1.0e-7) / step))
    last_index = int(math.floor((maximum + 1.0e-7) / step))
    return [round(index * step, 6) for index in range(first_index, last_index + 1)]


def _point_to_segment_distance(
    point: list[float], start: list[float], end: list[float]
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 1.0e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    ratio = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / length_squared
    ratio = min(1.0, max(0.0, ratio))
    closest_x = start[0] + ratio * delta_x
    closest_y = start[1] + ratio * delta_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def _polygons_have_clearance(
    first: list[list[float]],
    second: list[list[float]],
    clearance_mm: float,
) -> bool:
    """Check non-overlap with a small contour-noise guard band."""
    if _polygons_overlap_with_area(first, second):
        return False
    if clearance_mm <= 0.0:
        return True
    minimum_distance = math.inf
    for polygon, other in ((first, second), (second, first)):
        for point in polygon:
            for index, start in enumerate(other):
                end = other[(index + 1) % len(other)]
                minimum_distance = min(
                    minimum_distance,
                    _point_to_segment_distance(point, start, end),
                )
                if minimum_distance < clearance_mm - 1.0e-7:
                    return False
    return minimum_distance >= clearance_mm - 1.0e-7


class Mode1TargetLatch:
    """Hold one validated transfer layout while the source pieces stay valid.

    The row or polygon packer may find several equally safe layouts.  Re-running the
    search on a contour that changes by a fraction of a millimetre can choose a
    different layout even though the pieces have not moved.  Once one layout
    passes the full boundary, reachability, overlap, and clearance checks, keep
    its target centres, rotations, and execution order.  Live pick centres and
    contours are still rebuilt every frame and revalidated before execution.
    """

    def __init__(self, clearance_mm: float = 3.0) -> None:
        self.clearance_mm = float(clearance_mm)
        self._anchors: dict[str, tuple[list[float], float]] = {}
        self._order: list[str] = []
        self._revision = 0

    def clear(self) -> None:
        self._anchors.clear()
        self._order.clear()

    @staticmethod
    def _eligible(plan: dict[str, Any]) -> bool:
        return bool(
            plan.get("ready", False)
            and plan.get("source_region") == "lower"
            and plan.get("strategy") in {
                "BBOX_ROW_PACKING", "POLYGON_GRID_PACKING"
            }
            and len(plan.get("moves") or []) == 4
        )

    @staticmethod
    def _candidate_signature(
        plan: dict[str, Any],
    ) -> dict[str, float] | None:
        try:
            return {
                str(move["piece_id"]): float(move["rotate_deg_clockwise"])
                for move in plan["moves"]
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _adopt(self, plan: dict[str, Any]) -> bool:
        signature = self._candidate_signature(plan)
        if signature is None or len(signature) != 4:
            return False
        moves = sorted(plan["moves"], key=lambda move: int(move.get("order", 0)))
        try:
            anchors = {
                str(move["piece_id"]): (
                    [
                        float(move["place_a4_mm"][0]),
                        float(move["place_a4_mm"][1]),
                    ],
                    float(move["rotate_deg_clockwise"]),
                )
                for move in moves
            }
        except (KeyError, TypeError, ValueError, IndexError):
            return False
        if len(anchors) != 4:
            return False
        self._anchors = anchors
        self._order = [str(move["piece_id"]) for move in moves]
        self._revision += 1
        return True

    def _rebuild(
        self,
        plan: dict[str, Any],
        status: dict[str, Any],
        calibration: StageCalibration,
    ) -> dict[str, Any] | None:
        pieces = {
            str(piece.get("id", "?")): piece
            for piece in status.get("pieces", [])
        }
        if set(pieces) != set(self._anchors) or len(pieces) != 4:
            return None
        try:
            target_y_min, target_y_max = (
                float(value) for value in plan["target_y_range_mm"]
            )
        except (KeyError, TypeError, ValueError):
            return None

        moves: list[dict[str, Any]] = []
        for order, piece_id in enumerate(self._order, start=1):
            piece = pieces[piece_id]
            center = [float(value) for value in piece["center_mm"]]
            target, rotation = self._anchors[piece_id]
            move = _make_transfer_move(
                piece,
                target[0] - center[0],
                target[1] - center[1],
                calibration,
                rotation,
            )
            move["order"] = order
            moves.append(move)

        problems, _error = _validate_transfer_moves(
            moves, calibration, target_y_min, target_y_max
        )
        if problems:
            return None
        for first_index, first in enumerate(moves):
            for second in moves[first_index + 1:]:
                if not _polygons_have_clearance(
                    first["target_vertices_mm"],
                    second["target_vertices_mm"],
                    self.clearance_mm,
                ):
                    return None

        latched = deepcopy(plan)
        latched["moves"] = moves
        latched["unreachable"] = []
        packing = dict(latched.get("packing") or {})
        packing.update(
            {
                "target_latched": True,
                "target_latch_revision": self._revision,
                "target_clearance_mm": round(self.clearance_mm, 2),
            }
        )
        latched["packing"] = packing
        return latched

    def update(
        self,
        plan: dict[str, Any],
        status: dict[str, Any],
        calibration: StageCalibration,
    ) -> dict[str, Any]:
        if not self._eligible(plan):
            self.clear()
            return plan

        signature = self._candidate_signature(plan)
        current_signature = {
            piece_id: anchor[1] for piece_id, anchor in self._anchors.items()
        }
        if not self._anchors or signature != current_signature:
            self.clear()
            if not self._adopt(plan):
                return plan

        rebuilt = self._rebuild(plan, status, calibration)
        if rebuilt is not None:
            return rebuilt

        # A real orientation or divider change can invalidate the old layout.
        # Re-lock only to a freshly generated plan that also passes the same
        # conservative live-contour checks.
        self.clear()
        if not self._adopt(plan):
            return plan
        rebuilt = self._rebuild(plan, status, calibration)
        if rebuilt is None:
            self.clear()
            return plan
        return rebuilt


def _grid_pack_rotated_polygons(
    geometries: list[dict[str, Any]],
    calibration: StageCalibration,
    x_min: float,
    x_max: float,
    target_y_min: float,
    target_y_max: float,
    step_mm: float = 4.0,
) -> tuple[list[dict[str, Any]] | None, tuple[float, float] | None]:
    # Contours and centroids vary by a few tenths of a millimetre frame to
    # frame.  Keep every accepted placement at least this far from a boundary
    # and from another piece, so feasibility cannot toggle on normal noise.
    noise_guard_mm = 5.0
    candidate_sets: dict[str, list[tuple[float, float, dict[str, Any]]]] = {}
    for item in geometries:
        piece_id = str(item["piece"].get("id", "?"))
        source_center = [float(value) for value in item["piece"]["center_mm"]]
        relative_min_x = item["min_x"] - source_center[0]
        relative_max_x = item["max_x"] - source_center[0]
        relative_min_y = item["min_y"] - source_center[1]
        relative_max_y = item["max_y"] - source_center[1]
        center_x_values = _fixed_lattice_candidates(
            x_min + noise_guard_mm - relative_min_x,
            x_max - noise_guard_mm - relative_max_x,
            step_mm,
        )
        center_y_values = _fixed_lattice_candidates(
            target_y_min + noise_guard_mm - relative_min_y,
            target_y_max - noise_guard_mm - relative_max_y,
            step_mm,
        )
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for target_center_x in center_x_values:
            for target_center_y in center_y_values:
                move = _make_transfer_move(
                    item["piece"],
                    target_center_x - source_center[0],
                    target_center_y - source_center[1],
                    calibration,
                    item["angle_deg"],
                )
                problems, _error = _validate_transfer_moves(
                    [move], calibration, target_y_min, target_y_max
                )
                if problems:
                    continue
                direct = _move_time_metric(
                    move["pick_a4_mm"], move["place_a4_mm"]
                )
                divider_distance = target_y_max - max(
                    point[1] for point in move["target_vertices_mm"]
                )
                candidates.append((direct, divider_distance, move))
        # Candidate order is deliberately independent of pick position and
        # exact contour dimensions.  The target centres are fixed lattice
        # points; normal measurement jitter therefore produces the same DFS
        # path and the same first feasible layout.
        candidates.sort(
            key=lambda entry: (
                entry[2]["place_a4_mm"][0],
                entry[2]["place_a4_mm"][1],
            )
        )
        # The strategic boundary positions plus the 4 mm grid provide far
        # more candidates than four pieces need.  A cap keeps worst-case
        # backtracking bounded on Raspberry Pi while retaining low-travel
        # placements first.
        candidate_sets[piece_id] = candidates[:320]
        if not candidate_sets[piece_id]:
            return None, None

    ordered = sorted(
        geometries,
        key=lambda item: str(item["piece"].get("id", "?")),
    )
    minimum_remaining = [0.0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        piece_id = str(ordered[index]["piece"].get("id", "?"))
        minimum_remaining[index] = (
            minimum_remaining[index + 1] + candidate_sets[piece_id][0][0]
        )

    best_score: tuple[float, float] | None = None
    best_moves: list[dict[str, Any]] | None = None
    selected: list[dict[str, Any]] = []
    visited = 0
    visit_limit = 300_000

    def search(index: int, direct_cost: float, divider_cost: float) -> None:
        nonlocal best_score, best_moves, visited
        # Candidate order is already low-travel first.  Stop at the first
        # complete packing for this signed rotation assignment; the caller
        # still compares that score across every assignment having the same
        # minimum total rotation.
        if best_moves is not None or visited >= visit_limit:
            return
        visited += 1
        if (
            best_score is not None
            and direct_cost + minimum_remaining[index] > best_score[0] + 1.0e-7
        ):
            return
        if index == len(ordered):
            score = (round(direct_cost, 6), round(divider_cost, 6))
            if best_score is None or score < best_score:
                best_score = score
                best_moves = list(selected)
            return

        piece_id = str(ordered[index]["piece"].get("id", "?"))
        for direct, divider_distance, move in candidate_sets[piece_id]:
            if any(
                not _polygons_have_clearance(
                    move["target_vertices_mm"],
                    placed["target_vertices_mm"],
                    noise_guard_mm,
                )
                for placed in selected
            ):
                continue
            selected.append(move)
            search(
                index + 1,
                direct_cost + direct,
                divider_cost + divider_distance,
            )
            selected.pop()
            if best_moves is not None:
                return

    search(0, 0.0, 0.0)
    return best_moves, best_score


def _pack_upper_region(
    pieces: list[dict[str, Any]],
    calibration: StageCalibration,
    target_y_min: float,
    target_y_max: float,
    margin_mm: float,
    gap_mm: float = TRANSFER_PACKING_GAP_MM,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    x_min = max(float(margin_mm), calibration.x_min_a4_mm)
    x_max = min(A4_WIDTH_MM - float(margin_mm), calibration.x_max_a4_mm)
    available_width = x_max - x_min
    available_height = target_y_max - target_y_min
    packing_guard_mm = 3.0
    guarded_width = available_width - 2.0 * packing_guard_mm
    guarded_height = available_height - 2.0 * packing_guard_mm
    if any(not _piece_vertices(piece) for piece in pieces):
        return None, {"reason": "PIECE_GEOMETRY_MISSING"}

    # Mode 1 only transfers every piece across the divider.  Rotation is not
    # part of the first question and would add unnecessary motor-5 motion.
    # Search only translations of the live measured contours.  If the current
    # orientations cannot fit safely, reject the plan instead of silently
    # rotating a piece.
    angle_sets = [(0.0,) * len(pieces)]
    rotation_costs = sorted(
        {sum(abs(angle) for angle in angles) for angles in angle_sets}
    )
    for rotation_cost in rotation_costs:
        best: tuple[tuple[float, ...], list[dict[str, Any]], str] | None = None
        geometry_options: list[list[dict[str, Any]]] = []
        for angles in angle_sets:
            if abs(sum(abs(angle) for angle in angles) - rotation_cost) > 1.0e-7:
                continue
            geometries = [
                _rotated_piece_geometry(piece, angle)
                for piece, angle in zip(pieces, angles)
            ]
            if any(
                item["width"] > available_width + 1.0e-7
                or item["height"] > available_height + 1.0e-7
                for item in geometries
            ):
                continue
            geometry_options.append(geometries)

            # Fast and mechanically conservative first choice: non-overlapping
            # bounding boxes arranged in rows immediately above the divider.
            for ordered in permutations(geometries):
                for rows in _row_groups(ordered):
                    row_widths = [
                        sum(item["width"] for item in row)
                        + gap_mm * max(0, len(row) - 1)
                        for row in rows
                    ]
                    row_heights = [
                        max(item["height"] for item in row) for row in rows
                    ]
                    total_height = (
                        sum(row_heights) + gap_mm * max(0, len(rows) - 1)
                    )
                    if (
                        any(
                            width > guarded_width + 1.0e-7
                            for width in row_widths
                        )
                        or total_height > guarded_height + 1.0e-7
                    ):
                        continue
                    moves: list[dict[str, Any]] = []
                    row_bottom = target_y_max - packing_guard_mm
                    for row, row_width, row_height in zip(
                        rows, row_widths, row_heights
                    ):
                        cursor_x = (
                            x_min
                            + packing_guard_mm
                            + 0.5 * (guarded_width - row_width)
                        )
                        for item in row:
                            moves.append(
                                _make_transfer_move(
                                    item["piece"],
                                    cursor_x - item["min_x"],
                                    row_bottom - item["max_y"],
                                    calibration,
                                    item["angle_deg"],
                                )
                            )
                            cursor_x += item["width"] + gap_mm
                        row_bottom -= row_height + gap_mm
                    problems, _error = _validate_transfer_moves(
                        moves, calibration, target_y_min, target_y_max
                    )
                    if problems:
                        continue
                    direct_travel = sum(
                        _move_time_metric(
                            move["pick_a4_mm"], move["place_a4_mm"]
                        )
                        for move in moves
                    )
                    divider_distance = sum(
                        target_y_max
                        - max(point[1] for point in move["target_vertices_mm"])
                        for move in moves
                    )
                    # +90 and -90 have identical absolute rotation time but
                    # can exchange rank after a sub-millimetre contour change.
                    # Prefer +90 deterministically; keep -90 as a fallback when
                    # geometry genuinely requires it.
                    negative_quarter_turns = sum(
                        1 for item in geometries if item["angle_deg"] < 0.0
                    )
                    score = (
                        float(negative_quarter_turns),
                        round(direct_travel, 6),
                        round(divider_distance, 6),
                        round(total_height, 6),
                    )
                    if best is None or score < best[0]:
                        best = (score, moves, "BBOX_ROW_PACKING")

        # Prefer a bbox-separated row solution whenever one exists at this
        # minimum rotation cost.  Only slender/interlocking shapes need the
        # more expensive polygon grid search.
        if best is None:
            for geometries in geometry_options:
                polygon_moves, polygon_score = _grid_pack_rotated_polygons(
                    geometries,
                    calibration,
                    x_min,
                    x_max,
                    target_y_min,
                    target_y_max,
                )
                if polygon_moves is None or polygon_score is None:
                    continue
                negative_quarter_turns = sum(
                    1 for item in geometries if item["angle_deg"] < 0.0
                )
                # Select between equal-cost signed rotation assignments by
                # categorical rotation and fixed target coordinates, never by
                # the noisy pick-to-place distance.
                canonical_moves = sorted(
                    polygon_moves, key=lambda move: str(move["piece_id"])
                )
                rotation_key = tuple(
                    0.0
                    if abs(float(move["rotate_deg_clockwise"])) < 1.0e-7
                    else (
                        1.0
                        if float(move["rotate_deg_clockwise"]) > 0.0
                        else 2.0
                    )
                    for move in canonical_moves
                )
                placement_key = tuple(
                    coordinate
                    for move in canonical_moves
                    for coordinate in (
                        round(float(move["place_a4_mm"][0]), 6),
                        round(float(move["place_a4_mm"][1]), 6),
                    )
                )
                score = (
                    float(negative_quarter_turns),
                    *rotation_key,
                    *placement_key,
                )
                if best is None or score < best[0]:
                    best = (score, polygon_moves, "POLYGON_GRID_PACKING")

        if best is not None:
            selected_angles = {
                move["piece_id"]: move["rotate_deg_clockwise"]
                for move in best[1]
            }
            return best[1], {
                "reason": None,
                "strategy": best[2],
                "available_size_mm": [
                    round(available_width, 2), round(available_height, 2)
                ],
                "gap_mm": round(gap_mm, 2),
                "rotation_total_deg": round(rotation_cost, 2),
                "rotations_deg": selected_angles,
            }

    return None, {
        "reason": "NO_UNROTATED_PACKING_LAYOUT",
        "available_size_mm": [
            round(available_width, 2), round(available_height, 2)
        ],
        "gap_mm": round(gap_mm, 2),
        "allowed_rotations_deg": [0.0],
    }


def build_transfer_plan(
    status: dict[str, Any],
    calibration: StageCalibration = StageCalibration(),
    margin_mm: float = TRANSFER_EDGE_MARGIN_MM,
) -> dict[str, Any]:
    """Plan mode 1 in either direction using translation only.

    ``status['source_region'] == 'lower'`` is the corrected physical setup:
    the pieces start below the divider and are packed into the reachable upper
    band.  Both common translation and the lower-to-upper packing fallback keep
    motor 5 at zero.  The historical upper-to-lower direction remains the
    default for captures without ``source_region``.
    """
    if not bool(
        status.get("a4_physical_orientation_locked", False)
        or (status.get("config") or {}).get("a4_locked", False)
    ):
        return _not_ready(1, "A4_PHYSICAL_ORIENTATION_NOT_LOCKED")
    if not status.get("a4_found", False):
        return _not_ready(1, "A4_NOT_READY")
    if not status.get("divider_found", False):
        return _not_ready(1, "DIVIDER_NOT_READY")
    if not status.get("stable_four_pieces", False):
        return _not_ready(1, "WAITING_FOR_STABLE_FOUR_PIECES")

    pieces = list(status.get("pieces") or [])
    if len(pieces) != 4:
        return _not_ready(1, "NEED_EXACTLY_FOUR_PIECES")
    divider_y_mm = status.get("divider_y_mm")
    if divider_y_mm is None:
        return _not_ready(1, "DIVIDER_NOT_READY")
    divider_y_mm = float(divider_y_mm)

    source_region = str(status.get("source_region", "upper")).lower()
    if source_region not in {"upper", "lower"}:
        return _not_ready(1, "INVALID_SOURCE_REGION")

    all_vertices = [
        vertex
        for piece in pieces
        for vertex in _piece_vertices(piece)
    ]
    if not all_vertices:
        return _not_ready(1, "PIECE_GEOMETRY_MISSING")
    min_source_y = min(point[1] for point in all_vertices)
    max_source_y = max(point[1] for point in all_vertices)
    if source_region == "lower" and min_source_y < divider_y_mm - 1.0e-7:
        return _not_ready(1, "PIECE_OUTSIDE_LOWER_SOURCE_REGION")
    if source_region == "upper" and max_source_y > divider_y_mm + 1.0e-7:
        return _not_ready(1, "PIECE_OUTSIDE_UPPER_SOURCE_REGION")

    if source_region == "lower":
        # The slide's Y minimum constrains the magnet centre, not every edge of
        # the carried plate.  Let the unrotated contour use the full upper A4
        # half; _check_point still rejects an unreachable target centroid.
        target_y_min = margin_mm
        target_y_max = divider_y_mm - margin_mm
        name = "MOVE_ALL_TO_UPPER_REGION"
        place_near_lower_edge = True
    else:
        target_y_min = divider_y_mm + margin_mm
        target_y_max = A4_HEIGHT_MM - margin_mm
        name = "MOVE_ALL_TO_LOWER_REGION"
        place_near_lower_edge = False
    if target_y_min > target_y_max:
        return _not_ready(1, "TARGET_REGION_TOO_SMALL")

    moves, shift_y_mm, common_diagnostics = _common_translation_moves(
        pieces,
        calibration,
        target_y_min,
        target_y_max,
        place_near_lower_edge,
    )
    strategy = "COMMON_TRANSLATION"
    packing: dict[str, Any] | None = None
    if moves is not None and source_region == "lower":
        common_problems, common_error = _validate_transfer_moves(
            moves, calibration, target_y_min, target_y_max
        )
        if common_problems and common_error == "STAGE_POINT_UNREACHABLE":
            moves = None
            shift_y_mm = None
    if moves is None and source_region == "lower":
        moves, packing = _pack_upper_region(
            pieces,
            calibration,
            target_y_min,
            target_y_max,
            margin_mm,
        )
        strategy = (
            "BBOX_ROW_PACKING"
            if packing is None
            else str(packing.get("strategy") or "ROTATED_PACKING")
        )
    if moves is None:
        error = (
            "UPPER_REGION_PACKING_FAILED"
            if source_region == "lower"
            else "LOWER_REGION_CANNOT_FIT_COMMON_TRANSLATION"
        )
        return _not_ready(
            1,
            error,
            source_region=source_region,
            divider_y_mm=round(divider_y_mm, 2),
            target_y_range_mm=[round(target_y_min, 2), round(target_y_max, 2)],
            common_translation=common_diagnostics,
            packing=packing,
        )

    unreachable, validation_error = _validate_transfer_moves(
        moves, calibration, target_y_min, target_y_max
    )

    moves = _nearest_pick_order(
        moves, [calibration.home_x_a4_mm, calibration.home_y_a4_mm]
    )
    return {
        "mode": 1,
        "name": name,
        "source_region": source_region,
        "strategy": strategy,
        "ready": not unreachable,
        "error": None if not unreachable else validation_error,
        "divider_y_mm": round(divider_y_mm, 2),
        "target_y_range_mm": [round(target_y_min, 2), round(target_y_max, 2)],
        "common_translation_mm": (
            None
            if shift_y_mm is None
            else [0.0, round(shift_y_mm, 2)]
        ),
        "common_translation": common_diagnostics,
        "packing": packing,
        "margin_mm": round(margin_mm, 2),
        "stage_calibration": asdict(calibration),
        "unreachable": unreachable,
        "moves": moves,
    }


def build_assembly_motion_plan(
    status: dict[str, Any],
    calibration: StageCalibration = StageCalibration(),
) -> dict[str, Any]:
    """Plan mode 2 from the stable figure-2 assembly solution."""
    if not bool(
        status.get("a4_physical_orientation_locked", False)
        or (status.get("config") or {}).get("a4_locked", False)
    ):
        return _not_ready(2, "A4_PHYSICAL_ORIENTATION_NOT_LOCKED")
    if not status.get("stable_four_pieces", False):
        return _not_ready(2, "WAITING_FOR_STABLE_FOUR_PIECES")
    source = status.get("assembly_plan")
    if not source or not source.get("ready", False):
        return _not_ready(
            2,
            "ASSEMBLY_PLAN_NOT_READY",
            assembly_error=None if not source else source.get("error"),
        )

    moves: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []
    for source_move in source.get("moves", []):
        piece_id = str(source_move.get("piece_id", "?"))
        pick = calibration.clamp_visual_boundary_noise(
            source_move["current_centroid_mm"]
        )
        place = calibration.clamp_visual_boundary_noise(
            source_move["target_centroid_mm"]
        )
        source_problem = _check_point(calibration, pick, piece_id, "PICK")
        target_problem = _check_point(calibration, place, piece_id, "PLACE")
        if source_problem is not None:
            unreachable.append(source_problem)
        if target_problem is not None:
            unreachable.append(target_problem)
        rotation = float(source_move["rotate_deg_clockwise"])
        rotation = (rotation + 180.0) % 360.0 - 180.0
        moves.append(
            {
                "order": int(source_move["order"]),
                "piece_id": piece_id,
                "target_piece": source_move.get("target_piece"),
                "pick_a4_mm": [round(value, 2) for value in pick],
                "place_a4_mm": [round(value, 2) for value in place],
                "pick_stage_mm": calibration.a4_to_stage_mm(pick),
                "place_stage_mm": calibration.a4_to_stage_mm(place),
                "rotate_deg_clockwise": round(rotation, 2),
                "motor5_rotate_deg": round(
                    calibration.rotation_sign * rotation, 2
                ),
                "target_vertices_mm": source_move.get("target_vertices_mm", []),
                "shape_residual": source_move.get("shape_residual"),
            }
        )
    moves.sort(key=lambda move: int(move["order"]))
    return {
        "mode": 2,
        "name": "ASSEMBLE_FIGURE_2_RECTANGLE",
        "ready": not unreachable and len(moves) == 4,
        "error": (
            None
            if not unreachable and len(moves) == 4
            else (
                "STAGE_POINT_UNREACHABLE"
                if unreachable
                else "ASSEMBLY_MOVE_COUNT_INVALID"
            )
        ),
        "target_rectangle_mm": source.get("target_rectangle_mm"),
        "stage_calibration": asdict(calibration),
        "unreachable": unreachable,
        "moves": moves,
    }


def build_task_plan(
    mode: int,
    status: dict[str, Any],
    calibration: StageCalibration = StageCalibration(),
) -> dict[str, Any]:
    if mode == 1:
        return build_transfer_plan(status, calibration)
    if mode == 2:
        return build_assembly_motion_plan(status, calibration)
    return _not_ready(mode, "UNKNOWN_MODE")


def _profile_duration_seconds(
    distance_revolutions: float,
    speed_rpm: float,
    acceleration_level: int,
    margin_seconds: float = 0.8,
) -> float:
    """Mirror the Emm V5 triangular/trapezoidal curve used by the MCU."""
    distance = max(0.0, float(distance_revolutions))
    speed = float(speed_rpm)
    if distance <= 0.0 or speed <= 0.0:
        return 0.0
    if acceleration_level <= 0:
        return distance * 60.0 / speed + margin_seconds

    difference = 256.0 - float(min(255, acceleration_level))
    acceleration_rpm_s = 20000.0 / difference
    velocity_rev_s = speed / 60.0
    acceleration_rev_s2 = acceleration_rpm_s / 60.0
    acceleration_time = velocity_rev_s / acceleration_rev_s2
    ramp_distance = velocity_rev_s * acceleration_time
    if distance >= ramp_distance:
        base = distance / velocity_rev_s + acceleration_time
    else:
        base = 2.0 * math.sqrt(distance / acceleration_rev_s2)
    return base + margin_seconds


def _xy_profile_duration_seconds(
    start: Iterable[float],
    target: Iterable[float],
    max_speed_rpm: int = 600,
    min_speed_rpm: int = 60,
    acceleration_level: int = 220,
    slide_lead_mm: float = 4.0,
    settle_seconds: float = 0.5,
) -> float:
    sx, sy = (float(value) for value in start)
    tx, ty = (float(value) for value in target)
    distances = [abs(tx - sx), abs(ty - sy)]
    maximum = max(distances)
    if maximum <= 0.0:
        return 0.0

    durations: list[float] = []
    for distance in distances:
        if distance <= 0.0:
            continue
        speed = int(math.floor(max_speed_rpm * distance / maximum + 0.5))
        speed = max(min_speed_rpm, min(max_speed_rpm, speed))
        durations.append(
            _profile_duration_seconds(
                distance / slide_lead_mm,
                speed,
                acceleration_level,
                margin_seconds=0.8 + settle_seconds,
            )
        )
    return max(durations, default=0.0)


def estimate_plan_seconds(
    plan: dict[str, Any],
    calibration: StageCalibration = StageCalibration(),
) -> float:
    """Estimate the same acceleration-aware profile used by the MCU firmware."""
    if not plan.get("moves"):
        return 0.0
    z_motion_seconds = _profile_duration_seconds(
        distance_revolutions=8.0 / 4.0,
        speed_rpm=220.0,
        acceleration_level=220,
    )
    pick_or_place_seconds = 2.0 * z_motion_seconds + 0.4
    current = [calibration.home_x_a4_mm, calibration.home_y_a4_mm]
    total = 0.0
    for move in plan["moves"]:
        pick = move["pick_a4_mm"]
        place = move["place_a4_mm"]
        total += _xy_profile_duration_seconds(current, pick)
        total += pick_or_place_seconds
        rotation = abs(float(move.get("motor5_rotate_deg", 0.0)))
        if rotation > 0.0:
            rotation_seconds = _profile_duration_seconds(
                distance_revolutions=rotation / 360.0,
                speed_rpm=120.0,
                acceleration_level=5,
            )
            total += 2.0 * rotation_seconds
        total += _xy_profile_duration_seconds(pick, place)
        total += pick_or_place_seconds
        current = list(place)
    total += _xy_profile_duration_seconds(
        current,
        [calibration.home_x_a4_mm, calibration.home_y_a4_mm],
    )
    return round(total, 1)


__all__ = [
    "MotionPlanStabilityGate",
    "StageCalibration",
    "build_assembly_motion_plan",
    "build_task_plan",
    "build_transfer_plan",
    "estimate_plan_seconds",
]
