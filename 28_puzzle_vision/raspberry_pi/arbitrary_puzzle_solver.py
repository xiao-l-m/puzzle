#!/usr/bin/env python3
"""Geometry-only solver for E-question 2(1) arbitrary white pieces.

The input polygons are measured in the physical A4 coordinate system.  The
solver never mirrors a polygon: it only applies rigid rotations/translations,
matches reversed edges, verifies the reconstructed rectangular outline, adds
a small mechanical seam, and places the result in the reachable upper half.

OpenCV is intentionally not required here so the combinatorial geometry can
be unit-tested on a development PC without a camera stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Any, Iterable

import numpy as np


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_DIVIDER_Y_MM = 148.5
DEFAULT_DIVIDER_MARGIN_MM = 8.0
DEFAULT_SEAM_GAP_MM = 4.0
# The official requirement is no more than 20 mm.  Exact full-edge layouts
# normally stay near the requested 4 mm seam; composite T-junction layouts
# may need most of the official allowance to absorb camera contour error.
MAX_ADJACENT_VERTEX_GAP_MM = 20.0
EDGE_MATCH_TOLERANCE_MM = 3.0
BOUNDARY_TOLERANCE_MM = 3.0
MIN_TARGET_WIDTH_MM = 90.0
MAX_TARGET_WIDTH_MM = 120.0
MIN_TARGET_HEIGHT_MM = 50.0
MAX_TARGET_HEIGHT_MM = 90.0
# A4 homography and white-mask contour fitting can make a physical dimension
# read a few percent large.  The current 100 x 90 mm field set measures about
# 102.4 x 92.4 mm, so the previous hidden 2 mm allowance was too narrow.
DIMENSION_CALIBRATION_TOLERANCE_MM = 5.0
MAX_LAYOUT_SOLUTIONS = 1000
COMPOSITE_EDGE_TOLERANCE_MM = 8.0
COMPOSITE_JUNCTION_TOLERANCE_MM = 18.0
COMPOSITE_COLLINEAR_TOLERANCE_DEG = 18.0
MULTI_SEGMENT_CONTACT_TOLERANCE_MM = 8.0
# Boundary placement is only a fallback after the much faster exact/partial
# edge matcher.  A 6 mm grid has at most 3 mm quantization error, comfortably
# inside the official 20 mm corresponding-vertex limit and the solver's later
# contact validation, while avoiding thousands of nearly identical raster
# states on the Raspberry Pi.
BOUNDARY_POSITION_STEP_MM = 6.0
BOUNDARY_BEAM_WIDTH = 96


@dataclass(frozen=True)
class RigidTransform:
    angle_rad: float
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return rotate_points(points, self.angle_rad) + self.translation


@dataclass(frozen=True)
class SeamMatch:
    first_piece: int
    first_edge: int
    second_piece: int
    second_edge: int
    length_error_mm: float


@dataclass(frozen=True)
class BoundaryPlacement:
    transform: RigidTransform
    polygon: np.ndarray
    edge_index: int
    side: int
    touches_corner: bool


def rotate_points(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    matrix = np.asarray([[cosine, -sine], [sine, cosine]], np.float64)
    return np.asarray(points, np.float64) @ matrix.T


def normalize_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def signed_area(points: np.ndarray) -> float:
    polygon = np.asarray(points, np.float64).reshape(-1, 2)
    following = np.roll(polygon, -1, axis=0)
    return 0.5 * float(
        np.sum(polygon[:, 0] * following[:, 1]
               - following[:, 0] * polygon[:, 1])
    )


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_area(points))


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(points, np.float64).reshape(-1, 2)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    denominator = float(np.sum(cross))
    if abs(denominator) < 1.0e-9:
        return np.mean(polygon, axis=0)
    return np.asarray(
        [
            np.sum((polygon[:, 0] + following[:, 0]) * cross),
            np.sum((polygon[:, 1] + following[:, 1]) * cross),
        ],
        np.float64,
    ) / (3.0 * denominator)


def canonical_polygon(points: Iterable[Iterable[float]]) -> np.ndarray:
    polygon = np.asarray(list(points), np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        raise ValueError("POLYGON_NEEDS_AT_LEAST_THREE_VERTICES")
    if not np.all(np.isfinite(polygon)):
        raise ValueError("POLYGON_HAS_NONFINITE_VERTEX")
    if signed_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()
    start = min(
        range(len(polygon)),
        key=lambda index: (round(float(polygon[index, 1]), 6),
                           round(float(polygon[index, 0]), 6)),
    )
    return np.roll(polygon, -start, axis=0)


def edge_points(polygon: np.ndarray, edge_index: int) -> tuple[np.ndarray, np.ndarray]:
    return polygon[edge_index], polygon[(edge_index + 1) % len(polygon)]


def edge_length(polygon: np.ndarray, edge_index: int) -> float:
    start, end = edge_points(polygon, edge_index)
    return float(np.linalg.norm(end - start))


def align_reversed_edge(
    moving: np.ndarray,
    moving_edge: int,
    fixed: np.ndarray,
    fixed_edge: int,
    match_other_endpoint: bool = False,
) -> RigidTransform:
    moving_start, moving_end = edge_points(moving, moving_edge)
    fixed_start, fixed_end = edge_points(fixed, fixed_edge)
    source_vector = moving_end - moving_start
    target_vector = fixed_start - fixed_end
    angle = normalize_angle_rad(
        math.atan2(float(target_vector[1]), float(target_vector[0]))
        - math.atan2(float(source_vector[1]), float(source_vector[0]))
    )
    moving_anchor = moving_end if match_other_endpoint else moving_start
    fixed_anchor = fixed_start if match_other_endpoint else fixed_end
    rotated_anchor = rotate_points(moving_anchor.reshape(1, 2), angle)[0]
    translation = fixed_anchor - rotated_anchor
    return RigidTransform(angle, translation)


def align_reversed_edge_to_segment(
    moving: np.ndarray,
    moving_edge: int,
    target_start: np.ndarray,
    target_end: np.ndarray,
) -> RigidTransform:
    """Place an observed edge against one part of a longer opposite edge.

    Only a rigid transform is returned: the moving polygon is never reflected
    and its measured edge length is never scaled.  ``target_start`` and
    ``target_end`` define the direction of the occupied sub-segment; the
    moving edge is aligned in the reverse direction so the two polygon
    interiors lie on opposite sides of the seam.
    """
    moving_start, moving_end = edge_points(moving, moving_edge)
    source_vector = moving_end - moving_start
    target_vector = np.asarray(target_start) - np.asarray(target_end)
    angle = normalize_angle_rad(
        math.atan2(float(target_vector[1]), float(target_vector[0]))
        - math.atan2(float(source_vector[1]), float(source_vector[0]))
    )
    rotated_start = rotate_points(moving_start.reshape(1, 2), angle)[0]
    translation = np.asarray(target_end, np.float64) - rotated_start
    return RigidTransform(angle, translation)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = b - a
    second = c - a
    # NumPy 2.0 deprecated, and newer releases reject, ``np.cross`` on
    # two-component vectors.  This is the scalar 2-D cross product.
    return float(first[0] * second[1] - first[1] * second[0])


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    segment_length = max(1.0, float(np.linalg.norm(end - start)))
    if abs(_orientation(start, end, point)) / segment_length > 1.0e-5:
        return False
    return bool(
        min(start[0], end[0]) - 1.0e-5 <= point[0]
        <= max(start[0], end[0]) + 1.0e-5
        and min(start[1], end[1]) - 1.0e-5 <= point[1]
        <= max(start[1], end[1]) + 1.0e-5
    )


def point_strictly_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    for index in range(len(polygon)):
        if _point_on_segment(point, polygon[index], polygon[(index + 1) % len(polygon)]):
            return False
    inside = False
    x_value, y_value = float(point[0]), float(point[1])
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if ((y1 > y_value) != (y2 > y_value)):
            crossing_x = (x2 - x1) * (y_value - y1) / (y2 - y1) + x1
            if crossing_x > x_value:
                inside = not inside
        previous = current
    return inside


def segments_cross_strictly(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    values = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if any(abs(value) <= 1.0e-4 for value in values):
        return False
    return bool(values[0] * values[1] < -1.0e-8 and values[2] * values[3] < -1.0e-8)


def polygons_overlap_with_area(first: np.ndarray, second: np.ndarray) -> bool:
    # Coincident or almost coincident convex pieces can overlap while every
    # vertex lies on/just outside the other boundary, so vertex-only tests
    # miss them.  The area centroid closes that common numerical hole.
    if point_strictly_inside_polygon(polygon_centroid(first), second):
        return True
    if point_strictly_inside_polygon(polygon_centroid(second), first):
        return True
    for point in first:
        if point_strictly_inside_polygon(point, second):
            return True
    for point in second:
        if point_strictly_inside_polygon(point, first):
            return True
    for first_index in range(len(first)):
        first_start, first_end = edge_points(first, first_index)
        for second_index in range(len(second)):
            second_start, second_end = edge_points(second, second_index)
            if segments_cross_strictly(
                first_start, first_end, second_start, second_end
            ):
                return True
    return False


def point_to_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    amount = float(np.dot(point - start, segment) / denominator)
    amount = min(1.0, max(0.0, amount))
    return float(np.linalg.norm(point - (start + amount * segment)))


def polygon_clearance(first: np.ndarray, second: np.ndarray) -> float:
    if polygons_overlap_with_area(first, second):
        return -1.0
    distances: list[float] = []
    for point in first:
        for index in range(len(second)):
            distances.append(
                point_to_segment_distance(
                    point, second[index], second[(index + 1) % len(second)]
                )
            )
    for point in second:
        for index in range(len(first)):
            distances.append(
                point_to_segment_distance(
                    point, first[index], first[(index + 1) % len(first)]
                )
            )
    return min(distances, default=float("inf"))


def _compose_with_global(
    transform: RigidTransform, global_angle: float
) -> RigidTransform:
    return RigidTransform(
        normalize_angle_rad(transform.angle_rad + global_angle),
        rotate_points(transform.translation.reshape(1, 2), global_angle)[0],
    )


def _piece_has_boundary_edge(
    polygon: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    tolerance: float = BOUNDARY_TOLERANCE_MM,
) -> bool:
    for index in range(len(polygon)):
        start, end = edge_points(polygon, index)
        if (
            max(abs(float(start[0] - minimum[0])), abs(float(end[0] - minimum[0])))
            <= tolerance
            or max(abs(float(start[0] - maximum[0])), abs(float(end[0] - maximum[0])))
            <= tolerance
            or max(abs(float(start[1] - minimum[1])), abs(float(end[1] - minimum[1])))
            <= tolerance
            or max(abs(float(start[1] - maximum[1])), abs(float(end[1] - maximum[1])))
            <= tolerance
        ):
            return True
    return False


def _dimensions_allowed(width: float, height: float) -> bool:
    long_side = max(float(width), float(height))
    short_side = min(float(width), float(height))
    return bool(
        MIN_TARGET_WIDTH_MM - DIMENSION_CALIBRATION_TOLERANCE_MM
        <= long_side
        <= MAX_TARGET_WIDTH_MM + DIMENSION_CALIBRATION_TOLERANCE_MM
        and MIN_TARGET_HEIGHT_MM - DIMENSION_CALIBRATION_TOLERANCE_MM
        <= short_side
        <= MAX_TARGET_HEIGHT_MM + DIMENSION_CALIBRATION_TOLERANCE_MM
    )


def _candidate_rectangle_sizes(polygons: list[np.ndarray]) -> list[tuple[float, float]]:
    """Infer likely rectangle dimensions from sums of observed outer edges."""
    total_area = sum(polygon_area(polygon) for polygon in polygons)
    measured_edges = [
        edge_length(polygon, edge_index)
        for polygon in polygons
        for edge_index in range(len(polygon))
    ]
    values: set[float] = set()

    def collect(piece_index: int, running: float) -> None:
        if piece_index >= len(polygons):
            if 45.0 <= running <= 125.0:
                values.add(round(running * 2.0) / 2.0)
            return
        collect(piece_index + 1, running)
        polygon = polygons[piece_index]
        for edge_index in range(len(polygon)):
            collect(
                piece_index + 1,
                running + edge_length(polygon, edge_index),
            )

    collect(0, 0.0)
    for value in np.arange(50.0, 120.01, 2.5):
        values.add(round(float(value), 1))
    long_values = sorted(value for value in values if 85.0 <= value <= 125.0)
    short_values = sorted(value for value in values if 45.0 <= value <= 95.0)
    sizes: list[tuple[float, float, float]] = []
    for width in long_values:
        for height in short_values:
            coverage = total_area / max(1.0, width * height)
            if not 0.58 <= coverage <= 1.06:
                continue
            sizes.append((abs(1.0 - coverage), width, height))
    sizes.sort()

    # The longest measured edge is normally one side (or nearly one side) of
    # the target rectangle.  Threshold erosion and line fitting shorten it by
    # a few millimetres, while the total white area is also smaller than the
    # target because the pieces are intentionally separated.  Derive a small
    # high-probability prefix from those two measurements.  This puts the
    # field set's 110 x 65 mm solution first instead of making the Raspberry
    # Pi reject roughly forty less plausible rectangles before reaching it.
    # The exhaustive candidates below are retained unchanged as a fallback.
    priority: list[tuple[float, float]] = []
    # Question 3 guarantees a nominal 90 mm side and a 50..120 mm other
    # side.  Put a few scale-aware candidates first; the variable side is
    # inferred from observed area, so this remains valid for every allowed
    # rectangle rather than hard-coding the current 100 x 90 mm field set.
    fixed_edge_measurements = sorted(
        (length for length in measured_edges if 82.0 <= length <= 98.0),
        key=lambda length: abs(length - 90.0),
    )
    measured_fixed_side = (
        round(fixed_edge_measurements[0] / 2.5) * 2.5
        if fixed_edge_measurements
        else 90.0
    )
    fixed_side_candidates: list[float] = []
    for value in (
        measured_fixed_side,
        measured_fixed_side + 2.5,
        90.0,
        measured_fixed_side - 2.5,
        95.0,
        87.5,
    ):
        value = float(min(95.0, max(85.0, value)))
        if value not in fixed_side_candidates:
            fixed_side_candidates.append(value)
    # Interleave fill targets across fixed-side hypotheses.  Otherwise one
    # slightly wrong fixed-side estimate can reach a loose 89% fill box before
    # the next (and correct) scale hypothesis gets its tight 95% candidate.
    for expected_fill in (0.98, 0.95, 0.92, 0.89):
        for fixed_side in fixed_side_candidates:
            other_side = round(
                total_area / max(1.0, fixed_side * expected_fill) / 2.5
            ) * 2.5
            width = max(float(fixed_side), float(other_side))
            height = min(float(fixed_side), float(other_side))
            candidate = (width, height)
            coverage = total_area / max(1.0, width * height)
            if (
                _dimensions_allowed(width, height)
                and 0.86 <= coverage <= 1.06
                and candidate not in priority
            ):
                priority.append(candidate)
    if measured_edges:
        longest = math.ceil(max(measured_edges) / 2.5) * 2.5
        short_edges = sorted(
            (length for length in measured_edges if 45.0 <= length <= 95.0)
        )
        for width_offset in (5.0, 2.5, 0.0, 7.5, 10.0):
            width = longest + width_offset
            if not 88.0 <= width <= 122.0:
                continue
            estimated_height = round(
                (total_area / max(1.0, 0.90 * width)) / 2.5
            ) * 2.5
            height_candidates = [estimated_height]
            if short_edges:
                nearest = min(
                    short_edges,
                    key=lambda length: abs(length - estimated_height),
                )
                rounded_nearest = math.ceil(nearest / 2.5) * 2.5
                height_candidates.extend((rounded_nearest, rounded_nearest + 2.5))
            for height in height_candidates:
                candidate = (float(width), float(height))
                coverage = total_area / max(1.0, width * height)
                if (
                    45.0 <= height <= 95.0
                    and 0.58 <= coverage <= 1.06
                    and candidate not in priority
                ):
                    priority.append(candidate)

    selected = list(priority)
    selected.extend(
        (width, height)
        for _error, width, height in sizes[:20]
        if (width, height) not in selected
    )
    # With real threshold contours, an exact-area rectangle can be a few
    # millimetres too tight even though its area estimate is excellent.  The
    # old fallback jumped directly to coarse 70/80 mm heights; on the July
    # field set that skipped the feasible 100 x 65 mm layout and accepted a
    # sparse 110 x 70 mm box.  Search a small, ordered band of near-full
    # standard sizes before those loose fallbacks.
    expanded_sizes: list[tuple[float, float, float]] = []
    for width in np.arange(90.0, 120.01, 2.5):
        for height in np.arange(50.0, 95.01, 2.5):
            coverage = total_area / max(1.0, float(width * height))
            candidate = (float(width), float(height))
            if 0.88 <= coverage <= 0.95 and candidate not in selected:
                expanded_sizes.append(
                    (abs(coverage - 0.93), float(width), float(height))
                )
    expanded_sizes.sort()
    selected.extend(
        (width, height) for _error, width, height in expanded_sizes[:16]
    )
    loose_sizes = []
    for width, height in (
        (90.0, 70.0), (90.0, 80.0),
        (100.0, 70.0), (100.0, 80.0),
        (110.0, 70.0), (110.0, 80.0),
        (120.0, 70.0), (120.0, 80.0),
    ):
        coverage = total_area / (width * height)
        if 0.58 <= coverage <= 1.06 and (width, height) not in selected:
            loose_sizes.append((abs(coverage - 0.75), width, height))
    loose_sizes.sort()
    selected.extend((width, height) for _error, width, height in loose_sizes)
    return selected


def _boundary_positions(side_length: float, edge_length_mm: float) -> list[float]:
    maximum = side_length - edge_length_mm
    if maximum < -2.0:
        return []
    maximum = max(0.0, maximum)
    values = list(
        np.arange(0.0, maximum + 0.01, BOUNDARY_POSITION_STEP_MM)
    )
    values.extend((0.0, maximum))
    return sorted({round(float(value), 2) for value in values})


def _boundary_states(
    polygon: np.ndarray,
    width: float,
    height: float,
) -> list[BoundaryPlacement]:
    states: list[BoundaryPlacement] = []
    seen: set[tuple[float, ...]] = set()
    # sides: top, right, bottom, left.  Both edge directions are tried and
    # only the orientation whose polygon interior lies inside is retained.
    side_angles = (
        (0, (0.0, math.pi), width),
        (1, (math.pi * 0.5, -math.pi * 0.5), height),
        (2, (0.0, math.pi), width),
        (3, (math.pi * 0.5, -math.pi * 0.5), height),
    )
    for edge_index in range(len(polygon)):
        start, end = edge_points(polygon, edge_index)
        source_angle = math.atan2(float((end - start)[1]), float((end - start)[0]))
        length = float(np.linalg.norm(end - start))
        for side, target_angles, side_length in side_angles:
            for target_angle in target_angles:
                angle = normalize_angle_rad(target_angle - source_angle)
                rotated = rotate_points(polygon, angle)
                rotated_start, rotated_end = edge_points(rotated, edge_index)
                if side in (0, 2):
                    boundary = 0.0 if side == 0 else height
                    normal_shift = boundary - 0.5 * (
                        float(rotated_start[1]) + float(rotated_end[1])
                    )
                    normal_translation = np.asarray([0.0, normal_shift])
                    normal_polygon = rotated + normal_translation
                    edge_minimum = min(
                        float(rotated_start[0]), float(rotated_end[0])
                    )
                    for position in _boundary_positions(side_length, length):
                        translation = normal_translation + np.asarray(
                            [position - edge_minimum, 0.0]
                        )
                        placed = rotated + translation
                        if (
                            float(np.min(placed[:, 0])) < -3.0
                            or float(np.max(placed[:, 0])) > width + 3.0
                            or float(np.min(placed[:, 1])) < -3.0
                            or float(np.max(placed[:, 1])) > height + 3.0
                        ):
                            continue
                        touches_corner = bool(
                            position <= 1.0
                            or side_length - position - length <= 1.0
                        )
                        key = (
                            round(angle, 3),
                            round(float(translation[0]), 1),
                            round(float(translation[1]), 1),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        states.append(
                            BoundaryPlacement(
                                RigidTransform(angle, translation),
                                placed,
                                edge_index,
                                side,
                                touches_corner,
                            )
                        )
                else:
                    boundary = width if side == 1 else 0.0
                    normal_shift = boundary - 0.5 * (
                        float(rotated_start[0]) + float(rotated_end[0])
                    )
                    normal_translation = np.asarray([normal_shift, 0.0])
                    edge_minimum = min(
                        float(rotated_start[1]), float(rotated_end[1])
                    )
                    for position in _boundary_positions(side_length, length):
                        translation = normal_translation + np.asarray(
                            [0.0, position - edge_minimum]
                        )
                        placed = rotated + translation
                        if (
                            float(np.min(placed[:, 0])) < -3.0
                            or float(np.max(placed[:, 0])) > width + 3.0
                            or float(np.min(placed[:, 1])) < -3.0
                            or float(np.max(placed[:, 1])) > height + 3.0
                        ):
                            continue
                        touches_corner = bool(
                            position <= 1.0
                            or side_length - position - length <= 1.0
                        )
                        key = (
                            round(angle, 3),
                            round(float(translation[0]), 1),
                            round(float(translation[1]), 1),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        states.append(
                            BoundaryPlacement(
                                RigidTransform(angle, translation),
                                placed,
                                edge_index,
                                side,
                                touches_corner,
                            )
                        )
    return states


def _contact_graph_maximum_gap(polygons: list[np.ndarray]) -> float | None:
    """Return the largest edge in a minimum spanning contact graph."""
    if len(polygons) <= 1:
        return 0.0
    edges = sorted(
        (
            polygon_clearance(polygons[first], polygons[second]),
            first,
            second,
        )
        for first in range(len(polygons))
        for second in range(first + 1, len(polygons))
    )
    parents = list(range(len(polygons)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    selected: list[float] = []
    for distance, first, second in edges:
        first_root, second_root = root(first), root(second)
        if first_root == second_root:
            continue
        parents[first_root] = second_root
        selected.append(float(distance))
        if len(selected) == len(polygons) - 1:
            break
    if len(selected) != len(polygons) - 1:
        return None
    return max(selected, default=0.0)


def _rasterize_polygon(
    polygon: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Rasterize at 1 mm/pixel using a vectorized even/odd fill rule."""
    raster_width = max(1, int(math.ceil(width)))
    raster_height = max(1, int(math.ceil(height)))
    y_grid, x_grid = np.mgrid[0:raster_height, 0:raster_width]
    x_grid = x_grid.astype(np.float64) + 0.5
    y_grid = y_grid.astype(np.float64) + 0.5
    inside = np.zeros((raster_height, raster_width), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        crossing = (y1 > y_grid) != (y2 > y_grid)
        denominator = y2 - y1
        if abs(denominator) < 1.0e-9:
            previous = current
            continue
        crossing_x = (x2 - x1) * (y_grid - y1) / denominator + x1
        inside ^= crossing & (x_grid < crossing_x)
        previous = current
    return inside


def _dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    return dilated


def _count_connected_components(mask: np.ndarray) -> int:
    """Count 4-connected foreground regions without requiring OpenCV."""
    foreground = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(foreground)
    components = 0
    height, width = foreground.shape
    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            components += 1
            visited[start_y, start_x] = True
            stack = [(start_y, start_x)]
            while stack:
                y_value, x_value = stack.pop()
                for next_y, next_x in (
                    (y_value - 1, x_value),
                    (y_value + 1, x_value),
                    (y_value, x_value - 1),
                    (y_value, x_value + 1),
                ):
                    if not (0 <= next_y < height and 0 <= next_x < width):
                        continue
                    if (
                        foreground[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
    return components


def _global_rectangle_quality(
    polygons: list[np.ndarray],
    width: float,
    height: float,
    total_area: float,
) -> dict[str, float | int]:
    """Score an inferred rectangle without assuming one fixed target size.

    This borrows the useful global-union idea from puzzle-vision-simulator but
    evaluates the candidate's inferred dimensions, rather than a hard-coded
    100 x 60 mm rectangle.  Local edge matches cannot hide holes, overlap, a
    disconnected union, or an implausible outer perimeter.
    """
    raster_width = max(1, int(math.ceil(float(width))))
    raster_height = max(1, int(math.ceil(float(height))))
    masks = [
        _rasterize_polygon(polygon, width, height)
        for polygon in polygons
    ]
    occupancy = np.zeros((raster_height, raster_width), np.uint8)
    for mask in masks:
        occupancy += mask.astype(np.uint8)
    union = occupancy > 0
    union_pixels = int(np.count_nonzero(union))
    overlap_pixels = int(np.sum(np.maximum(occupancy.astype(np.int16) - 1, 0)))
    raster_area = max(1, raster_width * raster_height)
    fill_ratio = union_pixels / float(raster_area)
    geometric_coverage = total_area / max(1.0, float(width * height))
    component_count = _count_connected_components(union)

    padded = np.pad(union, 1, mode="constant", constant_values=False)
    horizontal = int(np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
    vertical = int(np.count_nonzero(padded[1:, :] != padded[:-1, :]))
    perimeter_pixels = horizontal + vertical
    expected_perimeter = 2.0 * (raster_width + raster_height)
    perimeter_error = abs(perimeter_pixels - expected_perimeter)

    score = (
        90.0 * abs(1.0 - min(1.05, geometric_coverage))
        + 55.0 * max(0.0, 1.0 - fill_ratio)
        + 1.5 * overlap_pixels
        + 14.0 * max(0, component_count - 1)
        + 0.08 * perimeter_error
    )
    return {
        "score": float(score),
        "geometric_coverage": float(geometric_coverage),
        "raster_fill_ratio": float(fill_ratio),
        "overlap_pixels": overlap_pixels,
        "connected_components": component_count,
        "perimeter_error_mm": float(perimeter_error),
    }


def _augment_full_seam_constraints(
    polygons: list[np.ndarray],
    transforms: list[RigidTransform],
    seams: list[SeamMatch],
) -> list[SeamMatch]:
    """Add unrecorded closed-loop contacts to a DFS spanning tree."""
    constraints = [
        seam
        for seam in seams
        if abs(
            edge_length(polygons[seam.first_piece], seam.first_edge)
            - edge_length(polygons[seam.second_piece], seam.second_edge)
        ) <= EDGE_MATCH_TOLERANCE_MM
    ]
    seen = {
        (
            min(seam.first_piece, seam.second_piece),
            seam.first_edge if seam.first_piece < seam.second_piece else seam.second_edge,
            max(seam.first_piece, seam.second_piece),
            seam.second_edge if seam.first_piece < seam.second_piece else seam.first_edge,
        )
        for seam in constraints
    }
    assembled = [
        transforms[index].apply(polygons[index])
        for index in range(len(polygons))
    ]
    for first_piece in range(len(polygons)):
        for second_piece in range(first_piece + 1, len(polygons)):
            for first_edge in range(len(polygons[first_piece])):
                first_length = edge_length(polygons[first_piece], first_edge)
                first_start, first_end = edge_points(
                    assembled[first_piece], first_edge
                )
                for second_edge in range(len(polygons[second_piece])):
                    second_length = edge_length(polygons[second_piece], second_edge)
                    length_error = abs(first_length - second_length)
                    if length_error > EDGE_MATCH_TOLERANCE_MM:
                        continue
                    key = (
                        first_piece,
                        first_edge,
                        second_piece,
                        second_edge,
                    )
                    if key in seen:
                        continue
                    second_start, second_end = edge_points(
                        assembled[second_piece], second_edge
                    )
                    endpoint_rms = math.sqrt(
                        0.5
                        * (
                            float(np.dot(first_start - second_end, first_start - second_end))
                            + float(np.dot(first_end - second_start, first_end - second_start))
                        )
                    )
                    if endpoint_rms > 6.0:
                        continue
                    constraints.append(
                        SeamMatch(
                            first_piece,
                            first_edge,
                            second_piece,
                            second_edge,
                            length_error,
                        )
                    )
                    seen.add(key)
    return constraints


def _optimize_pose_graph(
    polygons: list[np.ndarray],
    seams: list[SeamMatch],
    initial: list[RigidTransform],
) -> tuple[list[RigidTransform], dict[str, float | int]]:
    """Distribute full-edge loop error over all movable pieces."""
    constraints = _augment_full_seam_constraints(polygons, initial, seams)
    if len(polygons) < 2 or not constraints:
        return initial, {
            "constraint_count": len(constraints),
            "residual_before_mm": 0.0,
            "residual_after_mm": 0.0,
            "iterations": 0,
        }

    def pack(transforms: list[RigidTransform]) -> np.ndarray:
        values: list[float] = []
        for transform in transforms[1:]:
            values.extend(
                [
                    float(transform.angle_rad),
                    float(transform.translation[0]),
                    float(transform.translation[1]),
                ]
            )
        return np.asarray(values, np.float64)

    def unpack(values: np.ndarray) -> list[RigidTransform]:
        transforms = [initial[0]]
        for index in range(len(polygons) - 1):
            angle, x_value, y_value = values[3 * index : 3 * index + 3]
            transforms.append(
                RigidTransform(
                    normalize_angle_rad(float(angle)),
                    np.asarray([x_value, y_value], np.float64),
                )
            )
        return transforms

    def residual(values: np.ndarray) -> np.ndarray:
        transforms = unpack(values)
        result: list[float] = []
        for seam in constraints:
            first_start, first_end = edge_points(
                polygons[seam.first_piece], seam.first_edge
            )
            second_start, second_end = edge_points(
                polygons[seam.second_piece], seam.second_edge
            )
            first_world = transforms[seam.first_piece].apply(
                np.asarray([first_start, first_end])
            )
            second_world = transforms[seam.second_piece].apply(
                np.asarray([second_end, second_start])
            )
            result.extend((first_world - second_world).reshape(-1).tolist())
        return np.asarray(result, np.float64)

    values = pack(initial)
    before_vector = residual(values)
    iterations = 0
    for iteration in range(20):
        current = residual(values)
        jacobian = np.empty((len(current), len(values)), np.float64)
        for index in range(len(values)):
            step = 1.0e-5 if index % 3 == 0 else 1.0e-3
            shifted = values.copy()
            shifted[index] += step
            jacobian[:, index] = (residual(shifted) - current) / step
        delta, *_unused = np.linalg.lstsq(jacobian, -current, rcond=None)
        # A bad accidental loop contact must not launch a large pose jump.
        for piece_index in range(len(polygons) - 1):
            delta[3 * piece_index] = np.clip(
                delta[3 * piece_index], -math.radians(3.0), math.radians(3.0)
            )
            delta[3 * piece_index + 1 : 3 * piece_index + 3] = np.clip(
                delta[3 * piece_index + 1 : 3 * piece_index + 3], -3.0, 3.0
            )
        values += delta
        iterations = iteration + 1
        if float(np.linalg.norm(delta)) < 1.0e-7:
            break
    after_vector = residual(values)
    before_rms = math.sqrt(float(np.mean(before_vector * before_vector)))
    after_rms = math.sqrt(float(np.mean(after_vector * after_vector)))
    if after_rms > before_rms + 1.0e-6:
        return initial, {
            "constraint_count": len(constraints),
            "residual_before_mm": before_rms,
            "residual_after_mm": before_rms,
            "iterations": 0,
        }
    return unpack(values), {
        "constraint_count": len(constraints),
        "residual_before_mm": before_rms,
        "residual_after_mm": after_rms,
        "iterations": iterations,
    }


def _refine_winning_candidate(
    candidate: dict[str, Any],
    polygons: list[np.ndarray],
    total_area: float,
) -> None:
    """Refine only the winning topology, then rebuild its target transforms."""
    candidate["pre_optimization_polygons"] = [
        np.asarray(polygon, np.float64).copy()
        for polygon in candidate["polygons"]
    ]
    seams = list(candidate.get("seams") or [])
    if not seams:
        candidate["pose_graph"] = {
            "constraint_count": 0,
            "residual_before_mm": 0.0,
            "residual_after_mm": 0.0,
            "iterations": 0,
        }
        return

    initial_transforms = list(candidate["transforms"])
    refined_transforms, diagnostics = _optimize_pose_graph(
        polygons, seams, initial_transforms
    )
    raw_refined = [
        refined_transforms[index].apply(polygons[index])
        for index in range(len(polygons))
    ]
    raw_points = np.concatenate(raw_refined, axis=0)
    refined_minimum = np.min(raw_points, axis=0)
    normalized_refined = [
        polygon - refined_minimum for polygon in raw_refined
    ]
    normalized_size = np.max(
        np.concatenate(normalized_refined, axis=0), axis=0
    )
    if not _dimensions_allowed(
        float(normalized_size[0]), float(normalized_size[1])
    ):
        diagnostics["accepted"] = False
        diagnostics["rejection_reason"] = "REFINED_DIMENSIONS_OUT_OF_RANGE"
        candidate["pose_graph"] = diagnostics
        return

    refined_quality = _global_rectangle_quality(
        normalized_refined,
        float(normalized_size[0]),
        float(normalized_size[1]),
        total_area,
    )
    previous_quality = candidate.get("global_quality") or {}
    previous_quality_score = float(previous_quality.get("score", math.inf))
    if float(refined_quality["score"]) > previous_quality_score + 0.75:
        diagnostics["accepted"] = False
        diagnostics["rejection_reason"] = "REFINED_GLOBAL_QUALITY_WORSE"
        candidate["pose_graph"] = diagnostics
        return

    gap_offsets = [
        np.asarray(offset, np.float64)
        for offset in candidate.get(
            "gap_offsets",
            [np.zeros(2, np.float64) for _ in polygons],
        )
    ]
    separated = [
        polygon + gap_offsets[index]
        for index, polygon in enumerate(normalized_refined)
    ]
    separated_points = np.concatenate(separated, axis=0)
    final_minimum = np.min(separated_points, axis=0)
    separated = [polygon - final_minimum for polygon in separated]
    final_size = np.max(np.concatenate(separated, axis=0), axis=0)

    adjusted_transforms: list[RigidTransform] = []
    for index, transform in enumerate(refined_transforms):
        adjustment = -refined_minimum + gap_offsets[index] - final_minimum
        adjusted_transforms.append(
            RigidTransform(
                transform.angle_rad,
                transform.translation + adjustment,
            )
        )
    diagnostics["accepted"] = True
    candidate.update(
        {
            "transforms": adjusted_transforms,
            "polygons": separated,
            "final_size": final_size,
            "exact_minimum_after_global": np.zeros(2, np.float64),
            "gap_offsets": [
                np.zeros(2, np.float64) for _ in polygons
            ],
            "global_quality": refined_quality,
            "pose_graph": diagnostics,
        }
    )


def _boundary_rectangle_pack(polygons: list[np.ndarray]) -> dict[str, Any] | None:
    """Pack pieces using the problem's guaranteed outer-boundary edge."""
    total_area = sum(polygon_area(polygon) for polygon in polygons)
    best: dict[str, Any] | None = None
    for width, height in _candidate_rectangle_sizes(polygons):
        states = [
            _boundary_states(polygon, width, height)
            for polygon in polygons
        ]
        if any(not values for values in states):
            continue
        raster_states = [
            [
                (state, _rasterize_polygon(state.polygon, width, height))
                for state in piece_states
            ]
            for piece_states in states
        ]
        areas = [polygon_area(polygon) for polygon in polygons]
        for anchor in range(len(polygons)):
            anchor_states = [
                (state, mask)
                for state, mask in raster_states[anchor]
                if state.touches_corner
            ]
            if not anchor_states:
                continue
            order = [anchor] + sorted(
                (index for index in range(len(polygons)) if index != anchor),
                key=lambda index: -areas[index],
            )
            beam: list[
                tuple[float, list[BoundaryPlacement | None], np.ndarray]
            ] = []
            for state, state_mask in anchor_states:
                selected: list[BoundaryPlacement | None] = [None] * len(polygons)
                selected[anchor] = state
                beam.append((0.0, selected, state_mask.copy()))
            beam = beam[:BOUNDARY_BEAM_WIDTH]
            for piece_index in order[1:]:
                next_beam: list[
                    tuple[
                        float,
                        list[BoundaryPlacement | None],
                        np.ndarray,
                    ]
                ] = []
                for score, selected, occupied in beam:
                    near_occupied = _dilate_one_pixel(
                        _dilate_one_pixel(occupied)
                    )
                    for state, state_mask in raster_states[piece_index]:
                        overlap_pixels = int(np.count_nonzero(occupied & state_mask))
                        # Contour erosion/dilation can create a narrow apparent
                        # overlap even when the physical cut edges coincide.
                        # Keep such states for exact geometric validation and
                        # later seam separation, but reject substantial area
                        # overlap at the 1 mm raster scale.
                        if overlap_pixels > 40:
                            continue
                        touches_existing = bool(
                            np.any(near_occupied & state_mask)
                        )
                        updated = list(selected)
                        updated[piece_index] = state
                        next_beam.append(
                            (
                                score
                                + (0.0 if touches_existing else 8.0)
                                + 0.2 * overlap_pixels,
                                updated,
                                occupied | state_mask,
                            )
                        )
                if not next_beam:
                    beam = []
                    break
                next_beam.sort(key=lambda item: item[0])
                unique: dict[
                    tuple[float, ...],
                    tuple[
                        float,
                        list[BoundaryPlacement | None],
                        np.ndarray,
                    ],
                ] = {}
                for candidate in next_beam:
                    key = tuple(
                        value
                        for state in candidate[1]
                        if state is not None
                        for value in np.round(polygon_centroid(state.polygon), 1)
                    )
                    if key not in unique:
                        unique[key] = candidate
                    if len(unique) >= BOUNDARY_BEAM_WIDTH:
                        break
                beam = list(unique.values())
            for contact_score, selected_optional, _occupied in beam:
                if any(state is None for state in selected_optional):
                    continue
                selected = [
                    state
                    for state in selected_optional
                    if state is not None
                ]
                raw_target_polygons = [
                    selected_optional[index].polygon
                    for index in range(len(polygons))
                ]
                raw_points = np.concatenate(raw_target_polygons, axis=0)
                actual_minimum = np.min(raw_points, axis=0)
                target_polygons = [
                    polygon - actual_minimum for polygon in raw_target_polygons
                ]
                actual_size = (
                    np.max(np.concatenate(target_polygons, axis=0), axis=0)
                )
                if not _dimensions_allowed(
                    float(actual_size[0]), float(actual_size[1])
                ):
                    continue
                if any(
                    polygons_overlap_with_area(
                        target_polygons[first], target_polygons[second]
                    )
                    for first in range(len(polygons))
                    for second in range(first + 1, len(polygons))
                ):
                    continue
                maximum_gap = _contact_graph_maximum_gap(target_polygons)
                if maximum_gap is None or maximum_gap > 20.0:
                    continue
                coverage = total_area / max(
                    1.0, float(actual_size[0] * actual_size[1])
                )
                # Pieces are cut from one rectangle, so their union must fill
                # almost all of it.  A 0.78-fill result can satisfy the outer
                # boundary/contact tests while still looking nothing like the
                # reconstructed rectangle.  Reject that false-positive class.
                if coverage < 0.86 or coverage > 1.06:
                    continue
                score = (
                    30.0 * abs(1.0 - coverage)
                    + 0.12 * maximum_gap
                    + 0.02 * contact_score
                )
                if best is None or score < float(best["score"]):
                    minimum_clearance = min(
                        (
                            polygon_clearance(
                                target_polygons[first], target_polygons[second]
                            )
                            for first in range(len(polygons))
                            for second in range(first + 1, len(polygons))
                        ),
                        default=float("inf"),
                    )
                    best = {
                        "score": score,
                        "adjacency_signature": tuple(
                            (first, second)
                            for first in range(len(polygons))
                            for second in range(first + 1, len(polygons))
                            if polygon_clearance(
                                target_polygons[first], target_polygons[second]
                            ) <= 20.0
                        ),
                        "polygons": target_polygons,
                        "transforms": [
                            RigidTransform(
                                selected_optional[index].transform.angle_rad,
                                selected_optional[index].transform.translation
                                - actual_minimum,
                            )
                            for index in range(len(polygons))
                        ],
                        "offsets": [
                            np.zeros(2, np.float64) for _ in polygons
                        ],
                        "pre_gap_minimum": np.zeros(2, np.float64),
                        "final_size": actual_size,
                        "coverage": coverage,
                        "seam_error": maximum_gap,
                        "selected_gap": minimum_clearance,
                        "minimum_clearance": minimum_clearance,
                        "maximum_vertex_gap": maximum_gap,
                        "global_angle": 0.0,
                        "exact_minimum_after_global": np.zeros(2, np.float64),
                        "gap_offsets": [
                            np.zeros(2, np.float64) for _ in polygons
                        ],
                        "solver": "outer_boundary_rectangle_pack_v1",
                    }
        if best is not None:
            # Candidate sizes are ordered from near-exact area fill to the
            # looser 20 mm-gap layouts.  The first feasible size is therefore
            # preferred and avoids an exhaustive search on the Raspberry Pi.
            return best
    return best


def _layout_key(polygons: list[np.ndarray]) -> tuple[float, ...]:
    centers = sorted(
        (polygon_centroid(polygon) for polygon in polygons),
        key=lambda center: (float(center[0]), float(center[1])),
    )
    points = np.concatenate(polygons, axis=0)
    minimum = np.min(points, axis=0)
    return tuple(
        round(float(value), 1)
        for center in centers
        for value in center - minimum
    )


def _separate_seams(
    polygons: list[np.ndarray],
    seams: list[SeamMatch],
    desired_gap: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    offsets = [np.zeros(2, np.float64) for _ in polygons]
    centers = [polygon_centroid(polygon) for polygon in polygons]
    for _iteration in range(32):
        changed = False
        for seam in seams:
            first_polygon = polygons[seam.first_piece]
            first_start, first_end = edge_points(first_polygon, seam.first_edge)
            direction = first_end - first_start
            length = float(np.linalg.norm(direction))
            if length <= 1.0e-9:
                continue
            normal = np.asarray([-direction[1], direction[0]], np.float64) / length
            if float(np.dot(centers[seam.second_piece] - centers[seam.first_piece], normal)) < 0.0:
                normal *= -1.0
            current = float(
                np.dot(
                    offsets[seam.second_piece] - offsets[seam.first_piece],
                    normal,
                )
            )
            if current + 1.0e-6 < desired_gap:
                correction = 0.5 * (desired_gap - current) * normal
                offsets[seam.first_piece] -= correction
                offsets[seam.second_piece] += correction
                changed = True
        if not changed:
            break
    average = np.mean(np.asarray(offsets), axis=0)
    offsets = [offset - average for offset in offsets]
    return [polygon + offset for polygon, offset in zip(polygons, offsets)], offsets


def _convex_overlap_mtv(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Return a separating-axis direction and penetration for convex pieces."""
    smallest_overlap = float("inf")
    smallest_axis: np.ndarray | None = None
    for polygon in (first, second):
        for edge_index in range(len(polygon)):
            start, end = edge_points(polygon, edge_index)
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length <= 1.0e-9:
                continue
            axis = np.asarray([-edge[1], edge[0]], np.float64) / length
            first_projection = np.asarray(first) @ axis
            second_projection = np.asarray(second) @ axis
            overlap = min(
                float(np.max(first_projection)),
                float(np.max(second_projection)),
            ) - max(
                float(np.min(first_projection)),
                float(np.min(second_projection)),
            )
            if overlap <= 1.0e-6:
                return None
            if overlap < smallest_overlap:
                smallest_overlap = overlap
                smallest_axis = axis
    if smallest_axis is None:
        return None
    center_delta = polygon_centroid(second) - polygon_centroid(first)
    if float(np.dot(center_delta, smallest_axis)) < 0.0:
        smallest_axis *= -1.0
    return smallest_axis, smallest_overlap


def _resolve_composite_overlaps(
    polygons: list[np.ndarray],
    clearance_mm: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Remove small contour-noise overlaps without rotating or mirroring."""
    adjusted = [np.asarray(polygon, np.float64).copy() for polygon in polygons]
    offsets = [np.zeros(2, np.float64) for _ in adjusted]
    for _iteration in range(32):
        changed = False
        for first_index in range(len(adjusted)):
            for second_index in range(first_index + 1, len(adjusted)):
                if not polygons_overlap_with_area(
                    adjusted[first_index], adjusted[second_index]
                ):
                    continue
                separation = _convex_overlap_mtv(
                    adjusted[first_index], adjusted[second_index]
                )
                if separation is None:
                    continue
                axis, penetration = separation
                correction = 0.5 * (
                    penetration + max(0.0, float(clearance_mm))
                ) * axis
                adjusted[first_index] -= correction
                adjusted[second_index] += correction
                offsets[first_index] -= correction
                offsets[second_index] += correction
                changed = True
        if not changed:
            break
    average = np.mean(np.asarray(offsets), axis=0)
    offsets = [offset - average for offset in offsets]
    adjusted = [
        polygon - average for polygon in adjusted
    ]
    return adjusted, offsets


def _resolve_composite_overlaps_grid(
    polygons: list[np.ndarray],
    maximum_shift_mm: float = 15.0,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Find small independent translations that make all pieces disjoint.

    Pairwise minimum-translation corrections can oscillate when four pieces
    meet at one T-junction.  A tiny bounded beam search is deterministic and
    directly enforces the mechanical 15 mm adjacent-vertex allowance.
    """
    count = len(polygons)
    if count <= 1:
        return ([np.asarray(polygons[0]).copy()], [np.zeros(2)])
    shift_candidates = [np.zeros(2, np.float64)]
    seen_shifts = {(0.0, 0.0)}
    for radius in np.arange(1.0, maximum_shift_mm + 0.01, 1.0):
        for angle_index in range(8):
            angle = 2.0 * math.pi * angle_index / 8.0
            shift = np.asarray(
                [radius * math.cos(angle), radius * math.sin(angle)],
                np.float64,
            )
            key = (round(float(shift[0]), 1), round(float(shift[1]), 1))
            if key in seen_shifts:
                continue
            seen_shifts.add(key)
            shift_candidates.append(shift)
    shift_candidates.sort(key=lambda shift: float(np.dot(shift, shift)))

    best: tuple[float, list[np.ndarray]] | None = None
    areas = [polygon_area(polygon) for polygon in polygons]
    for anchor in range(count):
        order = [anchor] + sorted(
            (index for index in range(count) if index != anchor),
            key=lambda index: -areas[index],
        )
        initial_shifts: list[np.ndarray | None] = [None] * count
        initial_shifts[anchor] = np.zeros(2, np.float64)
        beam: list[tuple[float, list[np.ndarray | None]]] = [
            (0.0, initial_shifts)
        ]
        for piece_index in order[1:]:
            next_beam: list[tuple[float, list[np.ndarray | None]]] = []
            for score, shifts in beam:
                for shift in shift_candidates:
                    candidate_polygon = polygons[piece_index] + shift
                    if any(
                        polygons_overlap_with_area(
                            candidate_polygon,
                            polygons[other_index] + other_shift,
                        )
                        for other_index, other_shift in enumerate(shifts)
                        if other_shift is not None
                    ):
                        continue
                    updated = list(shifts)
                    updated[piece_index] = shift
                    next_score = score + float(np.linalg.norm(shift))
                    next_beam.append((next_score, updated))
            if not next_beam:
                beam = []
                break
            next_beam.sort(key=lambda item: item[0])
            beam = next_beam[:2]
        for score, shifts_optional in beam:
            if any(shift is None for shift in shifts_optional):
                continue
            shifts = [
                np.asarray(shift, np.float64)
                for shift in shifts_optional
                if shift is not None
            ]
            shifted = [
                polygon + shifts[index]
                for index, polygon in enumerate(polygons)
            ]
            points = np.concatenate(shifted, axis=0)
            size = np.max(points, axis=0) - np.min(points, axis=0)
            if not _dimensions_allowed(float(size[0]), float(size[1])):
                continue
            extent_penalty = 0.01 * float(size[0] * size[1])
            candidate_score = score + extent_penalty
            if best is None or candidate_score < best[0]:
                best = (candidate_score, shifts)
    if best is None:
        return None
    shifts = best[1]
    average = np.mean(np.asarray(shifts), axis=0)
    offsets = [shift - average for shift in shifts]
    adjusted = [
        polygon + offsets[index]
        for index, polygon in enumerate(polygons)
    ]
    return adjusted, offsets


def _build_relative_assemblies(
    polygons: list[np.ndarray],
) -> list[tuple[list[RigidTransform], list[SeamMatch], float]]:
    count = len(polygons)
    if count == 1:
        return [([RigidTransform(0.0, np.zeros(2, np.float64))], [], 0.0)]

    results: list[tuple[list[RigidTransform], list[SeamMatch], float]] = []
    transforms: list[RigidTransform | None] = [None] * count
    transforms[0] = RigidTransform(0.0, np.zeros(2, np.float64))
    seen_states: set[tuple[Any, ...]] = set()
    edge_catalog = [
        (piece_index, edge_length(polygon, edge_index))
        for piece_index, polygon in enumerate(polygons)
        for edge_index in range(len(polygon))
    ]

    def recurse(
        placed: set[int],
        seams: list[SeamMatch],
        mismatch: float,
    ) -> None:
        if len(results) >= MAX_LAYOUT_SOLUTIONS:
            return
        if len(placed) == count:
            complete = [value for value in transforms if value is not None]
            results.append((complete, list(seams), mismatch))
            return

        candidates: list[
            tuple[float, float, int, int, int, int, RigidTransform]
        ] = []
        for moving_index in range(count):
            if moving_index in placed:
                continue
            moving_polygon = polygons[moving_index]
            for fixed_index in sorted(placed):
                fixed_transform = transforms[fixed_index]
                assert fixed_transform is not None
                fixed_polygon = fixed_transform.apply(polygons[fixed_index])
                for fixed_edge in range(len(fixed_polygon)):
                    fixed_length = edge_length(fixed_polygon, fixed_edge)
                    for moving_edge in range(len(moving_polygon)):
                        moving_length = edge_length(moving_polygon, moving_edge)
                        difference = abs(fixed_length - moving_length)
                        shorter = min(fixed_length, moving_length)
                        longer = max(fixed_length, moving_length)
                        full_edge_match = difference <= EDGE_MATCH_TOLERANCE_MM
                        # A valid arbitrary dissection may contain a T-junction:
                        # one long straight seam is shared by two shorter
                        # edges.  Such a partial match still shares an endpoint
                        # and an opposite collinear direction, but its lengths
                        # are intentionally unequal.
                        complement = longer - shorter
                        complement_error = min(
                            (
                                abs(other_length - complement)
                                for other_piece, other_length in edge_catalog
                                if other_piece not in {moving_index, fixed_index}
                            ),
                            default=math.inf,
                        )
                        # Do not admit a weak 72%-length match merely because
                        # the edges point in opposite directions.  On the
                        # field set, 60.7<->102.9 and 42.5<->102.9 consumed
                        # the bounded DFS before the useful topology was
                        # expanded.  A near-full noisy seam must be close in
                        # absolute length; a larger mismatch is admitted only
                        # when a third piece explicitly explains the remainder.
                        near_full_match = bool(
                            difference <= 8.0
                            and shorter / max(1.0e-6, longer) >= 0.85
                        )
                        composite_t_junction = complement_error <= 5.0
                        partial_edge_match = bool(
                            shorter >= 16.0
                            and (near_full_match or composite_t_junction)
                        )
                        if not (full_edge_match or partial_edge_match):
                            continue
                        if full_edge_match:
                            match_cost = difference
                        elif composite_t_junction:
                            # Prefer a long edge whose remaining length is
                            # explained by another piece edge.  This is the
                            # characteristic signature of a T-junction.
                            match_cost = 1.5 + 0.25 * complement_error
                        else:
                            match_cost = (
                                EDGE_MATCH_TOLERANCE_MM + 0.05 * difference
                            )
                        endpoint_options = (
                            (False,)
                            if full_edge_match
                            else (False, True)
                        )
                        for match_other_endpoint in endpoint_options:
                            transform = align_reversed_edge(
                                moving_polygon,
                                moving_edge,
                                fixed_polygon,
                                fixed_edge,
                                match_other_endpoint=match_other_endpoint,
                            )
                            # Do not reject a partial/T-junction placement on
                            # the first measured seam merely because another
                            # observed contour crosses it by a few pixels.
                            # Threshold erosion and paper overhang routinely
                            # create such small apparent overlaps.  The final
                            # 4 mm seam separation and strict no-overlap test
                            # below still reject physically intersecting plans.
                            candidates.append(
                                (
                                    match_cost,
                                    difference,
                                    moving_index,
                                    moving_edge,
                                    fixed_index,
                                    fixed_edge,
                                    transform,
                                )
                            )
        candidates.sort(key=lambda value: value[0])
        for (
            match_cost,
            difference,
            moving_index,
            moving_edge,
            fixed_index,
            fixed_edge,
            transform,
        ) in candidates[:96]:
            transforms[moving_index] = transform
            placed.add(moving_index)
            transformed_centers = []
            transformed_angles = []
            for index in sorted(placed):
                value = transforms[index]
                assert value is not None
                transformed_centers.extend(
                    np.round(value.apply(polygons[index]).mean(axis=0), 1).tolist()
                )
                transformed_angles.append(round(value.angle_rad, 3))
            state_key = (
                tuple(sorted(placed)),
                tuple(transformed_centers),
                tuple(transformed_angles),
            )
            if state_key not in seen_states:
                seen_states.add(state_key)
                seam = SeamMatch(
                    fixed_index,
                    fixed_edge,
                    moving_index,
                    moving_edge,
                    difference,
                )
                recurse(placed, seams + [seam], mismatch + match_cost)
            placed.remove(moving_index)
            transforms[moving_index] = None

    recurse({0}, [], 0.0)
    return results


def _undirected_angle_error(first: np.ndarray, second: np.ndarray) -> float:
    first_angle = math.atan2(float(first[1]), float(first[0]))
    second_angle = math.atan2(float(second[1]), float(second[0]))
    difference = abs(normalize_angle_rad(first_angle - second_angle))
    return min(difference, abs(math.pi - difference))


def _build_composite_t_junction_assemblies(
    polygons: list[np.ndarray],
) -> list[tuple[list[RigidTransform], list[SeamMatch], float]]:
    """Build four-piece layouts where a long seam is split by a T-junction.

    A common competition-piece topology has two centre pieces bridging two
    larger side pieces.  On each side, one observed long edge therefore
    corresponds to the *sum* of one edge from each centre piece.  Treating
    those as independent full-edge matches either fails or produces a mirror
    image.  This enumerator explicitly models the two composite seams while
    still using rotation and translation only.
    """
    if len(polygons) != 4:
        return []
    results: list[tuple[list[RigidTransform], list[SeamMatch], float]] = []
    seen: set[tuple[Any, ...]] = set()
    indices = tuple(range(4))

    for base_index in indices:
        base_polygon = polygons[base_index]
        base_transform = RigidTransform(0.0, np.zeros(2, np.float64))
        for base_edge in range(len(base_polygon)):
            base_start, base_end = edge_points(base_polygon, base_edge)
            base_vector = base_end - base_start
            base_length = float(np.linalg.norm(base_vector))
            if base_length < 35.0:
                continue
            base_direction = base_vector / base_length
            other_indices = [index for index in indices if index != base_index]
            for first_middle, second_middle in permutations(other_indices, 2):
                outer_index = next(
                    index
                    for index in other_indices
                    if index not in {first_middle, second_middle}
                )
                first_polygon = polygons[first_middle]
                second_polygon = polygons[second_middle]
                outer_polygon = polygons[outer_index]
                for first_edge in range(len(first_polygon)):
                    first_length = edge_length(first_polygon, first_edge)
                    for second_edge in range(len(second_polygon)):
                        second_length = edge_length(second_polygon, second_edge)
                        composite_error = abs(
                            base_length - first_length - second_length
                        )
                        if composite_error > COMPOSITE_EDGE_TOLERANCE_MM:
                            continue
                        # Distribute the observed length residual symmetrically.
                        # The pieces remain rigid; this only chooses where the
                        # measured short-edge chain sits on the measured long edge.
                        chain_offset = 0.5 * (
                            base_length - first_length - second_length
                        )
                        first_target_start = (
                            base_start + chain_offset * base_direction
                        )
                        first_target_end = (
                            first_target_start + first_length * base_direction
                        )
                        second_target_start = first_target_end
                        second_target_end = (
                            second_target_start
                            + second_length * base_direction
                        )
                        first_transform = align_reversed_edge_to_segment(
                            first_polygon,
                            first_edge,
                            first_target_start,
                            first_target_end,
                        )
                        second_transform = align_reversed_edge_to_segment(
                            second_polygon,
                            second_edge,
                            second_target_start,
                            second_target_end,
                        )
                        first_placed = first_transform.apply(first_polygon)
                        second_placed = second_transform.apply(second_polygon)

                        # The remaining, unused edges of the two centre pieces
                        # must form a second approximately straight composite
                        # seam for the fourth piece.
                        for first_outer_edge in range(len(first_polygon)):
                            if first_outer_edge == first_edge:
                                continue
                            first_outer_start, first_outer_end = edge_points(
                                first_placed, first_outer_edge
                            )
                            first_outer_vector = (
                                first_outer_end - first_outer_start
                            )
                            first_outer_length = float(
                                np.linalg.norm(first_outer_vector)
                            )
                            for second_outer_edge in range(len(second_polygon)):
                                if second_outer_edge == second_edge:
                                    continue
                                second_outer_start, second_outer_end = edge_points(
                                    second_placed, second_outer_edge
                                )
                                second_outer_vector = (
                                    second_outer_end - second_outer_start
                                )
                                second_outer_length = float(
                                    np.linalg.norm(second_outer_vector)
                                )
                                angle_error = _undirected_angle_error(
                                    first_outer_vector, second_outer_vector
                                )
                                if angle_error > math.radians(
                                    COMPOSITE_COLLINEAR_TOLERANCE_DEG
                                ):
                                    continue
                                endpoint_pairs = (
                                    (first_outer_start, first_outer_end,
                                     second_outer_start, second_outer_end),
                                    (first_outer_start, first_outer_end,
                                     second_outer_end, second_outer_start),
                                    (first_outer_end, first_outer_start,
                                     second_outer_start, second_outer_end),
                                    (first_outer_end, first_outer_start,
                                     second_outer_end, second_outer_start),
                                )
                                # tuple entries are (junction1, far1,
                                # junction2, far2); select the closest junction.
                                junction1, far1, junction2, far2 = min(
                                    endpoint_pairs,
                                    key=lambda values: float(
                                        np.linalg.norm(values[0] - values[2])
                                    ),
                                )
                                junction_gap = float(
                                    np.linalg.norm(junction1 - junction2)
                                )
                                if junction_gap > COMPOSITE_JUNCTION_TOLERANCE_MM:
                                    continue
                                for outer_edge in range(len(outer_polygon)):
                                    outer_length = edge_length(
                                        outer_polygon, outer_edge
                                    )
                                    second_composite_error = abs(
                                        outer_length
                                        - first_outer_length
                                        - second_outer_length
                                    )
                                    if (
                                        second_composite_error
                                        > COMPOSITE_EDGE_TOLERANCE_MM
                                    ):
                                        continue
                                    span = far2 - far1
                                    if float(np.linalg.norm(span)) < 20.0:
                                        continue
                                    outer_transform = (
                                        align_reversed_edge_to_segment(
                                            outer_polygon,
                                            outer_edge,
                                            far1,
                                            far2,
                                        )
                                    )
                                    transforms: list[RigidTransform | None] = [
                                        None
                                    ] * 4
                                    transforms[base_index] = base_transform
                                    transforms[first_middle] = first_transform
                                    transforms[second_middle] = second_transform
                                    transforms[outer_index] = outer_transform
                                    complete = [
                                        value
                                        for value in transforms
                                        if value is not None
                                    ]
                                    if len(complete) != 4:
                                        continue
                                    key = tuple(
                                        value
                                        for index in indices
                                        for value in (
                                            round(
                                                transforms[index].angle_rad, 3
                                            ),
                                            round(
                                                float(
                                                    transforms[index].translation[0]
                                                ),
                                                1,
                                            ),
                                            round(
                                                float(
                                                    transforms[index].translation[1]
                                                ),
                                                1,
                                            ),
                                        )
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    seams = [
                                        SeamMatch(
                                            base_index,
                                            base_edge,
                                            first_middle,
                                            first_edge,
                                            composite_error,
                                        ),
                                        SeamMatch(
                                            base_index,
                                            base_edge,
                                            second_middle,
                                            second_edge,
                                            composite_error,
                                        ),
                                        SeamMatch(
                                            outer_index,
                                            outer_edge,
                                            first_middle,
                                            first_outer_edge,
                                            second_composite_error,
                                        ),
                                        SeamMatch(
                                            outer_index,
                                            outer_edge,
                                            second_middle,
                                            second_outer_edge,
                                            second_composite_error,
                                        ),
                                    ]
                                    mismatch = (
                                        composite_error
                                        + second_composite_error
                                        + 0.2 * junction_gap
                                        + math.degrees(angle_error) * 0.08
                                    )
                                    results.append(
                                        (complete, seams, mismatch)
                                    )
    results.sort(key=lambda item: float(item[2]))
    return results[:MAX_LAYOUT_SOLUTIONS]


def _build_three_segment_composite_assemblies(
    polygons: list[np.ndarray],
) -> list[tuple[list[RigidTransform], list[SeamMatch], float]]:
    """Build four-piece layouts with one long seam split into three edges.

    The fixed 100x60 field set, and valid arbitrary guillotine/Voronoi cuts,
    can contain a long edge on one piece that is shared by three consecutive
    edges from the other pieces.  A spanning-tree edge matcher can align only
    an endpoint segment of that seam; the middle segment has a non-zero offset
    and was therefore unreachable.  Enumerate the three segment orderings,
    place them rigidly along the long edge, then require two additional exact
    contacts to connect the three short-side pieces.
    """
    if len(polygons) != 4:
        return []

    results: list[tuple[list[RigidTransform], list[SeamMatch], float]] = []
    seen: set[tuple[Any, ...]] = set()
    indices = tuple(range(4))

    for base_index in indices:
        base_polygon = polygons[base_index]
        for base_edge in range(len(base_polygon)):
            base_start, base_end = edge_points(base_polygon, base_edge)
            base_vector = base_end - base_start
            base_length = float(np.linalg.norm(base_vector))
            if base_length < 55.0:
                continue
            base_direction = base_vector / base_length
            other_indices = tuple(
                index for index in indices if index != base_index
            )
            for ordered_indices in permutations(other_indices):
                edge_ranges = [
                    range(len(polygons[index])) for index in ordered_indices
                ]
                for first_edge in edge_ranges[0]:
                    first_length = edge_length(
                        polygons[ordered_indices[0]], first_edge
                    )
                    for second_edge in edge_ranges[1]:
                        second_length = edge_length(
                            polygons[ordered_indices[1]], second_edge
                        )
                        for third_edge in edge_ranges[2]:
                            segment_edges = (
                                first_edge, second_edge, third_edge
                            )
                            segment_lengths = (
                                first_length,
                                second_length,
                                edge_length(
                                    polygons[ordered_indices[2]],
                                    third_edge,
                                ),
                            )
                            composite_error = abs(
                                base_length - sum(segment_lengths)
                            )
                            if (
                                composite_error
                                > COMPOSITE_EDGE_TOLERANCE_MM
                            ):
                                continue

                            chain_offset = 0.5 * (
                                base_length - sum(segment_lengths)
                            )
                            cursor = base_start + chain_offset * base_direction
                            transforms_by_index: list[
                                RigidTransform | None
                            ] = [None] * 4
                            transforms_by_index[base_index] = RigidTransform(
                                0.0, np.zeros(2, np.float64)
                            )
                            placed_by_index: list[np.ndarray | None] = [None] * 4
                            placed_by_index[base_index] = base_polygon
                            composite_seams: list[SeamMatch] = []
                            for piece_index, edge_index, length in zip(
                                ordered_indices,
                                segment_edges,
                                segment_lengths,
                            ):
                                target_start = cursor
                                target_end = cursor + length * base_direction
                                transform = align_reversed_edge_to_segment(
                                    polygons[piece_index],
                                    edge_index,
                                    target_start,
                                    target_end,
                                )
                                transforms_by_index[piece_index] = transform
                                placed_by_index[piece_index] = transform.apply(
                                    polygons[piece_index]
                                )
                                composite_seams.append(
                                    SeamMatch(
                                        base_index,
                                        base_edge,
                                        piece_index,
                                        edge_index,
                                        composite_error,
                                    )
                                )
                                cursor = target_end

                            # Find the best unused reversed-edge contact for
                            # every pair of short-side pieces in the placed
                            # geometry.  A valid layout needs two such contacts
                            # whose pair graph spans all three pieces.
                            pair_contacts: list[
                                tuple[float, int, int, int, int, float]
                            ] = []
                            for first_position in range(3):
                                first_index = ordered_indices[first_position]
                                first_placed = placed_by_index[first_index]
                                assert first_placed is not None
                                for second_position in range(
                                    first_position + 1, 3
                                ):
                                    second_index = ordered_indices[
                                        second_position
                                    ]
                                    second_placed = placed_by_index[second_index]
                                    assert second_placed is not None
                                    best_contact = None
                                    for first_other_edge in range(
                                        len(polygons[first_index])
                                    ):
                                        if (
                                            first_other_edge
                                            == segment_edges[first_position]
                                        ):
                                            continue
                                        first_contact_start, first_contact_end = (
                                            edge_points(
                                                first_placed,
                                                first_other_edge,
                                            )
                                        )
                                        for second_other_edge in range(
                                            len(polygons[second_index])
                                        ):
                                            if (
                                                second_other_edge
                                                == segment_edges[second_position]
                                            ):
                                                continue
                                            second_contact_start, second_contact_end = (
                                                edge_points(
                                                    second_placed,
                                                    second_other_edge,
                                                )
                                            )
                                            endpoint_error = max(
                                                float(
                                                    np.linalg.norm(
                                                        first_contact_start
                                                        - second_contact_end
                                                    )
                                                ),
                                                float(
                                                    np.linalg.norm(
                                                        first_contact_end
                                                        - second_contact_start
                                                    )
                                                ),
                                            )
                                            length_error = abs(
                                                edge_length(
                                                    polygons[first_index],
                                                    first_other_edge,
                                                )
                                                - edge_length(
                                                    polygons[second_index],
                                                    second_other_edge,
                                                )
                                            )
                                            contact_cost = (
                                                endpoint_error
                                                + 0.25 * length_error
                                            )
                                            candidate = (
                                                contact_cost,
                                                first_index,
                                                first_other_edge,
                                                second_index,
                                                second_other_edge,
                                                length_error,
                                            )
                                            if (
                                                best_contact is None
                                                or candidate[0]
                                                < best_contact[0]
                                            ):
                                                best_contact = candidate
                                    if (
                                        best_contact is not None
                                        and best_contact[0]
                                        <= MULTI_SEGMENT_CONTACT_TOLERANCE_MM
                                    ):
                                        pair_contacts.append(best_contact)

                            pair_contacts.sort(key=lambda value: value[0])
                            parent = {index: index for index in ordered_indices}

                            def find(index: int) -> int:
                                while parent[index] != index:
                                    parent[index] = parent[parent[index]]
                                    index = parent[index]
                                return index

                            selected_contacts = []
                            for contact in pair_contacts:
                                first_root = find(contact[1])
                                second_root = find(contact[3])
                                if first_root == second_root:
                                    continue
                                parent[first_root] = second_root
                                selected_contacts.append(contact)
                                if len(selected_contacts) == 2:
                                    break
                            if len(selected_contacts) != 2:
                                continue

                            transforms = [
                                value
                                for value in transforms_by_index
                                if value is not None
                            ]
                            if len(transforms) != 4:
                                continue
                            key = tuple(
                                value
                                for index in indices
                                for value in (
                                    round(
                                        transforms_by_index[index].angle_rad,
                                        3,
                                    ),
                                    round(
                                        float(
                                            transforms_by_index[index].translation[0]
                                        ),
                                        1,
                                    ),
                                    round(
                                        float(
                                            transforms_by_index[index].translation[1]
                                        ),
                                        1,
                                    ),
                                )
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            contact_seams = [
                                SeamMatch(
                                    contact[1],
                                    contact[2],
                                    contact[3],
                                    contact[4],
                                    contact[5],
                                )
                                for contact in selected_contacts
                            ]
                            mismatch = composite_error + sum(
                                contact[0] for contact in selected_contacts
                            )
                            results.append(
                                (
                                    transforms,
                                    composite_seams + contact_seams,
                                    mismatch,
                                )
                            )

    results.sort(key=lambda item: float(item[2]))
    return results[:MAX_LAYOUT_SOLUTIONS]


def reconstruct_rectangle(
    pieces: list[dict[str, Any]],
    requested_gap_mm: float = DEFAULT_SEAM_GAP_MM,
    use_boundary_pack: bool = True,
    allow_best_effort: bool = True,
) -> dict[str, Any]:
    """Return a target-local rigid layout or a diagnostic rejection."""
    if not 1 <= len(pieces) <= 4:
        return {"ready": False, "error": "NEED_ONE_TO_FOUR_PIECES"}
    try:
        polygons = [canonical_polygon(piece["vertices_mm"]) for piece in pieces]
    except (KeyError, TypeError, ValueError) as exc:
        return {"ready": False, "error": "INVALID_PIECE_POLYGON", "detail": str(exc)}
    for index, polygon in enumerate(polygons):
        if not 3 <= len(polygon) <= 5:
            return {
                "ready": False,
                "error": "PIECE_VERTEX_COUNT_OUT_OF_RANGE",
                "piece_id": pieces[index].get("id", index),
            }
        measured_edges = [
            edge_length(polygon, edge_index)
            for edge_index in range(len(polygon))
        ]
        # A segmented camera contour can contain one short corner segment even
        # when the manufactured edge satisfies the 20 mm rule.  Detection has
        # already required <=15% polygon/contour area error, so retain those
        # corners instead of collapsing the whole piece to the wrong shape.
        minimum_measured_edge = min(measured_edges)
        camera_area_fit = pieces[index].get("polygon_area_error_ratio")
        short_edge_is_validated_camera_corner = bool(
            camera_area_fit is not None
            and float(camera_area_fit) <= 0.15
            and minimum_measured_edge >= 8.0
        )
        if (
            minimum_measured_edge < 8.0
            or (
                minimum_measured_edge < 16.0
                and not short_edge_is_validated_camera_corner
            )
        ):
            return {
                "ready": False,
                "error": "PIECE_EDGE_CLEARLY_SHORTER_THAN_20MM",
                "piece_id": pieces[index].get("id", index),
                "minimum_edge_mm": round(minimum_measured_edge, 2),
            }

    # The reversed-edge/T-junction solver is normally two orders of magnitude
    # faster than the outer-boundary beam search on real four-piece contours.
    # Try it first and keep the expensive boundary pack only as a compatibility
    # fallback for layouts whose camera edges cannot be paired reliably.  The
    # recursive call disables this fast-path block, so it executes exactly once.
    edge_best_effort_result: dict[str, Any] | None = None
    if use_boundary_pack:
        edge_match_result = reconstruct_rectangle(
            pieces,
            requested_gap_mm=requested_gap_mm,
            use_boundary_pack=False,
            allow_best_effort=True,
        )
        if edge_match_result.get("ready", False):
            edge_match_result.setdefault("search_path", "EDGE_MATCH_FAST_PATH")
            if not edge_match_result.get("best_effort_forced", False):
                return edge_match_result
            edge_best_effort_result = edge_match_result
            forced_reasons = set(
                edge_match_result.get("forced_rejection_reasons", [])
            )
            # An edge topology rejected only for measured contour overlap or
            # vertex offset already has a plausible rectangular envelope.
            # Return its once-refined fallback immediately; the boundary beam
            # cannot improve its topology and would duplicate several seconds
            # of work on the camera thread.
            if (
                float(edge_match_result.get("coverage_ratio", 0.0)) >= 0.86
                and forced_reasons <= {"OVERLAP", "VERTEX_GAP"}
            ):
                return edge_match_result

    total_area = sum(polygon_area(polygon) for polygon in polygons)
    boundary_candidate = (
        _boundary_rectangle_pack(polygons) if use_boundary_pack else None
    )
    if boundary_candidate is None and edge_best_effort_result is not None:
        return edge_best_effort_result
    if boundary_candidate is not None:
        boundary_size = np.asarray(
            boundary_candidate["final_size"], np.float64
        )
        boundary_quality = _global_rectangle_quality(
            boundary_candidate["polygons"],
            float(boundary_size[0]),
            float(boundary_size[1]),
            total_area,
        )
        boundary_candidate["global_quality"] = boundary_quality
        boundary_candidate["score"] = (
            float(boundary_candidate["score"])
            + float(boundary_quality["score"])
        )
    composite_assemblies = (
        []
        if boundary_candidate is not None
        else (
            _build_three_segment_composite_assemblies(polygons)
            + _build_composite_t_junction_assemblies(polygons)
        )
    )
    standard_assemblies = (
        [] if boundary_candidate is not None
        else _build_relative_assemblies(polygons)
    )
    relative_assemblies = standard_assemblies + composite_assemblies
    candidates: list[dict[str, Any]] = (
        [boundary_candidate] if boundary_candidate is not None else []
    )
    best_rejected_candidate: dict[str, Any] | None = None
    best_rejected_score = float("inf")
    seen_layouts: set[tuple[float, ...]] = set()
    rejection_counts = {
        "dimensions": 0,
        "coverage": 0,
        "boundary": 0,
        "duplicate": 0,
        "overlap": 0,
        "vertex_gap": 0,
    }
    smallest_rejected_vertex_gap = float("inf")
    for assembly_index, (transforms, seams, seam_error) in enumerate(
        relative_assemblies
    ):
        is_composite_assembly = assembly_index >= len(standard_assemblies)
        if is_composite_assembly and candidates:
            # A conventional full-edge reconstruction is both more precise
            # and substantially cheaper than the T-junction fallback.
            break
        assembled = [
            transforms[index].apply(polygons[index]) for index in range(len(polygons))
        ]
        all_edge_angles: list[float] = []
        for polygon in assembled:
            for edge_index in range(len(polygon)):
                start, end = edge_points(polygon, edge_index)
                vector = end - start
                all_edge_angles.append(math.atan2(float(vector[1]), float(vector[0])))

        def edge_angle_priority(edge_angle: float) -> float:
            angle = normalize_angle_rad(-edge_angle)
            rotated_points = np.concatenate(
                [rotate_points(polygon, angle) for polygon in assembled],
                axis=0,
            )
            size = np.max(rotated_points, axis=0) - np.min(
                rotated_points, axis=0
            )
            bounding_area = max(1.0, float(size[0] * size[1]))
            # Correct rectangle boundary directions maximize fill before the
            # more expensive overlap/clearance checks.  This makes composite
            # layouts both deterministic and fast without changing topology.
            return -(total_area / bounding_area)

        for edge_angle in sorted(all_edge_angles, key=edge_angle_priority):
            soft_rejection_penalty = 0.0
            soft_rejection_reasons: list[str] = []
            global_angle = normalize_angle_rad(-edge_angle)
            exact = [rotate_points(polygon, global_angle) for polygon in assembled]
            points = np.concatenate(exact, axis=0)
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            width, height = (float(value) for value in maximum - minimum)
            if not _dimensions_allowed(width, height):
                rejection_counts["dimensions"] += 1
                continue
            bounding_area = max(1.0, width * height)
            coverage = total_area / bounding_area
            if not 0.68 <= coverage <= 1.03:
                rejection_counts["coverage"] += 1
                soft_rejection_reasons.append("COVERAGE")
                soft_rejection_penalty += 300.0 * (
                    max(0.0, 0.68 - coverage)
                    + max(0.0, coverage - 1.03)
                )
            boundary_piece_count = sum(
                int(
                    _piece_has_boundary_edge(
                        polygon,
                        minimum,
                        maximum,
                        tolerance=BOUNDARY_TOLERANCE_MM,
                    )
                )
                for polygon in exact
            )
            if (
                not is_composite_assembly
                and boundary_piece_count != len(exact)
            ):
                rejection_counts["boundary"] += 1
                soft_rejection_reasons.append("BOUNDARY")
                soft_rejection_penalty += 35.0 * (
                    len(exact) - boundary_piece_count
                )
            # Most rejected layouts can be ranked using edge mismatch,
            # bounding-box fill and boundary contact alone.  Only a layout
            # capable of improving the retained fallback proceeds to raster
            # quality and overlap processing.  This keeps best-effort
            # bookkeeping O(1) without a second full solver pass.
            cheap_rejected_score = (
                seam_error
                + soft_rejection_penalty
                + 90.0 * abs(1.0 - min(1.05, coverage))
            )
            if (
                soft_rejection_reasons
                and (
                    bool(candidates)
                    or cheap_rejected_score >= best_rejected_score
                )
            ):
                continue
            normalized_exact = [polygon - minimum for polygon in exact]
            # Parallel outer/seam edges produce the same normalized layout
            # several times with different "global edge" choices.  The old
            # order rasterized every duplicate at 1 mm resolution before
            # noticing it was identical, which dominates 2/3-piece runtime on
            # the Pi.  The key depends only on rigid geometry, so deduplicating
            # here is equivalent and avoids repeated expensive quality checks.
            key = _layout_key(normalized_exact)
            if key in seen_layouts:
                rejection_counts["duplicate"] += 1
                continue
            seen_layouts.add(key)
            global_quality = _global_rectangle_quality(
                normalized_exact,
                width,
                height,
                total_area,
            )
            # A candidate can satisfy several local edge matches yet still be
            # a sparse L-shape.  Keep limited camera erosion tolerance, but do
            # not allow a visibly disconnected/mostly empty "rectangle".
            if (
                float(global_quality["raster_fill_ratio"]) < 0.72
                or int(global_quality["connected_components"]) > 2
            ):
                rejection_counts["coverage"] += 1
                soft_rejection_reasons.append("GLOBAL_QUALITY")
                soft_rejection_penalty += (
                    250.0
                    * max(
                        0.0,
                        0.72
                        - float(global_quality["raster_fill_ratio"]),
                    )
                    + 30.0
                    * max(
                        0,
                        int(global_quality["connected_components"]) - 2,
                    )
                )
            quality_rejected_score = (
                seam_error
                + float(global_quality["score"])
                + soft_rejection_penalty
            )
            if (
                soft_rejection_reasons
                and (
                    bool(candidates)
                    or quality_rejected_score >= best_rejected_score
                )
            ):
                continue

            transformed_seams = list(seams)
            selected_gap = (
                min(1.0, max(0.0, float(requested_gap_mm)))
                if is_composite_assembly
                else max(0.0, float(requested_gap_mm))
            )
            separated = normalized_exact
            offsets = [np.zeros(2, np.float64) for _ in exact]
            while selected_gap >= 0.75:
                trial, trial_offsets = _separate_seams(
                    normalized_exact, transformed_seams, selected_gap
                )
                # Rejected layouts are ranked during the scan without the
                # combinatorial grid resolver.  If one becomes the forced
                # winner, overlap refinement runs exactly once below.
                if is_composite_assembly and not soft_rejection_reasons:
                    has_positive_overlap = any(
                        polygons_overlap_with_area(
                            trial[first_index], trial[second_index]
                        )
                        for first_index in range(len(trial))
                        for second_index in range(
                            first_index + 1, len(trial)
                        )
                    )
                    if has_positive_overlap:
                        grid_result = _resolve_composite_overlaps_grid(trial)
                        if grid_result is not None:
                            trial, overlap_offsets = grid_result
                        else:
                            trial, overlap_offsets = _resolve_composite_overlaps(
                                trial, clearance_mm=1.0
                            )
                    else:
                        overlap_offsets = [
                            np.zeros(2, np.float64) for _ in trial
                        ]
                    trial_offsets = [
                        seam_offset + overlap_offset
                        for seam_offset, overlap_offset in zip(
                            trial_offsets, overlap_offsets
                        )
                    ]
                trial_points = np.concatenate(trial, axis=0)
                trial_minimum = np.min(trial_points, axis=0)
                trial = [polygon - trial_minimum for polygon in trial]
                trial_size = np.max(np.concatenate(trial, axis=0), axis=0)
                trial_dimensions_allowed = _dimensions_allowed(
                    float(trial_size[0]), float(trial_size[1])
                )
                if trial_dimensions_allowed:
                    separated = trial
                    offsets = [offset - trial_minimum for offset in trial_offsets]
                    break
                if is_composite_assembly:
                    # The overlap search is the expensive operation and the
                    # relative arrangement is unchanged by retrying smaller
                    # cosmetic gaps.  Reject this topology once rather than
                    # repeating the same search seven times.
                    break
                selected_gap -= 0.5
            final_points = np.concatenate(separated, axis=0)
            final_minimum = np.min(final_points, axis=0)
            separated = [polygon - final_minimum for polygon in separated]
            offsets = [offset - final_minimum for offset in offsets]
            final_size = np.max(np.concatenate(separated, axis=0), axis=0)
            minimum_clearance = float("inf")
            for first_index in range(len(separated)):
                for second_index in range(first_index + 1, len(separated)):
                    minimum_clearance = min(
                        minimum_clearance,
                        polygon_clearance(separated[first_index], separated[second_index]),
                    )
            if len(separated) == 1:
                minimum_clearance = float("inf")
            # ``polygon_clearance`` deliberately returns -1 for any positive
            # overlap, even a sub-millimetre sliver caused by two independent
            # contour fits of the same physical cut.  Let the raster union
            # quantify that case and give pose-graph refinement a chance;
            # substantial overlaps remain a hard rejection.
            if (
                minimum_clearance < -0.5
                and int(global_quality["overlap_pixels"]) > 40
            ):
                rejection_counts["overlap"] += 1
                soft_rejection_reasons.append("OVERLAP")
                soft_rejection_penalty += (
                    500.0
                    + 2.0 * int(global_quality["overlap_pixels"])
                )
            maximum_vertex_gap = max(
                (
                    float(np.linalg.norm(offsets[seam.second_piece] - offsets[seam.first_piece]))
                    for seam in transformed_seams
                ),
                default=0.0,
            )
            if maximum_vertex_gap > MAX_ADJACENT_VERTEX_GAP_MM + 1.0e-6:
                rejection_counts["vertex_gap"] += 1
                smallest_rejected_vertex_gap = min(
                    smallest_rejected_vertex_gap, maximum_vertex_gap
                )
                soft_rejection_reasons.append("VERTEX_GAP")
                soft_rejection_penalty += 20.0 * (
                    maximum_vertex_gap - MAX_ADJACENT_VERTEX_GAP_MM
                )
            score = (
                seam_error
                + float(global_quality["score"])
                + 0.05 * max(0.0, requested_gap_mm - selected_gap)
                + soft_rejection_penalty
            )
            composed = [
                _compose_with_global(transform, global_angle)
                for transform in transforms
            ]
            # DFS records only a spanning tree of matched seams.  A valid
            # four-piece rectangle can have extra contacts (for example a
            # 2x2 layout), so two DFS trees may describe the exact same
            # physical assembly.  Compare the complete contact graph when
            # deciding whether two low-score answers are genuinely distinct.
            full_adjacency = tuple(
                (first, second)
                for first in range(len(normalized_exact))
                for second in range(first + 1, len(normalized_exact))
                if polygon_clearance(
                    normalized_exact[first], normalized_exact[second]
                ) <= 0.12
            )
            candidate = {
                    "score": score,
                    "global_quality": global_quality,
                    "adjacency_signature": full_adjacency,
                    "polygons": separated,
                    "transforms": composed,
                    "offsets": offsets,
                    "pre_gap_minimum": minimum,
                    "final_size": final_size,
                    "coverage": coverage,
                    "seam_error": seam_error,
                    "selected_gap": selected_gap,
                    "minimum_clearance": minimum_clearance,
                    "maximum_vertex_gap": maximum_vertex_gap,
                    "global_angle": global_angle,
                    "exact_minimum_after_global": minimum,
                    "gap_offsets": offsets,
                    "seams": transformed_seams,
                    "best_effort_forced": bool(soft_rejection_reasons),
                    "forced_rejection_reasons": tuple(
                        soft_rejection_reasons
                    ),
                }
            if soft_rejection_reasons:
                if score < best_rejected_score:
                    best_rejected_score = score
                    best_rejected_candidate = candidate
            else:
                candidates.append(candidate)
            if is_composite_assembly and not soft_rejection_reasons:
                # All global-edge rotations of the same rigid composite
                # topology describe the same physical placement.  Once one
                # passes the rectangle, overlap and vertex-gap checks, do not
                # repeat the expensive overlap search for its other edges.
                break
    if not candidates:
        if allow_best_effort and best_rejected_candidate is not None:
            best_rejected_candidate["search_path"] = (
                "FORCED_BEST_REJECTED_CANDIDATE"
            )
            best_rejected_candidate["normal_rejection_counts"] = dict(
                rejection_counts
            )
            candidates.append(best_rejected_candidate)
        else:
            return {
                "ready": False,
                "error": "NO_RECTANGULAR_EDGE_MATCH_SOLUTION",
                "relative_assemblies_checked": len(relative_assemblies),
                "composite_assemblies_checked": len(composite_assemblies),
                "rejection_counts": rejection_counts,
                "smallest_rejected_vertex_gap_mm": (
                    None
                    if not math.isfinite(smallest_rejected_vertex_gap)
                    else round(smallest_rejected_vertex_gap, 2)
                ),
            }
    candidates.sort(key=lambda value: float(value["score"]))
    best = candidates[0]
    if best.get("best_effort_forced", False):
        forced_grid_result = _resolve_composite_overlaps_grid(
            best["polygons"]
        )
        if forced_grid_result is not None:
            forced_polygons, forced_offsets = forced_grid_result
            forced_points = np.concatenate(forced_polygons, axis=0)
            forced_minimum = np.min(forced_points, axis=0)
            best["polygons"] = [
                polygon - forced_minimum for polygon in forced_polygons
            ]
            best["gap_offsets"] = [
                existing + additional - forced_minimum
                for existing, additional in zip(
                    best["gap_offsets"], forced_offsets
                )
            ]
            best["offsets"] = list(best["gap_offsets"])
            best["final_size"] = np.max(
                np.concatenate(best["polygons"], axis=0), axis=0
            )
            best["minimum_clearance"] = min(
                (
                    polygon_clearance(
                        best["polygons"][first],
                        best["polygons"][second],
                    )
                    for first in range(len(best["polygons"]))
                    for second in range(first + 1, len(best["polygons"]))
                ),
                default=float("inf"),
            )
            best["maximum_vertex_gap"] = max(
                (
                    float(
                        np.linalg.norm(
                            best["gap_offsets"][seam.second_piece]
                            - best["gap_offsets"][seam.first_piece]
                        )
                    )
                    for seam in best.get("seams", [])
                ),
                default=0.0,
            )
    different = next(
        (
            candidate
            for candidate in candidates[1:]
            if candidate["adjacency_signature"] != best["adjacency_signature"]
        ),
        None,
    )
    confidence_margin = (
        None if different is None else float(different["score"] - best["score"])
    )
    # A very small margin between different adjacency graphs means the camera
    # data cannot uniquely determine which edges belong together.
    ambiguous_best_selected = bool(
        confidence_margin is not None and confidence_margin < 0.08
    )

    _refine_winning_candidate(best, polygons, total_area)

    target_polygons = best["polygons"]
    moves: list[dict[str, Any]] = []
    for index, piece in enumerate(pieces):
        pick = np.asarray(
            piece.get("pick_point_mm", piece.get("center_mm")), np.float64
        ).reshape(2)
        transform: RigidTransform = best["transforms"][index]
        target_pick = transform.apply(pick.reshape(1, 2))[0]
        target_pick -= best["exact_minimum_after_global"]
        target_pick += best["gap_offsets"][index]
        all_points = np.concatenate(target_polygons, axis=0)
        target_pick -= np.min(all_points, axis=0)
        moves.append(
            {
                "piece_id": str(piece.get("id", f"W{index + 1}")),
                "area_mm2": round(
                    float(piece.get("area_mm2", polygon_area(polygons[index]))),
                    2,
                ),
                "pick_a4_mm": np.round(pick, 2).tolist(),
                "pick_method": piece.get("pick_method", "AREA_CENTROID"),
                "target_pick_local_mm": np.round(target_pick, 2).tolist(),
                "rotate_deg_clockwise": round(
                    math.degrees(transform.angle_rad), 2
                ),
                "target_vertices_local_mm": np.round(
                    target_polygons[index], 2
                ).tolist(),
                "pre_optimization_vertices_local_mm": np.round(
                    best["pre_optimization_polygons"][index], 2
                ).tolist(),
            }
        )
    pose_graph = dict(best.get("pose_graph") or {})
    matched_seams = [
        {
            "first_piece_id": str(
                pieces[seam.first_piece].get(
                    "id", f"W{seam.first_piece + 1}"
                )
            ),
            "first_edge": seam.first_edge,
            "second_piece_id": str(
                pieces[seam.second_piece].get(
                    "id", f"W{seam.second_piece + 1}"
                )
            ),
            "second_edge": seam.second_edge,
            "length_error_mm": round(float(seam.length_error_mm), 3),
        }
        for seam in best.get("seams", [])
    ]
    return {
        "ready": True,
        "error": None,
        "search_path": best.get("search_path"),
        "normal_rejection_counts": best.get("normal_rejection_counts"),
        "solver": best.get("solver", "rigid_reversed_edge_search_v1"),
        "piece_count": len(pieces),
        "layout_size_mm": np.round(best["final_size"], 2).tolist(),
        "coverage_ratio": round(float(best["coverage"]), 4),
        "edge_match_error_mm": round(float(best["seam_error"]), 3),
        "requested_seam_gap_mm": round(float(requested_gap_mm), 2),
        "actual_seam_gap_mm": round(float(best["selected_gap"]), 2),
        "minimum_piece_clearance_mm": (
            None
            if not math.isfinite(float(best["minimum_clearance"]))
            else round(float(best["minimum_clearance"]), 2)
        ),
        "maximum_adjacent_vertex_gap_mm": round(
            float(best["maximum_vertex_gap"]), 2
        ),
        "score": round(float(best["score"]), 4),
        "global_quality": {
            key: (
                round(float(value), 4)
                if isinstance(value, (float, np.floating))
                else int(value)
            )
            for key, value in dict(best.get("global_quality") or {}).items()
        },
        "pose_graph_optimization": {
            key: (
                round(float(value), 6)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in pose_graph.items()
        },
        "matched_seams": matched_seams,
        "confidence_margin": (
            None if confidence_margin is None else round(confidence_margin, 4)
        ),
        "confidence": round(
            1.0
            if confidence_margin is None
            else min(1.0, max(0.0, confidence_margin / 1.5)),
            3,
        ),
        "best_effort_forced": bool(best.get("best_effort_forced", False)),
        "forced_rejection_reasons": list(
            best.get("forced_rejection_reasons", ())
        ),
        "ambiguous_best_selected": ambiguous_best_selected,
        "moves": moves,
    }


def _route_metric(
    order: tuple[int, ...],
    picks: list[np.ndarray],
    places: list[np.ndarray],
    home: np.ndarray,
) -> float:
    current = home
    total = 0.0
    for index in order:
        total += max(abs(float(picks[index][0] - current[0])),
                     abs(float(picks[index][1] - current[1])))
        total += max(abs(float(places[index][0] - picks[index][0])),
                     abs(float(places[index][1] - picks[index][1])))
        current = places[index]
    total += max(abs(float(home[0] - current[0])), abs(float(home[1] - current[1])))
    return total


def place_in_upper_half(
    reconstruction: dict[str, Any],
    divider_y_mm: float = DEFAULT_DIVIDER_Y_MM,
    divider_margin_mm: float = DEFAULT_DIVIDER_MARGIN_MM,
    home_a4_mm: tuple[float, float] = (5.0, 62.0),
    reachable_x_mm: tuple[float, float] = (5.0, 205.0),
    reachable_y_mm: tuple[float, float] = (62.0, 287.0),
) -> dict[str, Any]:
    if not reconstruction.get("ready", False):
        return reconstruction
    source_moves = list(reconstruction.get("moves") or [])
    picks = [np.asarray(move["pick_a4_mm"], np.float64) for move in source_moves]
    unreachable_picks = [
        source_moves[index]["piece_id"]
        for index, pick in enumerate(picks)
        if not (
            reachable_x_mm[0] <= pick[0] <= reachable_x_mm[1]
            and reachable_y_mm[0] <= pick[1] <= reachable_y_mm[1]
            and pick[1] > divider_y_mm + divider_margin_mm
        )
    ]
    width, height = (float(value) for value in reconstruction["layout_size_mm"])
    x_min = 0.0
    x_max = A4_WIDTH_MM - width
    y_min = 0.0
    y_max = divider_y_mm - divider_margin_mm - height
    if x_max < x_min or y_max < y_min:
        result = dict(reconstruction)
        result.update({"ready": False, "error": "TARGET_RECTANGLE_DOES_NOT_FIT_UPPER_HALF"})
        return result

    local_places = [
        np.asarray(move["target_pick_local_mm"], np.float64)
        for move in source_moves
    ]
    piece_areas = np.asarray(
        [float(move.get("area_mm2", 0.0)) for move in source_moves],
        np.float64,
    )
    target_center_y = np.asarray(
        [
            float(
                polygon_centroid(
                    np.asarray(move["target_vertices_local_mm"], np.float64)
                )[1]
            )
            for move in source_moves
        ],
        np.float64,
    )
    if len(source_moves) >= 2:
        smallest_index = int(np.argmin(piece_areas))
        largest_index = int(np.argmax(piece_areas))
        # User-selected placement preference: the smallest fragment stays
        # near the divider, and the largest, most stable fragment sits near
        # the bottom edge of the reconstructed rectangle.
        normalized_height = max(height, 1.0)
        smallest_divider_distance = (
            (height - target_center_y[smallest_index]) / normalized_height
        )
        largest_bottom_distance = (
            (height - target_center_y[largest_index]) / normalized_height
        )
        mean_divider_distance = (
            (height - float(np.mean(target_center_y))) / normalized_height
        )
        # This score is compared before route length in solve_arbitrary_puzzle:
        # small piece near the divider, large piece near the rectangle bottom,
        # then all piece centres collectively near the divider.
        vertical_priority_penalty = (
            10.0 * smallest_divider_distance
            + 10.0 * largest_bottom_distance
            + mean_divider_distance
        )
    else:
        smallest_index = 0
        largest_index = 0
        smallest_divider_distance = 0.0
        largest_bottom_distance = 0.0
        mean_divider_distance = (
            (height - float(target_center_y[0])) / max(height, 1.0)
        )
        vertical_priority_penalty = 0.0
    home = np.asarray(home_a4_mm, np.float64)
    best: tuple[float, np.ndarray, tuple[int, ...], list[np.ndarray]] | None = None
    # Keep the reconstructed rectangle centred along the A4 divider.  The old
    # two-millimetre X search selected the shortest gantry route and therefore
    # biased the result toward the home position on the left.
    centered_x = 0.5 * (A4_WIDTH_MM - width)
    x_values = np.asarray([min(x_max, max(x_min, centered_x))], np.float64)
    # The source is the visually lower half.  Put the reconstructed rectangle
    # as far down as allowed in the visually upper half, i.e. directly beside
    # the divider.  Moving it toward the A4 top would only enter the gantry's
    # unreachable Y<62 mm band.
    y_values = np.asarray([float(y_max)], np.float64)
    for origin_y in y_values:
        for origin_x in x_values:
            origin = np.asarray([origin_x, origin_y], np.float64)
            places = [local + origin for local in local_places]
            if not all(
                reachable_x_mm[0] <= place[0] <= reachable_x_mm[1]
                and reachable_y_mm[0] <= place[1] <= reachable_y_mm[1]
                for place in places
            ):
                continue
            for order in permutations(range(len(source_moves))):
                metric = _route_metric(order, picks, places, home)
                if best is None or metric < best[0]:
                    best = (metric, origin, order, places)
    if best is None:
        result = dict(reconstruction)
        result.update({"ready": False, "error": "NO_REACHABLE_UPPER_HALF_TARGET"})
        return result

    metric, origin, order, places = best
    ordered_moves: list[dict[str, Any]] = []
    for sequence, index in enumerate(order, start=1):
        move = dict(source_moves[index])
        move["order"] = sequence
        move["place_a4_mm"] = np.round(places[index], 2).tolist()
        move["target_vertices_mm"] = np.round(
            np.asarray(move["target_vertices_local_mm"], np.float64) + origin,
            2,
        ).tolist()
        ordered_moves.append(move)
    result = dict(reconstruction)
    result.update(
        {
            "ready": not bool(unreachable_picks),
            "motion_ready": not bool(unreachable_picks),
            "geometry_ready": True,
            "error": (
                "SOURCE_PICK_POINT_UNREACHABLE_OR_NOT_LOWER_HALF"
                if unreachable_picks
                else None
            ),
            "unreachable_piece_ids": unreachable_picks,
            "source_region": "lower",
            "target_region": "upper",
            "divider_y_mm": round(float(divider_y_mm), 2),
            "divider_margin_mm": round(float(divider_margin_mm), 2),
            "target_rectangle_mm": {
                "origin": np.round(origin, 2).tolist(),
                "size": [round(width, 2), round(height, 2)],
                "center": np.round(origin + np.asarray([width, height]) * 0.5, 2).tolist(),
            },
            "target_horizontal_alignment": "A4_DIVIDER_CENTER",
            "route_metric_mm": round(float(metric), 2),
            "vertical_priority_penalty": round(
                float(vertical_priority_penalty), 2
            ),
            "placement_priority_score": round(
                float(vertical_priority_penalty), 6
            ),
            "placement_priority": {
                "smallest_piece_id": str(
                    source_moves[smallest_index].get("piece_id", "?")
                ),
                "largest_piece_id": str(
                    source_moves[largest_index].get("piece_id", "?")
                ),
                "smallest_piece_divider_distance_ratio": round(
                    float(smallest_divider_distance), 4
                ),
                "largest_piece_bottom_distance_ratio": round(
                    float(largest_bottom_distance), 4
                ),
                "mean_piece_divider_distance_ratio": round(
                    float(mean_divider_distance), 4
                ),
            },
            "vision_to_stage_y": "IDENTITY",
            "stage_reachable_range_mm": {
                "x": [float(reachable_x_mm[0]), float(reachable_x_mm[1])],
                "y": [float(reachable_y_mm[0]), float(reachable_y_mm[1])],
            },
            "moves": ordered_moves,
        }
    )
    return result


# Compatibility alias for older imports.  The current physical setup places
# question-2 source fragments in the lower half and reconstructs above them.
place_in_lower_half = place_in_upper_half


def solve_arbitrary_puzzle(
    pieces: list[dict[str, Any]],
    divider_y_mm: float = DEFAULT_DIVIDER_Y_MM,
    seam_gap_mm: float = DEFAULT_SEAM_GAP_MM,
    **placement_options: Any,
) -> dict[str, Any]:
    reconstruction = reconstruct_rectangle(pieces, seam_gap_mm)
    if not reconstruction.get("ready", False):
        return reconstruction

    width, height = (float(value) for value in reconstruction["layout_size_mm"])
    alternatives: list[dict[str, Any]] = []
    for quarter_turn in range(4):
        candidate = dict(reconstruction)
        rotated_moves: list[dict[str, Any]] = []
        for source_move in reconstruction.get("moves", []):
            move = dict(source_move)
            pick = np.asarray(move["target_pick_local_mm"], np.float64)
            polygon = np.asarray(
                move["target_vertices_local_mm"], np.float64
            )
            if quarter_turn == 0:
                rotated_pick = pick
                rotated_polygon = polygon
                rotated_size = (width, height)
            elif quarter_turn == 1:
                rotated_pick = np.asarray([height - pick[1], pick[0]])
                rotated_polygon = np.column_stack(
                    (height - polygon[:, 1], polygon[:, 0])
                )
                rotated_size = (height, width)
            elif quarter_turn == 2:
                rotated_pick = np.asarray(
                    [width - pick[0], height - pick[1]]
                )
                rotated_polygon = np.column_stack(
                    (width - polygon[:, 0], height - polygon[:, 1])
                )
                rotated_size = (width, height)
            else:
                rotated_pick = np.asarray([pick[1], width - pick[0]])
                rotated_polygon = np.column_stack(
                    (polygon[:, 1], width - polygon[:, 0])
                )
                rotated_size = (height, width)
            move["target_pick_local_mm"] = np.round(
                rotated_pick, 2
            ).tolist()
            move["target_vertices_local_mm"] = np.round(
                rotated_polygon, 2
            ).tolist()
            move["rotate_deg_clockwise"] = round(
                math.degrees(
                    normalize_angle_rad(
                        math.radians(float(move["rotate_deg_clockwise"]))
                        + quarter_turn * math.pi * 0.5
                    )
                ),
                2,
            )
            rotated_moves.append(move)
        candidate["layout_size_mm"] = [
            round(float(rotated_size[0]), 2),
            round(float(rotated_size[1]), 2),
        ]
        candidate["moves"] = rotated_moves
        candidate["target_global_quarter_turn_deg"] = float(
            quarter_turn * 90
        )
        alternatives.append(candidate)

    placed = [
        place_in_upper_half(
            candidate,
            divider_y_mm=divider_y_mm,
            **placement_options,
        )
        for candidate in alternatives
    ]
    feasible = [candidate for candidate in placed if candidate.get("ready", False)]
    if not feasible:
        geometry_only = [
            candidate
            for candidate in placed
            if candidate.get("geometry_ready", False)
        ]
        if geometry_only:
            return min(
                geometry_only,
                key=lambda candidate: (
                    float(candidate.get("placement_priority_score", math.inf)),
                    float(candidate["route_metric_mm"]),
                ),
            )
        return placed[0]
    best = min(
        feasible,
        key=lambda candidate: (
            float(candidate.get("placement_priority_score", math.inf)),
            float(candidate["route_metric_mm"]),
        ),
    )
    return best


__all__ = [
    "DEFAULT_DIVIDER_MARGIN_MM",
    "DEFAULT_DIVIDER_Y_MM",
    "DEFAULT_SEAM_GAP_MM",
    "MAX_ADJACENT_VERTEX_GAP_MM",
    "canonical_polygon",
    "place_in_lower_half",
    "place_in_upper_half",
    "polygon_area",
    "polygon_centroid",
    "reconstruct_rectangle",
    "solve_arbitrary_puzzle",
]
