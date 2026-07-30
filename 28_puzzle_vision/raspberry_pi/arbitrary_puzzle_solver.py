#!/usr/bin/env python3
"""Geometry-only solver for E-question 2(1) arbitrary white pieces.

The input polygons are measured in the physical A4 coordinate system.  The
solver never mirrors a polygon: it only applies rigid rotations/translations,
matches reversed edges, verifies the reconstructed rectangular outline, adds
a small mechanical seam, and places the result in the reachable lower half.

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
MAX_ADJACENT_VERTEX_GAP_MM = 15.0
EDGE_MATCH_TOLERANCE_MM = 3.0
BOUNDARY_TOLERANCE_MM = 3.0
MIN_TARGET_WIDTH_MM = 90.0
MAX_TARGET_WIDTH_MM = 120.0
MIN_TARGET_HEIGHT_MM = 50.0
MAX_TARGET_HEIGHT_MM = 90.0
MAX_LAYOUT_SOLUTIONS = 320


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
) -> RigidTransform:
    moving_start, moving_end = edge_points(moving, moving_edge)
    fixed_start, fixed_end = edge_points(fixed, fixed_edge)
    source_vector = moving_end - moving_start
    target_vector = fixed_start - fixed_end
    angle = normalize_angle_rad(
        math.atan2(float(target_vector[1]), float(target_vector[0]))
        - math.atan2(float(source_vector[1]), float(source_vector[0]))
    )
    rotated_start = rotate_points(moving_start.reshape(1, 2), angle)[0]
    translation = fixed_end - rotated_start
    return RigidTransform(angle, translation)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


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
        MIN_TARGET_WIDTH_MM - 2.0
        <= long_side
        <= MAX_TARGET_WIDTH_MM + 2.0
        and MIN_TARGET_HEIGHT_MM - 2.0
        <= short_side
        <= MAX_TARGET_HEIGHT_MM + 2.0
    )


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

        candidates: list[tuple[float, int, int, int, int, RigidTransform]] = []
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
                        if difference > EDGE_MATCH_TOLERANCE_MM:
                            continue
                        transform = align_reversed_edge(
                            moving_polygon,
                            moving_edge,
                            fixed_polygon,
                            fixed_edge,
                        )
                        transformed = transform.apply(moving_polygon)
                        if any(
                            polygons_overlap_with_area(
                                transformed,
                                transforms[index].apply(polygons[index]),  # type: ignore[union-attr]
                            )
                            for index in placed
                        ):
                            continue
                        candidates.append(
                            (
                                difference,
                                moving_index,
                                moving_edge,
                                fixed_index,
                                fixed_edge,
                                transform,
                            )
                        )
        candidates.sort(key=lambda value: value[0])
        for difference, moving_index, moving_edge, fixed_index, fixed_edge, transform in candidates[:80]:
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
                recurse(placed, seams + [seam], mismatch + difference)
            placed.remove(moving_index)
            transforms[moving_index] = None

    recurse({0}, [], 0.0)
    return results


def reconstruct_rectangle(
    pieces: list[dict[str, Any]],
    requested_gap_mm: float = DEFAULT_SEAM_GAP_MM,
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
        if min(measured_edges) < 16.0:
            return {
                "ready": False,
                "error": "PIECE_EDGE_CLEARLY_SHORTER_THAN_20MM",
                "piece_id": pieces[index].get("id", index),
                "minimum_edge_mm": round(min(measured_edges), 2),
            }

    relative_assemblies = _build_relative_assemblies(polygons)
    candidates: list[dict[str, Any]] = []
    seen_layouts: set[tuple[float, ...]] = set()
    total_area = sum(polygon_area(polygon) for polygon in polygons)
    for transforms, seams, seam_error in relative_assemblies:
        assembled = [
            transforms[index].apply(polygons[index]) for index in range(len(polygons))
        ]
        all_edge_angles: list[float] = []
        for polygon in assembled:
            for edge_index in range(len(polygon)):
                start, end = edge_points(polygon, edge_index)
                vector = end - start
                all_edge_angles.append(math.atan2(float(vector[1]), float(vector[0])))
        for edge_angle in all_edge_angles:
            global_angle = normalize_angle_rad(-edge_angle)
            exact = [rotate_points(polygon, global_angle) for polygon in assembled]
            points = np.concatenate(exact, axis=0)
            minimum = np.min(points, axis=0)
            maximum = np.max(points, axis=0)
            width, height = (float(value) for value in maximum - minimum)
            if not _dimensions_allowed(width, height):
                continue
            bounding_area = max(1.0, width * height)
            coverage = total_area / bounding_area
            if not 0.72 <= coverage <= 1.03:
                continue
            if not all(
                _piece_has_boundary_edge(polygon, minimum, maximum)
                for polygon in exact
            ):
                continue
            normalized_exact = [polygon - minimum for polygon in exact]
            key = _layout_key(normalized_exact)
            if key in seen_layouts:
                continue
            seen_layouts.add(key)

            transformed_seams = list(seams)
            selected_gap = max(0.0, float(requested_gap_mm))
            separated = normalized_exact
            offsets = [np.zeros(2, np.float64) for _ in exact]
            while selected_gap >= 0.75:
                trial, trial_offsets = _separate_seams(
                    normalized_exact, transformed_seams, selected_gap
                )
                trial_points = np.concatenate(trial, axis=0)
                trial_minimum = np.min(trial_points, axis=0)
                trial = [polygon - trial_minimum for polygon in trial]
                trial_size = np.max(np.concatenate(trial, axis=0), axis=0)
                if _dimensions_allowed(float(trial_size[0]), float(trial_size[1])):
                    separated = trial
                    offsets = [offset - trial_minimum for offset in trial_offsets]
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
            if minimum_clearance < -0.5:
                continue
            maximum_vertex_gap = max(
                (
                    float(np.linalg.norm(offsets[seam.second_piece] - offsets[seam.first_piece]))
                    for seam in transformed_seams
                ),
                default=0.0,
            )
            if maximum_vertex_gap > MAX_ADJACENT_VERTEX_GAP_MM + 1.0e-6:
                continue
            score = (
                seam_error
                + 24.0 * abs(1.0 - min(1.0, coverage))
                + 0.05 * max(0.0, requested_gap_mm - selected_gap)
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
            candidates.append(
                {
                    "score": score,
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
                }
            )
    if not candidates:
        return {
            "ready": False,
            "error": "NO_RECTANGULAR_EDGE_MATCH_SOLUTION",
            "relative_assemblies_checked": len(relative_assemblies),
        }
    candidates.sort(key=lambda value: float(value["score"]))
    best = candidates[0]
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
    if confidence_margin is not None and confidence_margin < 0.08:
        return {
            "ready": False,
            "error": "AMBIGUOUS_RECTANGLE_SOLUTION",
            "best_score": round(float(best["score"]), 4),
            "confidence_margin": round(confidence_margin, 4),
        }

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
                "pick_a4_mm": np.round(pick, 2).tolist(),
                "pick_method": piece.get("pick_method", "AREA_CENTROID"),
                "target_pick_local_mm": np.round(target_pick, 2).tolist(),
                "rotate_deg_clockwise": round(
                    math.degrees(transform.angle_rad), 2
                ),
                "target_vertices_local_mm": np.round(
                    target_polygons[index], 2
                ).tolist(),
            }
        )
    return {
        "ready": True,
        "error": None,
        "solver": "rigid_reversed_edge_search_v1",
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
        "confidence_margin": (
            None if confidence_margin is None else round(confidence_margin, 4)
        ),
        "confidence": round(
            1.0
            if confidence_margin is None
            else min(1.0, max(0.0, confidence_margin / 1.5)),
            3,
        ),
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


def place_in_lower_half(
    reconstruction: dict[str, Any],
    divider_y_mm: float = DEFAULT_DIVIDER_Y_MM,
    divider_margin_mm: float = DEFAULT_DIVIDER_MARGIN_MM,
    home_a4_mm: tuple[float, float] = (5.0, 61.0),
    reachable_x_mm: tuple[float, float] = (5.0, 205.0),
    reachable_y_mm: tuple[float, float] = (61.0, 286.0),
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
            and pick[1] < divider_y_mm - divider_margin_mm
        )
    ]
    if unreachable_picks:
        result = dict(reconstruction)
        result.update(
            {
                "ready": False,
                "error": "SOURCE_PICK_POINT_UNREACHABLE_OR_NOT_UPPER_HALF",
                "unreachable_piece_ids": unreachable_picks,
            }
        )
        return result

    width, height = (float(value) for value in reconstruction["layout_size_mm"])
    x_min = 0.0
    x_max = A4_WIDTH_MM - width
    y_min = divider_y_mm + divider_margin_mm
    y_max = A4_HEIGHT_MM - height
    if x_max < x_min or y_max < y_min:
        result = dict(reconstruction)
        result.update({"ready": False, "error": "TARGET_RECTANGLE_DOES_NOT_FIT_LOWER_HALF"})
        return result

    local_places = [
        np.asarray(move["target_pick_local_mm"], np.float64)
        for move in source_moves
    ]
    home = np.asarray(home_a4_mm, np.float64)
    best: tuple[float, np.ndarray, tuple[int, ...], list[np.ndarray]] | None = None
    x_values = np.arange(math.ceil(x_min), math.floor(x_max) + 0.1, 2.0)
    y_values = np.arange(math.ceil(y_min), math.floor(y_max) + 0.1, 2.0)
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
                # Prefer a target close to the divider when route metrics tie.
                metric += 0.01 * (origin_y - y_min)
                if best is None or metric < best[0]:
                    best = (metric, origin, order, places)
    if best is None:
        result = dict(reconstruction)
        result.update({"ready": False, "error": "NO_REACHABLE_LOWER_HALF_TARGET"})
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
            "ready": True,
            "error": None,
            "source_region": "upper",
            "target_region": "lower",
            "divider_y_mm": round(float(divider_y_mm), 2),
            "divider_margin_mm": round(float(divider_margin_mm), 2),
            "target_rectangle_mm": {
                "origin": np.round(origin, 2).tolist(),
                "size": [round(width, 2), round(height, 2)],
                "center": np.round(origin + np.asarray([width, height]) * 0.5, 2).tolist(),
            },
            "route_metric_mm": round(float(metric), 2),
            "moves": ordered_moves,
        }
    )
    return result


def solve_arbitrary_puzzle(
    pieces: list[dict[str, Any]],
    divider_y_mm: float = DEFAULT_DIVIDER_Y_MM,
    seam_gap_mm: float = DEFAULT_SEAM_GAP_MM,
    **placement_options: Any,
) -> dict[str, Any]:
    reconstruction = reconstruct_rectangle(pieces, seam_gap_mm)
    if not reconstruction.get("ready", False):
        return reconstruction

    reconstruction["target_global_quarter_turn_deg"] = 0.0
    alternatives = [reconstruction]
    width, height = (float(value) for value in reconstruction["layout_size_mm"])
    rotated = dict(reconstruction)
    rotated_moves: list[dict[str, Any]] = []
    for source_move in reconstruction.get("moves", []):
        move = dict(source_move)
        pick = np.asarray(move["target_pick_local_mm"], np.float64)
        polygon = np.asarray(move["target_vertices_local_mm"], np.float64)
        # Standard +90 degree rotation followed by an X translation keeps the
        # target local coordinates non-negative: (x,y) -> (height-y, x).
        move["target_pick_local_mm"] = np.round(
            np.asarray([height - pick[1], pick[0]], np.float64), 2
        ).tolist()
        move["target_vertices_local_mm"] = np.round(
            np.column_stack((height - polygon[:, 1], polygon[:, 0])), 2
        ).tolist()
        move["rotate_deg_clockwise"] = round(
            math.degrees(
                normalize_angle_rad(
                    math.radians(float(move["rotate_deg_clockwise"]))
                    + math.pi * 0.5
                )
            ),
            2,
        )
        rotated_moves.append(move)
    rotated["layout_size_mm"] = [round(height, 2), round(width, 2)]
    rotated["moves"] = rotated_moves
    rotated["target_global_quarter_turn_deg"] = 90.0
    alternatives.append(rotated)

    placed = [
        place_in_lower_half(
            candidate,
            divider_y_mm=divider_y_mm,
            **placement_options,
        )
        for candidate in alternatives
    ]
    feasible = [candidate for candidate in placed if candidate.get("ready", False)]
    if not feasible:
        return placed[0]
    best = min(feasible, key=lambda candidate: float(candidate["route_metric_mm"]))
    return best


__all__ = [
    "DEFAULT_DIVIDER_MARGIN_MM",
    "DEFAULT_DIVIDER_Y_MM",
    "DEFAULT_SEAM_GAP_MM",
    "MAX_ADJACENT_VERTEX_GAP_MM",
    "canonical_polygon",
    "place_in_lower_half",
    "polygon_area",
    "polygon_centroid",
    "reconstruct_rectangle",
    "solve_arbitrary_puzzle",
]
