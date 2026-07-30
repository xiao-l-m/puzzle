#!/usr/bin/env python3
"""A4-based detector for the four galvanized puzzle pieces in E-question 1."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
from itertools import permutations
import json
import math
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from gantry_serial import GantryTaskController
from puzzle_motion import (
    Mode1TargetLatch,
    MotionPlanStabilityGate,
    StageCalibration,
    build_task_plan,
    estimate_plan_seconds,
)


DEFAULT_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-DHZJ-240229-XH_Integrated_Webcam_HD-video-index0"
)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
WARP_PX_PER_MM = 4.0
WARP_WIDTH = int(round(A4_WIDTH_MM * WARP_PX_PER_MM))
WARP_HEIGHT = int(round(A4_HEIGHT_MM * WARP_PX_PER_MM))
TARGET_RECT_WIDTH_MM = 100.0
TARGET_RECT_HEIGHT_MM = 60.0
ASSEMBLY_DIVIDER_GAP_MM = 18.0
ASSEMBLY_PIECE_SPACING_MM = 10.0
ASSEMBLY_STAGE_MIN_CENTER_Y_MM = 61.0
ASSEMBLY_STAGE_MAX_CENTER_Y_MM = 286.0
ASSEMBLY_STAGE_MIN_CENTER_X_MM = 5.0
ASSEMBLY_STAGE_MAX_CENTER_X_MM = 205.0
ASSEMBLY_MIN_CLEARANCE_MM = 3.0
ASSEMBLY_MAX_ADJACENT_VERTEX_GAP_MM = 15.0

# Figure 2 dimensions, in a 100 x 60 mm target-local coordinate frame.
# The main seam is the 6-8-10 right-triangle diagonal from (20, 0) to
# (100, 60). Q is 20 mm from its start and E is 30 mm from its end.
TARGET_PIECE_TEMPLATES: tuple[tuple[str, np.ndarray], ...] = (
    (
        "G1_BIG_TRIANGLE",
        np.asarray([[20, 0], [100, 0], [100, 60]], np.float64),
    ),
    (
        "G2_SMALL_TOP_LEFT",
        np.asarray([[0, 0], [20, 0], [36, 12], [0, 20]], np.float64),
    ),
    (
        "G3_MIDDLE",
        np.asarray([[0, 20], [36, 12], [76, 42], [0, 30]], np.float64),
    ),
    (
        "G4_BOTTOM",
        np.asarray([[0, 30], [76, 42], [100, 60], [0, 60]], np.float64),
    ),
)

# Translation directions point away from the seams of the exact Figure-2
# tiling.  The solver below scales them and validates the resulting polygons;
# these are search directions, not final hard-coded target offsets.
ASSEMBLY_OFFSET_DIRECTIONS: dict[str, np.ndarray] = {
    "G1_BIG_TRIANGLE": np.asarray([3.0, -4.0], np.float64),
    "G2_SMALL_TOP_LEFT": np.asarray([-4.0, -3.0], np.float64),
    "G3_MIDDLE": np.asarray([-5.0, 1.0], np.float64),
    "G4_BOTTOM": np.asarray([-1.0, 5.0], np.float64),
}
ASSEMBLY_ADJACENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("G1_BIG_TRIANGLE", "G2_SMALL_TOP_LEFT"),
    ("G1_BIG_TRIANGLE", "G3_MIDDLE"),
    ("G1_BIG_TRIANGLE", "G4_BOTTOM"),
    ("G2_SMALL_TOP_LEFT", "G3_MIDDLE"),
    ("G3_MIDDLE", "G4_BOTTOM"),
)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    difference = points[:, 0] - points[:, 1]
    return np.asarray(
        [
            points[np.argmin(total)],
            points[np.argmax(difference)],
            points[np.argmax(total)],
            points[np.argmin(difference)],
        ],
        dtype=np.float32,
    )


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    return encoded.tobytes() if ok else None


def pca_angle_deg(contour: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(points))
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def polygon_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    signed_area = 0.5 * float(np.sum(cross))
    if abs(signed_area) < 1.0e-9:
        return 0.0, np.mean(polygon, axis=0)
    centroid = np.asarray(
        [
            np.sum((polygon[:, 0] + following[:, 0]) * cross),
            np.sum((polygon[:, 1] + following[:, 1]) * cross),
        ],
        dtype=np.float64,
    ) / (6.0 * signed_area)
    return abs(signed_area), centroid


def polygon_clearance_mm(first: np.ndarray, second: np.ndarray) -> float:
    """Return boundary clearance, or a negative value for overlap."""
    first32 = np.asarray(first, np.float32).reshape(-1, 2)
    second32 = np.asarray(second, np.float32).reshape(-1, 2)
    overlap_area, _overlap = cv2.intersectConvexConvex(first32, second32)
    if float(overlap_area) > 1.0e-5:
        return -float(overlap_area)
    distances = [
        abs(float(cv2.pointPolygonTest(second32, tuple(point), True)))
        for point in first32
    ]
    distances.extend(
        abs(float(cv2.pointPolygonTest(first32, tuple(point), True)))
        for point in second32
    )
    return min(distances) if distances else 0.0


def solve_assembly_target_offsets() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Find the smallest valid loose-fit translation of the four templates."""
    templates = {name: target for name, target in TARGET_PIECE_TEMPLATES}
    names = tuple(templates)
    adjacent = {frozenset(pair) for pair in ASSEMBLY_ADJACENT_PAIRS}
    for scale in np.linspace(0.5, 1.5, 101):
        offsets = {
            name: ASSEMBLY_OFFSET_DIRECTIONS[name] * float(scale)
            for name in names
        }
        clearances: dict[str, float] = {}
        vertex_gaps: dict[str, float] = {}
        valid = True
        for first_index, first_name in enumerate(names):
            for second_name in names[first_index + 1:]:
                pair_key = f"{first_name}|{second_name}"
                first_polygon = templates[first_name] + offsets[first_name]
                second_polygon = templates[second_name] + offsets[second_name]
                clearance = polygon_clearance_mm(first_polygon, second_polygon)
                clearances[pair_key] = clearance
                pair_is_adjacent = frozenset((first_name, second_name)) in adjacent
                # Every pair must have a visible positive seam, including the
                # only non-adjacent pair in the exact tiling.
                required_clearance = ASSEMBLY_MIN_CLEARANCE_MM
                if clearance + 1.0e-6 < required_clearance:
                    valid = False
                    break
                if pair_is_adjacent:
                    vertex_gap = float(
                        np.linalg.norm(offsets[first_name] - offsets[second_name])
                    )
                    vertex_gaps[pair_key] = vertex_gap
                    if vertex_gap > ASSEMBLY_MAX_ADJACENT_VERTEX_GAP_MM + 1.0e-6:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            continue
        return offsets, {
            "solver": "scaled_seam_normal_search_v1",
            "scale": round(float(scale), 3),
            "minimum_required_clearance_mm": ASSEMBLY_MIN_CLEARANCE_MM,
            "actual_minimum_clearance_mm": round(min(clearances.values()), 3),
            "maximum_allowed_adjacent_vertex_gap_mm": (
                ASSEMBLY_MAX_ADJACENT_VERTEX_GAP_MM
            ),
            "actual_maximum_adjacent_vertex_gap_mm": round(
                max(vertex_gaps.values()), 3
            ),
            "pair_clearances_mm": {
                key: round(value, 3) for key, value in clearances.items()
            },
            "adjacent_vertex_gaps_mm": {
                key: round(value, 3) for key, value in vertex_gaps.items()
            },
        }
    raise ValueError("NO_TARGET_LAYOUT_SATISFIES_CLEARANCE_AND_15MM_VERTEX_LIMIT")


@dataclass
class DetectorConfig:
    contrast: float = 28.0
    min_area_mm2: float = 100.0
    max_area_mm2: float = 4500.0
    stable_frames: int = 5
    stable_center_mm: float = 2.0
    paper_mode: str = "auto"
    divider_y_mm: float = 0.0
    source_region: str = "lower"


class PuzzleDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.center_history: deque[np.ndarray] = deque(
            maxlen=config.stable_frames
        )
        self.a4_corner_history: deque[np.ndarray] = deque(maxlen=15)
        self.divider_history: deque[float] = deque(maxlen=15)
        self.plan_history: deque[dict[str, Any]] = deque(maxlen=12)

    def stabilize_a4_corners(
        self, corners: np.ndarray
    ) -> tuple[np.ndarray, int, float, bool]:
        ordered = order_quad(corners).astype(np.float64)
        if self.a4_corner_history:
            reference = np.median(
                np.stack(tuple(self.a4_corner_history), axis=0), axis=0
            )
            jump = float(
                np.max(np.linalg.norm(ordered - reference, axis=1))
            )
            if jump > 24.0:
                self.a4_corner_history.clear()
        self.a4_corner_history.append(ordered)
        samples = np.stack(tuple(self.a4_corner_history), axis=0)
        median = np.median(samples, axis=0)
        spread = float(
            np.max(np.linalg.norm(samples - median[None, :, :], axis=2))
        )
        # At 1280x720 the dark-paper contour moves by roughly 3-5 source
        # pixels because of MJPEG/exposure noise.  The 15-frame median keeps
        # the rectified metric output below 1 mm, so 5 px is the appropriate
        # lock threshold at this capture resolution.
        stable = len(samples) >= 8 and spread <= 5.0
        return median.astype(np.float32), len(samples), spread, stable

    def stabilize_divider(self, divider_y: int) -> tuple[int, int, float]:
        value = float(divider_y)
        if self.divider_history:
            reference = float(np.median(tuple(self.divider_history)))
            if abs(value - reference) > 20.0:
                self.divider_history.clear()
        self.divider_history.append(value)
        samples = np.asarray(tuple(self.divider_history), dtype=np.float64)
        median = float(np.median(samples))
        spread = float(np.max(np.abs(samples - median)))
        return int(round(median)), len(samples), spread

    def smooth_assembly_plan(
        self, plan: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not plan or not plan.get("moves"):
            self.plan_history.clear()
            return plan
        signature = tuple(
            sorted(
                (move["target_piece"], move["piece_id"])
                for move in plan["moves"]
            )
        )
        if self.plan_history and self.plan_history[-1]["signature"] != signature:
            self.plan_history.clear()
        sample = {
            "signature": signature,
            "moves": {
                move["target_piece"]: {
                    "current": np.asarray(
                        move["current_centroid_mm"], dtype=np.float64
                    ),
                    "angle": float(move["rotate_deg_clockwise"]),
                }
                for move in plan["moves"]
            },
        }
        self.plan_history.append(sample)
        position_spread = 0.0
        angle_spread = 0.0
        for move in plan["moves"]:
            target_name = move["target_piece"]
            centers = np.stack(
                [entry["moves"][target_name]["current"] for entry in self.plan_history],
                axis=0,
            )
            center = np.median(centers, axis=0)
            position_spread = max(
                position_spread,
                float(np.max(np.linalg.norm(centers - center[None, :], axis=1))),
            )
            angles = np.asarray(
                [entry["moves"][target_name]["angle"] for entry in self.plan_history],
                dtype=np.float64,
            )
            reference = float(angles[-1])
            unwrapped = reference + (angles - reference + 180.0) % 360.0 - 180.0
            angle = float(np.median(unwrapped))
            angle = (angle + 180.0) % 360.0 - 180.0
            differences = (angles - angle + 180.0) % 360.0 - 180.0
            angle_spread = max(
                angle_spread, float(np.max(np.abs(differences)))
            )
            target_center = np.asarray(
                move["target_centroid_mm"], dtype=np.float64
            )
            move["current_centroid_mm"] = np.round(center, 2).tolist()
            move["translation_mm"] = np.round(
                target_center - center, 2
            ).tolist()
            move["rotate_deg_clockwise"] = round(angle, 2)
        sample_count = len(self.plan_history)
        measurement_stable = (
            sample_count >= 8
            and position_spread <= 0.8
            and angle_spread <= 1.5
        )
        plan["measurement_samples"] = sample_count
        plan["position_spread_mm"] = round(position_spread, 3)
        plan["angle_spread_deg"] = round(angle_spread, 3)
        plan["measurement_stable"] = measurement_stable
        return plan

    @staticmethod
    def find_a4(frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _threshold, bright = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            5,
        )
        masks = [
            bright,
            cv2.bitwise_not(bright),
            adaptive,
            cv2.bitwise_not(adaptive),
        ]
        contours: list[np.ndarray] = []
        for mask in masks:
            raw_found, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(raw_found)
            # The contrasting divider physically cuts a thresholded black
            # sheet into two regions.  Try increasingly wide closing kernels
            # so the outer A4 contour survives while the divider is still
            # detected later on the unmodified rectified image.
            for close_size in (9, 15, 31, 47):
                close_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (close_size, close_size)
                )
                closed = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, close_kernel
                )
                found, _ = cv2.findContours(
                    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                contours.extend(found)
        frame_area = float(frame.shape[0] * frame.shape[1])
        # A full-width divider may keep the two paper regions disconnected.
        # Merge pairs of the largest non-frame contours using their convex
        # hull; the resulting hull supplies the four physical outer corners.
        ranked_mergeable = sorted(
            (
                contour
                for contour in contours
                if frame_area * 0.025
                <= cv2.contourArea(contour)
                <= frame_area * 0.75
            ),
            key=cv2.contourArea,
            reverse=True,
        )
        mergeable: list[np.ndarray] = []
        mergeable_boxes: list[tuple[int, int, int, int]] = []
        for contour in ranked_mergeable:
            box = cv2.boundingRect(contour)
            box_x, box_y, box_w, box_h = box
            duplicate = False
            for other_x, other_y, other_w, other_h in mergeable_boxes:
                intersection_w = max(
                    0,
                    min(box_x + box_w, other_x + other_w)
                    - max(box_x, other_x),
                )
                intersection_h = max(
                    0,
                    min(box_y + box_h, other_y + other_h)
                    - max(box_y, other_y),
                )
                intersection = float(intersection_w * intersection_h)
                union = float(
                    box_w * box_h
                    + other_w * other_h
                    - intersection
                )
                if union > 0.0 and intersection / union >= 0.78:
                    duplicate = True
                    break
            if duplicate:
                continue
            mergeable.append(contour)
            mergeable_boxes.append(box)
            if len(mergeable) >= 12:
                break
        for first_index, first in enumerate(mergeable):
            for second in mergeable[first_index + 1:]:
                hull = cv2.convexHull(np.vstack((first, second)))
                hull_area = float(cv2.contourArea(hull))
                if frame_area * 0.12 <= hull_area <= frame_area * 0.98:
                    contours.append(hull)
        best: np.ndarray | None = None
        best_score = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < frame_area * 0.12 or area > frame_area * 0.98:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = None
            # A close oblique view can make the A4 look like a pronounced
            # trapezoid.  Try several simplification strengths rather than
            # requiring one particular contour approximation.
            for epsilon in (0.015, 0.02, 0.025, 0.035, 0.05):
                candidate = cv2.approxPolyDP(
                    contour, epsilon * perimeter, True
                )
                if len(candidate) == 4 and cv2.isContourConvex(candidate):
                    polygon = candidate
                    break
            if polygon is None:
                continue
            quad = order_quad(polygon.reshape(4, 2))
            # A valid metric calibration requires all four physical paper
            # corners to be visible.  Reject contours clipped by the image
            # boundary instead of treating the frame edges as A4 edges.
            border = max(4, int(round(min(frame.shape[:2]) * 0.012)))
            if (
                np.any(quad[:, 0] < border)
                or np.any(quad[:, 0] > frame.shape[1] - 1 - border)
                or np.any(quad[:, 1] < border)
                or np.any(quad[:, 1] > frame.shape[0] - 1 - border)
            ):
                continue
            top = float(np.linalg.norm(quad[1] - quad[0]))
            bottom = float(np.linalg.norm(quad[2] - quad[3]))
            left = float(np.linalg.norm(quad[3] - quad[0]))
            right = float(np.linalg.norm(quad[2] - quad[1]))
            edges = np.asarray([top, right, bottom, left], dtype=np.float32)
            if float(np.min(edges)) < 35.0:
                continue
            opposite_balance = min(
                min(top, bottom) / max(top, bottom),
                min(left, right) / max(left, right),
            )
            if opposite_balance < 0.28:
                continue
            width = 0.5 * (top + bottom)
            height = 0.5 * (left + right)
            if min(width, height) < 40.0:
                continue
            ratio = max(width, height) / min(width, height)
            ratio_error = abs(ratio - (A4_HEIGHT_MM / A4_WIDTH_MM))
            rectangularity = area / max(1.0, width * height)
            # Edge-length ratio is only a soft A4 cue before rectification;
            # perspective projection does not preserve it.
            if not 0.72 <= ratio <= 2.65 or rectangularity < 0.48:
                continue

            edge_contrasts: list[float] = []
            valid_edge_samples = True
            for edge_index in range(4):
                start = quad[edge_index].astype(np.float64)
                end = quad[(edge_index + 1) % 4].astype(np.float64)
                vector = end - start
                length = float(np.linalg.norm(vector))
                if length < 1.0:
                    valid_edge_samples = False
                    break
                inward = np.asarray(
                    [-vector[1], vector[0]], dtype=np.float64
                ) / length
                inner_values: list[float] = []
                outer_values: list[float] = []
                for fraction in np.linspace(0.18, 0.82, 7):
                    edge_point = start + fraction * vector
                    outward = -inward
                    boundary_distances: list[float] = []
                    if outward[0] > 1.0e-6:
                        boundary_distances.append(
                            (frame.shape[1] - 1.0 - edge_point[0]) / outward[0]
                        )
                    elif outward[0] < -1.0e-6:
                        boundary_distances.append(
                            edge_point[0] / -outward[0]
                        )
                    if outward[1] > 1.0e-6:
                        boundary_distances.append(
                            (frame.shape[0] - 1.0 - edge_point[1]) / outward[1]
                        )
                    elif outward[1] < -1.0e-6:
                        boundary_distances.append(
                            edge_point[1] / -outward[1]
                        )
                    available = min(boundary_distances) if boundary_distances else 0.0
                    outer_offset = min(42.0, available - 2.0)
                    if outer_offset < 6.0:
                        valid_edge_samples = False
                        break
                    inner_point = edge_point + 12.0 * inward
                    outer_point = edge_point + outer_offset * outward
                    inner_x = int(
                        np.clip(round(inner_point[0]), 0, frame.shape[1] - 1)
                    )
                    inner_y = int(
                        np.clip(round(inner_point[1]), 0, frame.shape[0] - 1)
                    )
                    outer_x = int(
                        np.clip(round(outer_point[0]), 0, frame.shape[1] - 1)
                    )
                    outer_y = int(
                        np.clip(round(outer_point[1]), 0, frame.shape[0] - 1)
                    )
                    inner_values.append(float(blurred[inner_y, inner_x]))
                    outer_values.append(float(blurred[outer_y, outer_x]))
                if not valid_edge_samples:
                    break
                edge_contrasts.append(
                    float(np.median(outer_values) - np.median(inner_values))
                )
            if not valid_edge_samples or len(edge_contrasts) != 4:
                continue
            black_paper = float(np.median(edge_contrasts)) > 0.0
            signed_contrasts = (
                np.asarray(edge_contrasts)
                if black_paper
                else -np.asarray(edge_contrasts)
            )
            # One physical edge can be partly hidden by the gantry/cable in the
            # fixed overhead installation.  Require three strong outer edges;
            # an internal divider still has paper on both sides and cannot pass
            # the remaining A4 size/shape/orientation checks.
            if int(np.count_nonzero(signed_contrasts >= 8.0)) < 3:
                continue
            edge_contrast_score = min(
                2.0, float(np.median(signed_contrasts)) / 45.0
            )
            score = (
                (area / frame_area)
                * rectangularity
                * opposite_balance
                * edge_contrast_score
                / (1.0 + 0.35 * ratio_error)
            )
            if score > best_score:
                best = quad
                best_score = score
        return best, best_score

    @staticmethod
    def orient_a4_portrait(corners: np.ndarray) -> np.ndarray:
        source = order_quad(corners)
        top = float(np.linalg.norm(source[1] - source[0]))
        bottom = float(np.linalg.norm(source[2] - source[3]))
        left = float(np.linalg.norm(source[3] - source[0]))
        right = float(np.linalg.norm(source[2] - source[1]))
        if 0.5 * (top + bottom) > 0.5 * (left + right):
            # Landscape camera view: rotate clockwise into canonical portrait
            # coordinates.  This maps the source left-hand start area to the
            # canonical upper area used by the piece detector.
            source = np.roll(source, 1, axis=0)
        return source.astype(np.float32)

    @staticmethod
    def rectify(
        frame: np.ndarray, corners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        destination = np.asarray(
            [
                [0.0, 0.0],
                [WARP_WIDTH - 1.0, 0.0],
                [WARP_WIDTH - 1.0, WARP_HEIGHT - 1.0],
                [0.0, WARP_HEIGHT - 1.0],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(frame, matrix, (WARP_WIDTH, WARP_HEIGHT))
        return warped, matrix

    def choose_a4_orientation(
        self, frame: np.ndarray, corners: np.ndarray
    ) -> tuple[np.ndarray, str, float]:
        source = order_quad(corners)
        top = float(np.linalg.norm(source[1] - source[0]))
        bottom = float(np.linalg.norm(source[2] - source[3]))
        left = float(np.linalg.norm(source[3] - source[0]))
        right = float(np.linalg.norm(source[2] - source[1]))
        landscape = 0.5 * (top + bottom) > 0.5 * (left + right)
        if landscape:
            candidates = (
                (np.roll(source, 1, axis=0), "landscape_left_to_top"),
                (np.roll(source, -1, axis=0), "landscape_right_to_top"),
            )
        else:
            candidates = (
                (source, "portrait_as_seen"),
                (np.roll(source, 2, axis=0), "portrait_rotated_180"),
            )

        best_corners = candidates[0][0]
        best_name = candidates[0][1]
        best_score = -1.0e9
        margin = int(round(5.0 * WARP_PX_PER_MM))
        for candidate, name in candidates:
            warped, _matrix = self.rectify(frame, candidate)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            inner = gray[margin:-margin, margin:-margin]
            background = float(np.median(inner))
            contrast = max(18.0, 0.65 * self.config.contrast)
            if background < 128.0:
                foreground = inner > background + contrast
            else:
                foreground = inner < background - contrast
            row_coverage = np.mean(foreground, axis=1)
            foreground[row_coverage > 0.32, :] = False
            positions = np.argwhere(foreground)
            if len(positions) < 100:
                score = -1.0e6
            else:
                mean_y = float(np.mean(positions[:, 0])) / max(
                    1.0, float(inner.shape[0] - 1)
                )
                coverage = min(
                    0.2, float(len(positions)) / float(inner.size)
                )
                score = (
                    mean_y
                    if self.config.source_region == "lower"
                    else (1.0 - mean_y)
                ) + coverage
            if score > best_score:
                best_corners = candidate
                best_name = name
                best_score = score
        return best_corners.astype(np.float32), best_name, best_score

    @staticmethod
    def find_divider(gray: np.ndarray) -> tuple[int | None, float]:
        x0 = int(gray.shape[1] * 0.05)
        x1 = int(gray.shape[1] * 0.95)
        y0 = int(gray.shape[0] * 0.08)
        y1 = int(gray.shape[0] * 0.92)
        search = gray[y0:y1, x0:x1]
        background = float(np.median(search))
        dark_fraction = np.mean(search < background - 35.0, axis=1)
        bright_fraction = np.mean(search > background + 35.0, axis=1)
        line_fraction = np.maximum(dark_fraction, bright_fraction)
        index = int(np.argmax(line_fraction))
        score = float(line_fraction[index])
        if score < 0.45:
            return None, score
        return y0 + index, score

    def segment_pieces(
        self, warped: np.ndarray, divider_y: int
    ) -> tuple[np.ndarray, float, float, str]:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        margin = int(4.0 * WARP_PX_PER_MM)
        divider_margin = int(3.0 * WARP_PX_PER_MM)
        if self.config.source_region == "lower":
            region_start = min(
                gray.shape[0] - 20, divider_y + divider_margin
            )
            region_end = gray.shape[0]
        else:
            region_start = 0
            region_end = max(20, divider_y - divider_margin)
        roi = gray[region_start:region_end]
        inner = roi[
            margin:max(margin + 1, roi.shape[0] - margin),
            margin:max(margin + 1, roi.shape[1] - margin),
        ]
        median_luma = float(np.median(inner)) if inner.size else 240.0
        mode = self.config.paper_mode
        if mode == "auto":
            mode = "black" if median_luma < 128.0 else "white"
        if mode == "black":
            paper_luma = float(np.percentile(inner, 20.0)) if inner.size else 20.0
            threshold = max(10.0, min(235.0, paper_luma + self.config.contrast))
            foreground = roi > threshold
        else:
            paper_luma = float(np.percentile(inner, 88.0)) if inner.size else 240.0
            threshold = max(40.0, min(245.0, paper_luma - self.config.contrast))
            foreground = roi < threshold
        mask = np.zeros_like(gray)
        mask[region_start:region_end] = np.where(
            foreground, 255, 0
        ).astype(np.uint8)
        mask[:margin] = 0
        mask[-margin:] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask, paper_luma, threshold, mode

    def extract_pieces(
        self, mask: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[tuple[float, dict[str, Any], np.ndarray]] = []
        scale2 = WARP_PX_PER_MM * WARP_PX_PER_MM
        for contour in contours:
            area_px = float(cv2.contourArea(contour))
            area_mm2 = area_px / scale2
            if not self.config.min_area_mm2 <= area_mm2 <= self.config.max_area_mm2:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter < 20.0:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area_px / max(1.0, hull_area)
            if solidity < 0.62:
                continue
            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1.0:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            epsilon = max(3.0, 0.015 * perimeter)
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            rectangle = cv2.minAreaRect(contour)
            size_mm = sorted(
                [
                    float(rectangle[1][0]) / WARP_PX_PER_MM,
                    float(rectangle[1][1]) / WARP_PX_PER_MM,
                ],
                reverse=True,
            )
            item = {
                "center_mm": [
                    round(cx / WARP_PX_PER_MM, 1),
                    round(cy / WARP_PX_PER_MM, 1),
                ],
                "angle_deg": round(pca_angle_deg(contour), 1),
                "area_mm2": round(area_mm2, 1),
                "size_mm": [round(size_mm[0], 1), round(size_mm[1], 1)],
                "solidity": round(solidity, 3),
                "vertex_count": int(len(polygon)),
                "vertices_mm": [
                    [
                        round(float(point[0]) / WARP_PX_PER_MM, 1),
                        round(float(point[1]) / WARP_PX_PER_MM, 1),
                    ]
                    for point in polygon
                ],
            }
            candidates.append((area_mm2, item, contour))
        candidates.sort(key=lambda value: value[0], reverse=True)
        candidates = candidates[:4]
        pieces: list[dict[str, Any]] = []
        selected: list[np.ndarray] = []
        for index, (_area, item, contour) in enumerate(candidates, start=1):
            item["id"] = f"P{index}"
            pieces.append(item)
            selected.append(contour)
        return pieces, selected

    @staticmethod
    def best_vertex_fit(
        contour: np.ndarray, target: np.ndarray
    ) -> dict[str, Any] | None:
        target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
        vertex_count = len(target)
        perimeter = float(cv2.arcLength(contour, True))
        candidates: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for fraction in np.linspace(0.004, 0.12, 90):
            polygon = cv2.approxPolyDP(
                contour, float(fraction) * perimeter, True
            ).reshape(-1, 2)
            if len(polygon) != vertex_count:
                continue
            polygon_mm = polygon.astype(np.float64) / WARP_PX_PER_MM
            key = tuple(np.round(polygon_mm.reshape(-1), 2))
            if key not in seen:
                seen.add(key)
                candidates.append(polygon_mm)
        if not candidates:
            return None

        best: dict[str, Any] | None = None
        for source in candidates:
            source_center = np.mean(source, axis=0)
            centered_source = source - source_center
            denominator = float(np.sum(centered_source * centered_source))
            if denominator < 1.0e-9:
                continue
            for reverse in (False, True):
                base = target[::-1] if reverse else target
                for shift in range(vertex_count):
                    ordered_target = np.roll(base, shift, axis=0)
                    target_center = np.mean(ordered_target, axis=0)
                    centered_target = ordered_target - target_center
                    covariance = centered_source.T @ centered_target
                    left, _values, right_t = np.linalg.svd(covariance)
                    rotation = left @ right_t
                    if np.linalg.det(rotation) < 0.0:
                        adjusted = left.copy()
                        adjusted[:, -1] *= -1.0
                        rotation = adjusted @ right_t
                    rotated = centered_source @ rotation
                    scale = float(
                        np.sum(rotated * centered_target) / denominator
                    )
                    if scale <= 0.0:
                        continue
                    fitted = scale * rotated
                    target_radius = math.sqrt(
                        max(
                            1.0e-9,
                            float(
                                np.mean(
                                    np.sum(centered_target ** 2, axis=1)
                                )
                            ),
                        )
                    )
                    residual = math.sqrt(
                        float(np.mean(np.sum((fitted - centered_target) ** 2, axis=1)))
                    ) / target_radius
                    angle = math.degrees(
                        math.atan2(
                            float(rotation[0, 1]),
                            float(rotation[0, 0]),
                        )
                    )
                    while angle <= -180.0:
                        angle += 360.0
                    while angle > 180.0:
                        angle -= 360.0
                    result = {
                        "residual": residual,
                        "rotation_deg_clockwise": angle,
                        "scale_to_template": scale,
                        "source_vertices_mm": source,
                    }
                    if best is None or residual < best["residual"]:
                        best = result
        return best

    def build_assembly_plan(
        self,
        pieces: list[dict[str, Any]],
        contours: list[np.ndarray],
        divider_y_mm: float,
    ) -> dict[str, Any] | None:
        if len(pieces) != 4 or len(contours) != 4:
            return None
        target_gap_mm = ASSEMBLY_DIVIDER_GAP_MM
        try:
            target_offsets, target_constraints = solve_assembly_target_offsets()
        except ValueError as exc:
            return {"ready": False, "error": str(exc)}
        expanded_templates = [
            target
            + target_offsets[name]
            for name, target in TARGET_PIECE_TEMPLATES
        ]
        layout_points = np.concatenate(expanded_templates, axis=0)
        layout_min = np.min(layout_points, axis=0)
        layout_max = np.max(layout_points, axis=0)
        layout_size = layout_max - layout_min
        layout_width_mm = float(layout_size[0])
        layout_height_mm = float(layout_size[1])
        relative_target_centers = []
        for name, target in TARGET_PIECE_TEMPLATES:
            _target_area, target_centroid = polygon_area_centroid(target)
            relative_target_centers.append(
                target_centroid
                + target_offsets[name]
                - layout_min
            )
        minimum_target_origin_y = max(0.0, ASSEMBLY_STAGE_MIN_CENTER_Y_MM - min(
            float(center[1]) for center in relative_target_centers
        ))
        maximum_target_origin_y = min(
            A4_HEIGHT_MM - layout_height_mm,
            ASSEMBLY_STAGE_MAX_CENTER_Y_MM - max(
                float(center[1]) for center in relative_target_centers
            ),
        )
        if self.config.source_region == "lower":
            if divider_y_mm < layout_height_mm + 10.0:
                return {
                    "ready": False,
                    "error": "UPPER_REGION_TOO_SMALL_FOR_SPACED_TARGET",
                }
            target_y = divider_y_mm - target_gap_mm - layout_height_mm
        else:
            lower_height = A4_HEIGHT_MM - divider_y_mm
            if lower_height < layout_height_mm + 10.0:
                return {
                    "ready": False,
                    "error": "LOWER_REGION_TOO_SMALL_FOR_SPACED_TARGET",
                }
            target_y = divider_y_mm + target_gap_mm
        # A fluctuating divider estimate must not place the top piece's magnet
        # centre above the slide's calibrated Y minimum.  Shift the complete
        # aligned layout together; relative seams and the outer outline remain
        # unchanged.
        target_y = min(
            max(target_y, minimum_target_origin_y), maximum_target_origin_y
        )
        if self.config.source_region == "lower":
            if target_y + layout_height_mm >= divider_y_mm:
                return {
                    "ready": False,
                    "error": "NO_REACHABLE_SPACED_LAYOUT_ABOVE_DIVIDER",
                }
        elif target_y <= divider_y_mm:
            return {
                "ready": False,
                "error": "NO_REACHABLE_SPACED_LAYOUT_BELOW_DIVIDER",
            }

        # Place against the destination side of the divider to minimize Y
        # travel on the limited 225 mm slide.
        origin = np.asarray(
            [
                0.5 * (A4_WIDTH_MM - layout_width_mm),
                target_y,
            ],
            dtype=np.float64,
        )
        source_areas = np.asarray(
            [float(piece["area_mm2"]) for piece in pieces],
            dtype=np.float64,
        )
        target_areas = np.asarray(
            [polygon_area_centroid(shape)[0] for _name, shape in TARGET_PIECE_TEMPLATES],
            dtype=np.float64,
        )
        source_fractions = source_areas / max(1.0, float(np.sum(source_areas)))
        target_fractions = target_areas / float(np.sum(target_areas))

        fit_matrix: list[list[dict[str, Any]]] = []
        for source_index, contour in enumerate(contours):
            row: list[dict[str, Any]] = []
            for target_index, (_name, target) in enumerate(
                TARGET_PIECE_TEMPLATES
            ):
                fit = self.best_vertex_fit(contour, target)
                if fit is None:
                    row.append({"cost": 1.0e6, "residual": 1.0e6})
                    continue
                area_cost = abs(
                    math.log(
                        max(1.0e-6, source_fractions[source_index])
                        / max(1.0e-6, target_fractions[target_index])
                    )
                )
                fit["area_fraction_error"] = area_cost
                fit["cost"] = 2.5 * float(fit["residual"]) + 0.8 * area_cost
                row.append(fit)
            fit_matrix.append(row)

        best_assignment: tuple[int, ...] | None = None
        best_cost = float("inf")
        for assignment in permutations(range(4)):
            cost = sum(
                float(fit_matrix[index][target_index]["cost"])
                for index, target_index in enumerate(assignment)
            )
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
        # A missing vertex fit is represented by a large finite sentinel so it
        # can live in the assignment matrix.  Never let an assignment that
        # contains such a sentinel reach the move builder: it has no rotation
        # field and, more importantly, is not a valid geometric solution.
        if (
            best_assignment is None
            or not math.isfinite(best_cost)
            or best_cost >= 1.0e5
        ):
            return {"ready": False, "error": "PIECE_MATCH_FAILED"}

        placement_priority = {
            "G1_BIG_TRIANGLE": 1,
            "G4_BOTTOM": 2,
            # Place the middle P3 support before the small P4 piece.  This
            # avoids approaching the already placed small piece from above.
            "G3_MIDDLE": 3,
            "G2_SMALL_TOP_LEFT": 4,
        }
        moves: list[dict[str, Any]] = []
        for source_index, target_index in enumerate(best_assignment):
            name, target_local = TARGET_PIECE_TEMPLATES[target_index]
            fit = fit_matrix[source_index][target_index]
            _area, local_centroid = polygon_area_centroid(target_local)
            # Preserve every solved angle.  The offset solver has already
            # verified positive clearance and the 15 mm corresponding-vertex
            # limit for every adjacent target pair.
            target_offset = target_offsets[name]
            layout_origin = origin - layout_min
            target_centroid = layout_origin + local_centroid + target_offset
            current_centroid = np.asarray(
                pieces[source_index]["center_mm"], dtype=np.float64
            )
            target_vertices = target_local + layout_origin + target_offset
            moves.append(
                {
                    "order": placement_priority[name],
                    "piece_id": pieces[source_index]["id"],
                    "target_piece": name,
                    "current_centroid_mm": np.round(
                        current_centroid, 1
                    ).tolist(),
                    "target_centroid_mm": np.round(
                        target_centroid, 1
                    ).tolist(),
                    "translation_mm": np.round(
                        target_centroid - current_centroid, 1
                    ).tolist(),
                    "rotate_deg_clockwise": round(
                        float(fit["rotation_deg_clockwise"]), 1
                    ),
                    "target_vertices_mm": np.round(
                        target_vertices, 1
                    ).tolist(),
                    "shape_residual": round(float(fit["residual"]), 3),
                    "area_fraction_error": round(
                        float(fit["area_fraction_error"]), 3
                    ),
                }
            )
        moves.sort(key=lambda move: int(move["order"]))
        unreachable_target_centers = []
        for move in moves:
            target_x, target_y_value = (
                float(value) for value in move["target_centroid_mm"]
            )
            if not (
                ASSEMBLY_STAGE_MIN_CENTER_X_MM
                <= target_x
                <= ASSEMBLY_STAGE_MAX_CENTER_X_MM
                and ASSEMBLY_STAGE_MIN_CENTER_Y_MM
                <= target_y_value
                <= ASSEMBLY_STAGE_MAX_CENTER_Y_MM
            ):
                unreachable_target_centers.append(
                    {
                        "piece_id": move["piece_id"],
                        "target_centroid_mm": move["target_centroid_mm"],
                    }
                )
        target_constraints["centroids_reachable"] = not unreachable_target_centers
        target_constraints["unreachable_target_centers"] = (
            unreachable_target_centers
        )
        plan_ready = best_cost < 3.0 and not unreachable_target_centers
        return {
            "ready": plan_ready,
            "error": (
                None
                if plan_ready
                else (
                    "TARGET_CENTROID_UNREACHABLE"
                    if unreachable_target_centers
                    else "PIECE_MATCH_QUALITY_FAILED"
                )
            ),
            "target_rectangle_mm": {
                "origin": np.round(origin, 1).tolist(),
                "size": np.round(layout_size, 1).tolist(),
                "center": np.round(
                    origin + 0.5 * layout_size,
                    1,
                ).tolist(),
            },
            "rotation_convention": "positive is clockwise in A4 image",
            "divider_clearance_mm": ASSEMBLY_DIVIDER_GAP_MM,
            "piece_spacing_mm": ASSEMBLY_PIECE_SPACING_MM,
            "target_constraints": target_constraints,
            "assignment_cost": round(best_cost, 3),
            "moves": moves,
        }

    def stable(self, pieces: list[dict[str, Any]]) -> bool:
        if len(pieces) != 4:
            self.center_history.clear()
            return False
        centers = np.asarray(
            sorted((piece["center_mm"] for piece in pieces), key=lambda p: p[0]),
            dtype=np.float64,
        )
        self.center_history.append(centers)
        if len(self.center_history) < self.config.stable_frames:
            return False
        samples = np.stack(tuple(self.center_history), axis=0)
        spread = np.max(np.linalg.norm(samples - np.median(samples, axis=0), axis=2))
        return bool(spread <= self.config.stable_center_mm)

    @staticmethod
    def piece_set_quality(pieces: list[dict[str, Any]]) -> dict[str, Any]:
        """Reject four stable reflections/noise blobs before motion planning."""
        areas = [float(piece.get("area_mm2", 0.0)) for piece in pieces]
        total_area = sum(areas)
        problems: list[str] = []
        if len(pieces) != 4:
            problems.append("NEED_EXACTLY_FOUR_CONTOURS")
        if not 4320.0 <= total_area <= 7800.0:
            problems.append("TOTAL_AREA_NOT_NEAR_100X60_TEMPLATE")
        for piece in pieces:
            piece_id = str(piece.get("id", "?"))
            if not 250.0 <= float(piece.get("area_mm2", 0.0)) <= 3300.0:
                problems.append(piece_id + "_AREA_OUTLIER")
            if float(piece.get("solidity", 0.0)) < 0.82:
                problems.append(piece_id + "_LOW_SOLIDITY")
            if not 3 <= int(piece.get("vertex_count", 0)) <= 7:
                problems.append(piece_id + "_VERTEX_COUNT_OUTLIER")
        return {
            "valid": not problems,
            "total_area_mm2": round(total_area, 1),
            "expected_total_area_mm2": 6000.0,
            "problems": problems,
        }

    def detect(
        self, frame: np.ndarray, manual_corners: np.ndarray | None
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        original = frame.copy()
        auto_score = 0.0
        source = "manual"
        a4_corner_samples = 1
        a4_corner_spread_px = 0.0
        a4_corner_stable = manual_corners is not None
        if manual_corners is None:
            corners, auto_score = self.find_a4(frame)
            source = "auto"
            if corners is not None:
                (
                    corners,
                    a4_corner_samples,
                    a4_corner_spread_px,
                    a4_corner_stable,
                ) = self.stabilize_a4_corners(corners)
        else:
            # Locked/manual corners are already stored in physical A4 order:
            # A4 top-left, top-right, bottom-right, bottom-left.  Reordering
            # them from screen geometry would lose a 90/180 degree rotation.
            corners = np.asarray(manual_corners, dtype=np.float32).reshape(4, 2)
        if corners is None:
            self.center_history.clear()
            cv2.putText(
                original, "A4 NOT FOUND - CLICK 4 CORNERS", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            blank = np.full((WARP_HEIGHT, WARP_WIDTH, 3), 245, np.uint8)
            return (
                {
                    "a4_found": False,
                    "a4_source": source,
                    "a4_score": round(auto_score, 3),
                    "a4_corner_samples": len(self.a4_corner_history),
                    "a4_corner_stable": False,
                    "piece_count": 0,
                    "stable_four_pieces": False,
                    "pieces": [],
                },
                original,
                blank,
                cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY),
            )
        if manual_corners is None:
            (
                corners,
                a4_orientation,
                a4_orientation_score,
            ) = self.choose_a4_orientation(frame, corners)
        else:
            a4_orientation = "locked_physical_a4"
            a4_orientation_score = 1.0
        cv2.polylines(original, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        for index, point in enumerate(corners.astype(np.int32), start=1):
            cv2.circle(original, tuple(point), 7, (0, 0, 255), -1)
            cv2.putText(
                original, str(index), tuple(point + np.asarray([8, -8])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
        warped, _matrix = self.rectify(frame, corners)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        if self.config.divider_y_mm > 0.0:
            divider_y: int | None = int(
                round(self.config.divider_y_mm * WARP_PX_PER_MM)
            )
            divider_score = 1.0
            divider_source = "manual_mm"
            divider_samples = 1
            divider_spread_px = 0.0
        else:
            divider_y, divider_score = self.find_divider(gray)
            divider_source = "auto"
            if divider_y is None:
                divider_samples = len(self.divider_history)
                divider_spread_px = 0.0
            else:
                (
                    divider_y,
                    divider_samples,
                    divider_spread_px,
                ) = self.stabilize_divider(divider_y)
        divider_found = divider_y is not None
        annotated = warped.copy()
        if divider_found:
            mask, paper_luma, piece_threshold, paper_mode = self.segment_pieces(
                warped, int(divider_y)
            )
            pieces, contours = self.extract_pieces(mask)
            piece_quality = self.piece_set_quality(pieces)
            if piece_quality["valid"]:
                is_stable = self.stable(pieces)
            else:
                self.center_history.clear()
                is_stable = False
            cv2.line(
                annotated,
                (0, int(divider_y)),
                (WARP_WIDTH - 1, int(divider_y)),
                (255, 0, 255),
                3,
            )
        else:
            mask = np.zeros_like(gray)
            pieces = []
            contours = []
            is_stable = False
            piece_quality = self.piece_set_quality(pieces)
            self.center_history.clear()
            paper_luma = float(np.median(gray))
            piece_threshold = None
            paper_mode = self.config.paper_mode
            if paper_mode == "auto":
                paper_mode = "black" if paper_luma < 128.0 else "white"
            cv2.putText(
                annotated,
                "DIVIDER NOT FOUND - SET Y(mm) OR IMPROVE LINE",
                (12, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 255),
                2,
            )
        assembly_plan = (
            self.build_assembly_plan(
                pieces, contours, float(divider_y) / WARP_PX_PER_MM
            )
            if divider_y is not None
            else None
        )
        solver_ready = bool(
            assembly_plan is not None
            and assembly_plan.get("ready", False)
        )
        assembly_plan = self.smooth_assembly_plan(assembly_plan)
        if assembly_plan is not None:
            assembly_plan["input_stable"] = is_stable
            assembly_plan["ready"] = bool(
                solver_ready
                and is_stable
                and assembly_plan.get("measurement_stable", False)
            )
        for piece, contour in zip(pieces, contours):
            cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 3)
            center = tuple(
                int(round(value * WARP_PX_PER_MM))
                for value in piece["center_mm"]
            )
            cv2.circle(annotated, center, 6, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"{piece['id']} ({piece['center_mm'][0]:.0f},"
                f"{piece['center_mm'][1]:.0f})mm {piece['angle_deg']:.0f}deg",
                (center[0] + 9, center[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 40, 230), 2,
            )
        if assembly_plan and assembly_plan.get("target_rectangle_mm"):
            colors = (
                (255, 180, 0),
                (255, 0, 180),
                (0, 210, 255),
                (180, 255, 0),
            )
            rectangle = assembly_plan["target_rectangle_mm"]
            rectangle_origin = np.asarray(rectangle["origin"], np.float64)
            rectangle_size = np.asarray(rectangle["size"], np.float64)
            top_left = tuple(
                np.rint(rectangle_origin * WARP_PX_PER_MM).astype(int)
            )
            bottom_right = tuple(
                np.rint(
                    (rectangle_origin + rectangle_size) * WARP_PX_PER_MM
                ).astype(int)
            )
            cv2.rectangle(
                annotated, top_left, bottom_right, (255, 255, 255), 3
            )
            for move in assembly_plan["moves"]:
                color = colors[(int(move["order"]) - 1) % len(colors)]
                target_polygon = np.rint(
                    np.asarray(move["target_vertices_mm"], np.float64)
                    * WARP_PX_PER_MM
                ).astype(np.int32)
                cv2.polylines(
                    annotated, [target_polygon], True, color, 4
                )
                target_center = tuple(
                    np.rint(
                        np.asarray(move["target_centroid_mm"], np.float64)
                        * WARP_PX_PER_MM
                    ).astype(int)
                )
                current_center = tuple(
                    np.rint(
                        np.asarray(move["current_centroid_mm"], np.float64)
                        * WARP_PX_PER_MM
                    ).astype(int)
                )
                cv2.arrowedLine(
                    annotated,
                    current_center,
                    target_center,
                    color,
                    2,
                    tipLength=0.035,
                )
                cv2.putText(
                    annotated,
                    f"{move['order']}:{move['piece_id']} "
                    f"R{move['rotate_deg_clockwise']:+.0f}",
                    (target_center[0] - 55, target_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        state_color = (0, 170, 0) if is_stable else (0, 140, 255)
        cv2.putText(
            annotated,
            f"PIECES={len(pieces)} STABLE={int(is_stable)}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_color, 2,
        )
        status = {
            "a4_found": True,
            "a4_source": source,
            "a4_score": round(auto_score, 3),
            "a4_corners_px": np.rint(corners).astype(int).tolist(),
            "a4_corner_samples": a4_corner_samples,
            "a4_corner_spread_px": round(a4_corner_spread_px, 3),
            "a4_corner_stable": a4_corner_stable,
            "a4_orientation": a4_orientation,
            "a4_orientation_score": round(a4_orientation_score, 3),
            "a4_size_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
            "divider_found": divider_found,
            "divider_source": divider_source,
            "divider_y_mm": (
                None
                if divider_y is None
                else round(divider_y / WARP_PX_PER_MM, 1)
            ),
            "divider_score": round(divider_score, 3),
            "divider_samples": divider_samples,
            "divider_spread_px": round(divider_spread_px, 3),
            "paper_luma": round(paper_luma, 1),
            "paper_mode_used": paper_mode,
            "piece_gray_threshold": (
                None if piece_threshold is None else round(piece_threshold, 1)
            ),
            "piece_count": len(pieces),
            "piece_set_quality": piece_quality,
            "stable_four_pieces": is_stable,
            "coordinate_frame": "A4 top-left; +X right; +Y down; millimetres",
            "source_region": self.config.source_region,
            "target_region": (
                "upper" if self.config.source_region == "lower" else "lower"
            ),
            "pieces": pieces,
            "assembly_plan": assembly_plan,
        }
        return status, original, annotated, mask


class CameraUndistorter:
    """Optional calibrated remap before A4 detection and homography."""

    def __init__(self, path: str, width: int, height: int) -> None:
        self.path = Path(path)
        self.enabled = False
        self.error: str | None = None
        self.mean_reprojection_error: float | None = None
        self.calibration_method: str | None = None
        self.file_sha256: str | None = None
        self.map1: np.ndarray | None = None
        self.map2: np.ndarray | None = None
        if not self.path.exists():
            self.error = "camera calibration file not found"
            return
        try:
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.file_sha256 = digest.hexdigest()
            with np.load(str(self.path), allow_pickle=False) as calibration:
                camera_matrix = np.asarray(
                    calibration["camera_matrix"], np.float64
                )
                dist_coeffs = np.asarray(
                    calibration["dist_coeffs"], np.float64
                )
                image_size = tuple(
                    int(value) for value in calibration["image_size"]
                )
                if "mean_reprojection_error" in calibration:
                    error_value = np.asarray(
                        calibration["mean_reprojection_error"]
                    )
                    if error_value.size != 1:
                        raise ValueError("mean_reprojection_error must be scalar")
                    self.mean_reprojection_error = float(error_value.reshape(-1)[0])
                if "calibration_method" in calibration:
                    method_value = np.asarray(calibration["calibration_method"])
                    if method_value.size != 1:
                        raise ValueError("calibration_method must be scalar")
                    self.calibration_method = str(method_value.reshape(-1)[0])
                else:
                    self.calibration_method = "checkerboard"
                quality_gate_passed = (
                    bool(np.asarray(calibration["quality_gate_passed"]).reshape(-1)[0])
                    if "quality_gate_passed" in calibration
                    else None
                )
                validation_ratio = (
                    float(np.asarray(calibration["validation_rms_ratio"]).reshape(-1)[0])
                    if "validation_rms_ratio" in calibration
                    else None
                )
            if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
                raise ValueError("camera_matrix must be finite 3x3")
            dist_coeffs = dist_coeffs.reshape(-1)
            if dist_coeffs.size not in (4, 5, 8, 12, 14) or not np.all(
                np.isfinite(dist_coeffs)
            ):
                raise ValueError("dist_coeffs shape or values are invalid")
            if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
                raise ValueError("camera focal lengths must be positive")
            if self.mean_reprojection_error is None or not math.isfinite(
                self.mean_reprojection_error
            ):
                raise ValueError("calibration quality error is missing or invalid")
            if self.calibration_method == "single_frame_a4_plumb_line_candidate":
                if quality_gate_passed is not True:
                    raise ValueError("A4 plumb-line quality gate did not pass")
                if validation_ratio is None or not math.isfinite(validation_ratio):
                    raise ValueError("A4 plumb-line validation ratio is missing")
                if validation_ratio > 0.72:
                    raise ValueError("A4 plumb-line validation ratio is too high")
                if self.mean_reprojection_error > 1.20:
                    raise ValueError("A4 plumb-line corrected RMS exceeds 1.20 px")
            elif self.mean_reprojection_error > 0.80:
                raise ValueError("checkerboard reprojection error exceeds 0.80 px")
            if image_size != (int(width), int(height)):
                raise ValueError(
                    f"calibration resolution {image_size} != capture {(width, height)}"
                )
            new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                dist_coeffs,
                image_size,
                1.0,
                image_size,
            )
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                camera_matrix,
                dist_coeffs,
                None,
                new_matrix,
                image_size,
                cv2.CV_16SC2,
            )
            self.enabled = True
            self.error = None
        except Exception as exc:
            self.enabled = False
            self.error = str(exc)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(
            frame, self.map1, self.map2, cv2.INTER_LINEAR
        )

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.enabled,
            "file": str(self.path),
            "file_sha256": self.file_sha256,
            "calibration_method": self.calibration_method,
            "mean_reprojection_error_px": (
                None
                if self.mean_reprojection_error is None
                else round(self.mean_reprojection_error, 4)
            ),
            "error": self.error,
        }


class CameraSource:
    def __init__(self, width: int, height: int, device: str, fps: int) -> None:
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open camera: {device} or index 0")

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("camera frame read failed")
        return frame

    def close(self) -> None:
        self.capture.release()


class PuzzleVisionApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.running = True
        self.config = DetectorConfig(
            contrast=args.contrast,
            min_area_mm2=args.min_area,
            max_area_mm2=args.max_area,
            stable_frames=args.stable_frames,
            stable_center_mm=args.stable_center_mm,
            paper_mode=args.paper_mode,
            divider_y_mm=args.divider_y_mm,
            source_region=args.source_region,
        )
        self.detector = PuzzleDetector(self.config)
        self.motion_plan_stability = MotionPlanStabilityGate(
            required_samples=3,
            position_tolerance_mm=0.8,
            angle_tolerance_deg=1.5,
        )
        self.mode1_target_latch = Mode1TargetLatch(clearance_mm=3.0)
        self.camera: CameraSource | None = None
        self.undistorter = CameraUndistorter(
            args.camera_calibration_file, args.width, args.height
        )
        self.a4_calibration_file = Path(args.a4_calibration_file)
        # Re-detect the physical A4 on every service start.  The saved physical
        # corner order is only a hint: it is accepted after the currently
        # detected quadrangle matches it for several consecutive frames.  This
        # prevents a 180-degree auto-orientation flip while still rejecting a
        # moved camera or paper.
        self.saved_a4_hint = self._load_a4_calibration()
        self.saved_a4_match_count = 0
        self.a4_lock_method = "waiting_for_auto_detection"
        self.manual_corners: np.ndarray | None = None
        self.status_data: dict[str, Any] = {
            "state": "STARTING", "piece_count": 0, "pieces": []
        }
        self.raw_frame: np.ndarray | None = None
        self.camera_jpeg: bytes | None = None
        self.warp_jpeg: bytes | None = None
        self.mask_jpeg: bytes | None = None
        calibration = StageCalibration(rotation_sign=args.rotation_sign)
        self.controller = GantryTaskController(
            self.vision_status,
            args.serial_device,
            args.serial_baud,
            calibration,
            args.vision_wait_seconds,
        )
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _load_a4_calibration(self) -> np.ndarray | None:
        try:
            payload = json.loads(
                self.a4_calibration_file.read_text(encoding="utf-8")
            )
            if self.undistorter.enabled and not payload.get(
                "camera_undistortion_enabled", False
            ):
                raise ValueError(
                    "A4 corners predate lens calibration; relock A4"
                )
            if self.undistorter.enabled:
                expected_hash = payload.get("camera_calibration_sha256")
                if not expected_hash:
                    raise ValueError(
                        "A4 corners are not bound to a camera calibration; relock A4"
                    )
                if expected_hash != self.undistorter.file_sha256:
                    raise ValueError(
                        "A4 corners belong to a different camera calibration; relock A4"
                    )
            corners = np.asarray(payload["physical_a4_corners_px"], np.float32)
            if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
                raise ValueError("invalid saved A4 corners")
            return corners
        except FileNotFoundError:
            return None
        except Exception as exc:
            print(f"Ignoring invalid A4 calibration: {exc}", flush=True)
            return None

    def _save_a4_calibration(self) -> None:
        with self.lock:
            corners = (
                None
                if self.manual_corners is None
                else np.asarray(self.manual_corners, np.float32).tolist()
            )
        if corners is None:
            return
        payload = {
            "physical_a4_corners_px": corners,
            "order": "A4_TOP_LEFT,TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT",
            "capture_size_px": [self.args.width, self.args.height],
            "saved_at_unix": round(time.time(), 3),
            "camera_undistortion_enabled": self.undistorter.enabled,
        }
        if self.undistorter.enabled:
            payload["camera_calibration_sha256"] = self.undistorter.file_sha256
            payload["camera_calibration_method"] = (
                self.undistorter.calibration_method
            )
        self.a4_calibration_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _auto_lock_stable_a4(self, status: dict[str, Any]) -> bool:
        """Lock a fresh automatic A4 only after the complete scene is stable.

        Four valid lower-half pieces disambiguate the A4's physical 180-degree
        orientation.  Requiring them also prevents a hand, magnet, or bright
        calibration card from being used to lock an unsafe coordinate frame.
        """
        quality = status.get("piece_set_quality") or {}
        if not (
            status.get("a4_found", False)
            and status.get("a4_source") == "auto"
            and status.get("a4_corner_stable", False)
            and status.get("divider_found", False)
            and status.get("stable_four_pieces", False)
            and quality.get("valid", False)
            and float(status.get("a4_orientation_score", -1.0e9)) > 0.35
        ):
            return False
        try:
            corners = np.asarray(status["a4_corners_px"], np.float32)
        except (KeyError, TypeError, ValueError):
            return False
        if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
            return False
        with self.lock:
            if self.manual_corners is not None:
                return False
            self.manual_corners = corners.copy()
        self._save_a4_calibration()
        self.detector.center_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()
        self.a4_lock_method = "fresh_auto_four_piece_lock"
        return True

    def _auto_lock_verified_saved_a4(
        self, status: dict[str, Any], frame: np.ndarray
    ) -> bool:
        """Reuse physical corner order only after verifying today's geometry.

        Automatic A4 orientation can be ambiguous by 180 degrees when the two
        black halves look alike.  The unordered paper outline is not ambiguous,
        so compare it with the last confirmed outline without assuming a point
        order.  Five consecutive close matches verify that neither camera nor
        paper moved; only then restore the saved physical TL/TR/BR/BL order.
        """
        hint = self.saved_a4_hint
        if hint is None or not (
            status.get("a4_found", False)
            and status.get("a4_source") == "auto"
        ):
            self.saved_a4_match_count = 0
            return False
        try:
            detected = np.asarray(status["a4_corners_px"], np.float32)
        except (KeyError, TypeError, ValueError):
            self.saved_a4_match_count = 0
            return False
        if detected.shape != (4, 2) or not np.all(np.isfinite(detected)):
            self.saved_a4_match_count = 0
            return False

        best_mean = math.inf
        best_max = math.inf
        for permutation in permutations(range(4)):
            aligned = detected[list(permutation)]
            distances = np.linalg.norm(aligned - hint, axis=1)
            mean_distance = float(np.mean(distances))
            max_distance = float(np.max(distances))
            if (mean_distance, max_distance) < (best_mean, best_max):
                best_mean = mean_distance
                best_max = max_distance
        status["a4_saved_hint_mean_error_px"] = round(best_mean, 2)
        status["a4_saved_hint_max_error_px"] = round(best_max, 2)

        if best_mean <= 20.0 and best_max <= 30.0:
            self.saved_a4_match_count += 1
        else:
            self.saved_a4_match_count = 0
        status["a4_saved_hint_match_count"] = self.saved_a4_match_count
        if self.saved_a4_match_count < 5:
            return False

        # A previous failed auto-orientation may have saved the same outline
        # with a cyclically shifted physical corner order.  Evaluate all four
        # rotations and accept only an orientation that sees the complete,
        # valid four-piece set in the configured source half.
        valid_candidates: list[tuple[float, np.ndarray, int]] = []
        for shift in range(4):
            candidate = np.roll(hint, -shift, axis=0).copy()
            try:
                candidate_status, _, _, _ = self.detector.detect(frame, candidate)
            except Exception:
                continue
            candidate_quality = candidate_status.get("piece_set_quality") or {}
            if not (
                candidate_status.get("divider_found", False)
                and int(candidate_status.get("piece_count", 0)) == 4
                and candidate_quality.get("valid", False)
            ):
                continue
            total_area_error = abs(
                float(candidate_quality.get("total_area_mm2", 0.0))
                - float(candidate_quality.get("expected_total_area_mm2", 6000.0))
            )
            valid_candidates.append((total_area_error, candidate, shift))
        self.detector.center_history.clear()
        self.detector.plan_history.clear()
        if not valid_candidates:
            status["a4_saved_hint_orientation_error"] = (
                "NO_ROTATION_CONTAINS_VALID_FOUR_PIECES"
            )
            return False
        _, verified_corners, shift = min(valid_candidates, key=lambda item: item[0])
        status["a4_saved_hint_selected_rotation"] = int(shift)

        with self.lock:
            if self.manual_corners is not None:
                return False
            self.manual_corners = verified_corners
            self.saved_a4_hint = verified_corners.copy()
        # Repair a calibration whose geometry was correct but physical order
        # was previously saved with the wrong 180-degree rotation.
        self._save_a4_calibration()
        self.detector.center_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()
        self.a4_lock_method = "saved_order_after_live_geometry_verification"
        return True

    def _loop(self) -> None:
        previous = time.monotonic()
        while self.running:
            try:
                if self.camera is None:
                    try:
                        self.camera = CameraSource(
                            self.args.width,
                            self.args.height,
                            self.args.camera_device,
                            self.args.camera_fps,
                        )
                        previous = time.monotonic()
                    except Exception as exc:
                        with self.lock:
                            self.status_data = {
                                "state": "CAMERA_MISSING",
                                "error": str(exc),
                                "message": "等待摄像头接入，程序每2秒自动重连",
                                "piece_count": 0,
                                "pieces": [],
                            }
                        time.sleep(2.0)
                        continue
                sensor_frame = self.camera.read()
                frame = self.undistorter.apply(sensor_frame)
                with self.lock:
                    corners = (
                        None
                        if self.manual_corners is None
                        else self.manual_corners.copy()
                    )
                    a4_locked = self.manual_corners is not None
                try:
                    status, camera_view, warped, mask = self.detector.detect(
                        frame, corners
                    )
                except Exception as exc:
                    # A malformed/noisy vision frame must not be reported as a
                    # camera disconnect.  Keep the capture open and simply let
                    # the next frame recover.
                    with self.lock:
                        self.status_data = {
                            "state": "PROCESSING_ERROR",
                            "error": str(exc),
                            "message": "本帧视觉处理失败，下一帧自动恢复",
                            "piece_count": 0,
                            "pieces": [],
                        }
                    continue
                auto_locked_now = False
                if corners is None:
                    auto_locked_now = self._auto_lock_verified_saved_a4(
                        status, frame
                    )
                    if not auto_locked_now:
                        auto_locked_now = self._auto_lock_stable_a4(status)
                    if auto_locked_now:
                        a4_locked = True
                        status["a4_auto_orientation"] = status.get(
                            "a4_orientation"
                        )
                        status["a4_source"] = "auto_locked"
                        status["a4_orientation"] = "locked_physical_a4"
                        # Histories were cleared above; require fresh stable
                        # measurements in the newly locked coordinate frame.
                        status["stable_four_pieces"] = False
                now = time.monotonic()
                fps = 1.0 / max(1.0e-3, now - previous)
                previous = now
                status.update(
                    {
                        "state": (
                            "NEED_A4"
                            if not status["a4_found"]
                            else (
                                "READY"
                                if status.get("divider_found", False)
                                else "NEED_DIVIDER"
                            )
                        ),
                        "fps": round(fps, 1),
                        "a4_auto_locked_this_frame": auto_locked_now,
                        "config": {
                            "contrast": self.config.contrast,
                            "min_area_mm2": self.config.min_area_mm2,
                            "max_area_mm2": self.config.max_area_mm2,
                            "paper_mode": self.config.paper_mode,
                            "source_region": self.config.source_region,
                            "divider_y_mm": self.config.divider_y_mm,
                            "a4_locked": a4_locked,
                            "camera_calibration": self.undistorter.status(),
                        },
                    }
                )
                motion_plans: dict[str, Any] = {}
                for mode in (1, 2):
                    plan = build_task_plan(
                        mode, status, self.controller.calibration
                    )
                    if mode == 1:
                        plan = self.mode1_target_latch.update(
                            plan, status, self.controller.calibration
                        )
                    plan["estimated_seconds"] = estimate_plan_seconds(
                        plan, self.controller.calibration
                    )
                    plan["within_120_seconds"] = bool(
                        plan.get("ready", False)
                        and plan["estimated_seconds"] <= 120.0
                    )
                    plan["safe_time_budget"] = bool(
                        plan.get("ready", False)
                        and plan["estimated_seconds"] <= 105.0
                    )
                    blockers: list[str] = []
                    if not self.undistorter.enabled:
                        blockers.append("CAMERA_CALIBRATION_NOT_LOADED")
                    if not a4_locked or status.get("a4_orientation") != (
                        "locked_physical_a4"
                    ):
                        blockers.append("A4_PHYSICAL_ORIENTATION_NOT_LOCKED")
                    divider_is_stable = bool(
                        status.get("divider_found", False)
                        and (
                            status.get("divider_source") == "manual_mm"
                            or (
                                int(status.get("divider_samples", 0)) >= 12
                                and float(status.get("divider_spread_px", 999.0))
                                <= 1.0
                            )
                        )
                    )
                    if not divider_is_stable:
                        blockers.append("DIVIDER_NOT_STABLE")
                    if not status.get("stable_four_pieces", False):
                        blockers.append("FOUR_PIECES_NOT_STABLE")
                    if not (status.get("piece_set_quality") or {}).get(
                        "valid", False
                    ):
                        blockers.append("PIECE_SET_QUALITY_FAILED")
                    if mode == 2:
                        assembly = status.get("assembly_plan") or {}
                        if not (
                            assembly.get("ready", False)
                            and int(assembly.get("measurement_samples", 0)) >= 8
                            and float(assembly.get("position_spread_mm", 999.0))
                            <= 0.8
                            and float(assembly.get("angle_spread_deg", 999.0))
                            <= 1.5
                            and all(
                                float(move.get("shape_residual", 999.0)) <= 0.35
                                for move in assembly.get("moves", [])
                            )
                        ):
                            blockers.append("ASSEMBLY_MEASUREMENT_NOT_STABLE")
                    if blockers or not plan.get("ready", False):
                        self.motion_plan_stability.clear(mode)
                        plan_stability = {
                            "stable": False,
                            "samples": 0,
                            "required_samples": 3,
                            "position_spread_mm": None,
                            "angle_spread_deg": None,
                            "reason": "VISION_QUALITY_NOT_READY",
                        }
                    else:
                        plan_stability = self.motion_plan_stability.update(
                            mode, plan
                        )
                    plan["vision_quality_ready"] = not blockers
                    plan["execution_blockers"] = blockers
                    plan["plan_stability"] = plan_stability
                    plan["execution_ready"] = bool(
                        plan.get("ready", False)
                        and plan["safe_time_budget"]
                        and not blockers
                        and plan_stability["stable"]
                    )
                    motion_plans[str(mode)] = plan
                status["motion_plans"] = motion_plans
                with self.lock:
                    self.status_data = status
                    # Keep the untouched sensor frame for one-shot lens
                    # calibration.  Detector.detect() annotates only a copy.
                    self.raw_frame = sensor_frame
                    self.camera_jpeg = encode_jpeg(camera_view)
                    self.warp_jpeg = encode_jpeg(warped)
                    self.mask_jpeg = encode_jpeg(mask)
            except Exception as exc:
                camera = self.camera
                self.camera = None
                if camera is not None:
                    camera.close()
                with self.lock:
                    self.status_data = {
                        "state": "CAMERA_RECONNECTING", "error": str(exc),
                        "message": "摄像头读取失败，2秒后自动重连",
                        "piece_count": 0, "pieces": [],
                    }
                time.sleep(2.0)

    def set_corners(self, points: list[list[float]]) -> None:
        array = np.asarray(points, dtype=np.float32)
        if array.shape != (4, 2) or not np.all(np.isfinite(array)):
            raise ValueError("need exactly four finite [x,y] camera points")
        # The web page explicitly asks for physical A4 TL,TR,BR,BL order.
        with self.lock:
            self.manual_corners = array.copy()
        self._save_a4_calibration()
        self.detector.center_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()

    def lock_current_a4(self) -> None:
        with self.lock:
            if not self.status_data.get("a4_found", False):
                raise ValueError("A4 is not currently detected")
            if not self.status_data.get("a4_corner_stable", False):
                raise ValueError("wait until A4 corner measurement is stable")
            corners = np.asarray(
                self.status_data["a4_corners_px"], dtype=np.float32
            )
            # status corners already include the detector's chosen physical
            # A4 rotation.  Preserve that exact order across all later frames.
            self.manual_corners = corners.copy()
        self._save_a4_calibration()
        self.detector.center_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()

    def use_auto_a4(self) -> None:
        with self.lock:
            self.manual_corners = None
            self.saved_a4_hint = None
            self.saved_a4_match_count = 0
            self.a4_lock_method = "manual_fresh_auto_requested"
        try:
            self.a4_calibration_file.unlink()
        except FileNotFoundError:
            pass
        self.detector.center_history.clear()
        self.detector.a4_corner_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()

    def update_config(self, query: dict[str, list[str]]) -> None:
        contrast = float(query.get("contrast", [str(self.config.contrast)])[0])
        min_area = float(
            query.get("min_area", [str(self.config.min_area_mm2)])[0]
        )
        max_area = float(
            query.get("max_area", [str(self.config.max_area_mm2)])[0]
        )
        paper_mode = query.get("paper_mode", [self.config.paper_mode])[0]
        divider_y_mm = float(
            query.get("divider_y_mm", [str(self.config.divider_y_mm)])[0]
        )
        if not 5.0 <= contrast <= 100.0:
            raise ValueError("contrast must be 5..100")
        if not 20.0 <= min_area < max_area <= 10000.0:
            raise ValueError("area range is invalid")
        if paper_mode not in {"auto", "white", "black"}:
            raise ValueError("paper_mode must be auto, white or black")
        if divider_y_mm != 0.0 and not 20.0 <= divider_y_mm <= 277.0:
            raise ValueError("divider_y_mm must be 0(auto) or 20..277")
        self.config.contrast = contrast
        self.config.min_area_mm2 = min_area
        self.config.max_area_mm2 = max_area
        self.config.paper_mode = paper_mode
        self.config.divider_y_mm = divider_y_mm
        self.detector.center_history.clear()
        self.detector.divider_history.clear()
        self.detector.plan_history.clear()
        self.motion_plan_stability.clear()
        self.mode1_target_latch.clear()

    def vision_status(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.status_data, ensure_ascii=False))

    def status(self) -> dict[str, Any]:
        status = self.vision_status()
        status["robot"] = self.controller.status()
        status["a4_physical_orientation_locked"] = (
            self.manual_corners is not None
        )
        status["a4_calibration_file"] = str(self.a4_calibration_file)
        status["a4_startup_mode"] = "auto_detect_then_lock"
        status["a4_lock_method"] = self.a4_lock_method
        return status

    def start_task(self, mode: int) -> int:
        return self.controller.request_task(mode)

    def emergency_stop(self) -> None:
        self.controller.emergency_stop()

    def close(self) -> None:
        self.running = False
        self.controller.close()
        self.worker.join(timeout=1.0)
        if self.camera is not None:
            self.camera.close()


HTML = """<!doctype html><html lang=zh-CN><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>E题第一问拼图识别</title><style>
body{font-family:system-ui;background:#111;color:#eee;max-width:1180px;margin:auto;padding:16px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{background:#1d1d1d;padding:10px;border-radius:8px}
img{width:100%;border:1px solid #555;border-radius:6px}button{font-size:16px;padding:9px 14px;margin:5px}
input,select{font-size:16px;width:100px;padding:6px}pre{background:#222;padding:12px;max-height:540px;overflow:auto;white-space:pre-wrap}
.planbox{background:#1d1d1d;padding:12px;border-radius:8px;margin-top:12px;overflow:auto}table{border-collapse:collapse;width:100%;min-width:780px}th,td{border:1px solid #555;padding:7px;text-align:center}th{background:#2b2b2b}
.good{background:#137a39;color:white}.blue{background:#1769aa;color:white}.warn{color:#ffd65c}@media(max-width:760px){.grid{grid-template-columns:1fr}}
</style><h2>E题第一问：A4上的灰色镀锌拼图片识别</h2>
<p>坐标原点为A4左上角，X向右、Y向下，单位mm。自动A4失败时，在左图依次点击：左上、右上、右下、左下。</p>
<div class=grid><div class=panel><h3>原始相机 / 点击四角标定</h3><img id=cam src=/camera.mjpg><p id=clicks class=warn>已选0/4点</p></div>
<div class=panel><h3>A4透视展开与识别结果</h3><img src=/a4.mjpg></div>
<div class=panel><h3>铁片二值掩膜</h3><img src=/mask.mjpg></div>
<div class=panel><h3>参数</h3>
<label>纸张模式 <select id=papermode><option value=auto>自动</option><option value=black>黑纸</option><option value=white>白纸</option></select></label>
<label>分界线Y(mm，0自动) <input id=dividery type=number value=0 min=0 max=277 step=0.5></label>
<label>灰度差 <input id=contrast type=number value=28 min=5 max=100></label>
<label>最小面积mm² <input id=minarea type=number value=100 min=20></label>
<label>最大面积mm² <input id=maxarea type=number value=4500 min=100></label><br>
<button class=good onclick=applyConfig()>应用分割参数</button><button class=blue onclick=lockA4()>锁定稳定A4</button><button class=blue onclick=autoA4()>恢复自动A4</button>
<p>应识别到4片；绿色轮廓为铁片，紫线为上下区分界线。</p></div></div>
<div class=planbox><h3>拼图目标与滑台前置坐标（A4毫米坐标）</h3><p id=plansummary class=warn>等待稳定识别4片……</p><table><thead><tr><th>顺序</th><th>铁片</th><th>当前质心(mm)</th><th>目标质心(mm)</th><th>平移ΔXY(mm)</th><th>顺时针旋转(°)</th><th>匹配误差</th></tr></thead><tbody id=plantbody></tbody></table></div>
<h3>实时JSON</h3><pre id=status>loading...</pre><script>
let points=[];let cam=document.querySelector('#cam');cam.onclick=async(e)=>{let r=cam.getBoundingClientRect();let x=(e.clientX-r.left)*cam.naturalWidth/r.width;let y=(e.clientY-r.top)*cam.naturalHeight/r.height;points.push([Math.round(x),Math.round(y)]);document.querySelector('#clicks').textContent='已选'+points.length+'/4点 '+JSON.stringify(points);if(points.length===4){await fetch('/api/corners',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points})});points=[]}}
async function autoA4(){points=[];await fetch('/api/auto-a4',{method:'POST'})}
async function lockA4(){let r=await fetch('/api/lock-a4',{method:'POST'});if(!r.ok)alert(await r.text())}
async function applyConfig(){let q='paper_mode='+papermode.value+'&divider_y_mm='+dividery.value+'&contrast='+contrast.value+'&min_area='+minarea.value+'&max_area='+maxarea.value;let r=await fetch('/api/config?'+q,{method:'POST'});document.querySelector('#status').textContent=await r.text()}
function renderPlan(s){let p=s.assembly_plan;let body=document.querySelector('#plantbody');let summary=document.querySelector('#plansummary');if(!p||!p.moves){body.innerHTML='';summary.textContent='等待完整A4、分界线和稳定4片……';return}let r=p.target_rectangle_mm;summary.textContent='目标矩形 '+r.size.join('×')+' mm，原点 ('+r.origin.join(', ')+') mm；测量帧 '+p.measurement_samples+'，质心波动 '+p.position_spread_mm+' mm，角度波动 '+p.angle_spread_deg+'°；'+(p.ready?'方案已稳定':'正在多帧稳定');body.innerHTML=p.moves.map(m=>'<tr><td>'+m.order+'</td><td>'+m.piece_id+' → '+m.target_piece+'</td><td>'+m.current_centroid_mm.join(', ')+'</td><td>'+m.target_centroid_mm.join(', ')+'</td><td>'+m.translation_mm.join(', ')+'</td><td>'+m.rotate_deg_clockwise+'</td><td>'+m.shape_residual+'</td></tr>').join('')}
setInterval(async()=>{try{let s=await(await fetch('/api/status')).json();document.querySelector('#status').textContent=JSON.stringify(s,null,2);renderPlan(s)}catch(e){}},400)
</script></html>""".encode("utf-8")


# Control page used by the fixed-camera gantry.  Keep the older page above as
# a reference for the detector tuning controls, but serve this UTF-8 page.
CONTROL_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>E题第一问：搬运与拼图</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#101214;color:#eee;max-width:1450px;margin:auto;padding:14px}
h1{margin:.25em 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:#1b1e21;padding:12px;border-radius:9px;border:1px solid #333}
img{width:100%;border:1px solid #555;border-radius:6px}.actions{position:sticky;top:0;background:#101214eF;padding:8px 0;z-index:3}
button{font-size:18px;padding:11px 17px;margin:4px;border:0;border-radius:5px;color:#fff;cursor:pointer}.green{background:#168342}.blue{background:#1771b8}.red{background:#b52626}.gray{background:#555}
button:disabled{opacity:.35;cursor:not-allowed}.badge{display:inline-block;padding:5px 9px;border-radius:15px;margin:3px;background:#555}.ok{background:#147a3a}.bad{background:#a52727}.wait{background:#996500}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #555;padding:6px;text-align:center}th{background:#292d31}.problem{color:#ff7979;font-weight:700}.goodtext{color:#63e18f}.muted{color:#bbb}pre{background:#080909;padding:10px;max-height:380px;overflow:auto;white-space:pre-wrap}
input,select{font-size:15px;padding:5px;margin:3px;width:95px}@media(max-width:850px){.grid{grid-template-columns:1fr}.actions{position:static}}
</style></head><body>
<h1>E题第一问：4片搬运 / 图2拼接</h1>
<p>坐标：A4物理左上角为(0,0)，X向右、Y向下，单位mm。固定初始磁铁中心为(5,61)mm。</p>
<div class="actions">
 <button id="mode1" class="green" onclick="startTask(1)">按键1：全部搬到上半区</button>
 <button id="mode2" class="blue" onclick="startTask(2)">按键2：拼成100×60矩形</button>
 <button class="red" onclick="stopTask()">急停</button>
 <button class="gray" onclick="lockA4()">锁定当前A4物理方向</button>
 <button class="gray" onclick="autoA4()">重新自动识别A4</button>
</div>
<div id="badges"></div><p id="robotmsg"></p>
<div class="grid">
 <div class="card"><h3>A4毫米坐标与识别结果</h3><img src="/a4.mjpg"></div>
 <div class="card"><h3>原始相机</h3><img id="cam" src="/camera.mjpg"><p class="muted">自动失败时，依次点击A4物理：左上、右上、右下、左下。<span id="clicks">已选0/4点</span></p></div>
</div>
<div class="grid" style="margin-top:12px">
 <div class="card"><h3>模式1预览：从下半区搬到分界线上方</h3><div id="p1summary"></div><div id="p1table"></div></div>
 <div class="card"><h3>模式2预览：图2目标矩形</h3><div id="p2summary"></div><div id="p2table"></div></div>
</div>
<div class="card" style="margin-top:12px"><h3>识别参数</h3>
 <label>纸张 <select id="papermode"><option value="auto">自动</option><option value="black">黑纸</option><option value="white">白纸</option></select></label>
 <label>分界线Y(mm，0自动) <input id="dividery" type="number" value="0" step="0.5"></label>
 <label>灰度差 <input id="contrast" type="number" value="28"></label>
 <label>最小面积 <input id="minarea" type="number" value="100"></label>
 <label>最大面积 <input id="maxarea" type="number" value="4500"></label>
 <button class="gray" onclick="applyConfig()">应用</button>
</div>
<details><summary>实时JSON / 通信记录</summary><pre id="status">loading...</pre></details>
<script>
let last=null,points=[],cam=document.querySelector('#cam');
cam.onclick=async e=>{let r=cam.getBoundingClientRect(),x=(e.clientX-r.left)*cam.naturalWidth/r.width,y=(e.clientY-r.top)*cam.naturalHeight/r.height;points.push([Math.round(x),Math.round(y)]);clicks.textContent='已选'+points.length+'/4点 '+JSON.stringify(points);if(points.length===4){let z=await fetch('/api/corners',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({points})});if(!z.ok)alert(await z.text());points=[];clicks.textContent='已选0/4点'}};
async function post(url){let r=await fetch(url,{method:'POST'}),t=await r.text();if(!r.ok)alert(t);return t}
async function startTask(m){await post('/api/task?mode='+m)}
async function stopTask(){await post('/api/stop')}
async function lockA4(){await post('/api/lock-a4')}
async function autoA4(){points=[];await post('/api/auto-a4')}
async function applyConfig(){let q='paper_mode='+papermode.value+'&divider_y_mm='+dividery.value+'&contrast='+contrast.value+'&min_area='+minarea.value+'&max_area='+maxarea.value;await post('/api/config?'+q)}
function esc(v){return String(v??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function planView(id,p){let s=document.querySelector('#p'+id+'summary'),b=document.querySelector('#p'+id+'table');if(!p){s.innerHTML='<span class=problem>尚无方案</span>';b.innerHTML='';return}let un=p.unreachable||[],bad=un.map(x=>x.piece_id+' '+x.operation+' ('+x.point_a4_mm.join(',')+')').join('；'),detail=id===1?('排布策略 '+esc(p.strategy)+(p.packing?'；总旋转 '+esc(p.packing.rotation_total_deg)+'°':'；平移 '+esc(p.common_translation_mm))):('目标 '+esc(JSON.stringify(p.target_rectangle_mm)));s.innerHTML='<b class="'+(p.ready?'goodtext':'problem')+'">'+(p.ready?'可执行':'禁止执行：'+esc(p.error))+'</b>；预计 '+p.estimated_seconds+' 秒；'+detail+(bad?'<br><span class=problem>不可达：'+esc(bad)+'</span>':'');let rows=(p.moves||[]).map(m=>{let reachable=!un.some(x=>x.piece_id===m.piece_id);return '<tr><td>'+m.order+'</td><td>'+esc(m.piece_id)+'</td><td>'+m.pick_a4_mm.join(', ')+'</td><td>'+m.place_a4_mm.join(', ')+'</td><td>'+esc(m.rotate_deg_clockwise)+'</td><td class="'+(reachable?'goodtext':'problem')+'">'+(reachable?'是':'否')+'</td></tr>'});b.innerHTML='<table><thead><tr><th>顺序</th><th>铁片</th><th>拾取A4坐标</th><th>放置A4坐标</th><th>顺时针角度</th><th>可达</th></tr></thead><tbody>'+rows.join('')+'</tbody></table>'}
function render(s){last=s;let r=s.robot||{},ser=r.serial||{},m=s.motion_plans||{},cc=(s.config||{}).camera_calibration||{};badges.innerHTML='<span class="badge '+(s.stable_four_pieces?'ok':'wait')+'">4片稳定 '+(s.stable_four_pieces?'是':'否')+'</span><span class="badge '+(s.a4_physical_orientation_locked?'ok':'wait')+'">A4方向锁定 '+(s.a4_physical_orientation_locked?'是':'否')+'</span><span class="badge '+(cc.loaded?'ok':'wait')+'">镜头标定 '+(cc.loaded?'已加载':'未加载')+'</span><span class="badge '+(ser.connected?'ok':'bad')+'">MCU串口 '+(ser.connected?'已连接':'未连接')+'</span><span class="badge">分界线Y '+esc(s.divider_y_mm)+' mm</span><span class="badge '+(r.state==='ERROR'||r.state==='REJECTED'||r.state==='STOPPED'?'bad':'ok')+'">'+esc(r.state)+'</span>';robotmsg.textContent=r.message||'';mode1.disabled=!(m['1']&&m['1'].execution_ready)||!ser.connected;mode2.disabled=!(m['2']&&m['2'].execution_ready)||!ser.connected;planView(1,m['1']);planView(2,m['2']);status.textContent=JSON.stringify(s,null,2);let c=s.config||{};if(document.activeElement.tagName!=='INPUT'&&document.activeElement.tagName!=='SELECT'){papermode.value=c.paper_mode||'auto';dividery.value=c.divider_y_mm??0;contrast.value=c.contrast??28;minarea.value=c.min_area_mm2??100;maxarea.value=c.max_area_mm2??4500}}
setInterval(async()=>{try{render(await(await fetch('/api/status')).json())}catch(e){robotmsg.textContent='网页状态读取失败：'+e}},450);
</script></body></html>""".encode("utf-8")


def make_handler(app: PuzzleVisionApp):
    class Handler(BaseHTTPRequestHandler):
        def reply(
            self, body: bytes, kind: str = "text/plain; charset=utf-8"
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self, attribute: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while app.running:
                    with app.lock:
                        jpeg = getattr(app, attribute)
                    if jpeg:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + jpeg + b"\r\n"
                        )
                    time.sleep(0.06)
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_GET(self) -> None:
            if self.path == "/":
                self.reply(CONTROL_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.reply(
                    json.dumps(app.status(), ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            elif self.path == "/camera.mjpg":
                self.stream("camera_jpeg")
            elif self.path == "/a4.mjpg":
                self.stream("warp_jpeg")
            elif self.path == "/mask.mjpg":
                self.stream("mask_jpeg")
            elif self.path == "/raw.jpg":
                with app.lock:
                    raw = (
                        None
                        if app.raw_frame is None
                        else app.raw_frame.copy()
                    )
                jpeg = None if raw is None else encode_jpeg(raw)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.reply(jpeg, "image/jpeg")
            elif self.path in {"/camera.jpg", "/a4.jpg", "/mask.jpg"}:
                attribute = {
                    "/camera.jpg": "camera_jpeg",
                    "/a4.jpg": "warp_jpeg",
                    "/mask.jpg": "mask_jpeg",
                }[self.path]
                with app.lock:
                    jpeg = getattr(app, attribute)
                if jpeg is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.reply(jpeg, "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/corners":
                    size = int(self.headers.get("Content-Length", "0"))
                    body = json.loads(self.rfile.read(size).decode("utf-8"))
                    app.set_corners(body["points"])
                    self.reply("manual A4 corners accepted".encode())
                elif parsed.path == "/api/auto-a4":
                    app.use_auto_a4()
                    self.reply("automatic A4 detection enabled".encode())
                elif parsed.path == "/api/lock-a4":
                    app.lock_current_a4()
                    self.reply("stable A4 calibration locked".encode())
                elif parsed.path == "/api/config":
                    app.update_config(parse_qs(parsed.query))
                    self.reply("segmentation config updated".encode())
                elif parsed.path == "/api/task":
                    mode = int(parse_qs(parsed.query).get("mode", ["0"])[0])
                    request_id = app.start_task(mode)
                    self.reply(
                        json.dumps(
                            {"accepted": True, "request_id": request_id, "mode": mode}
                        ).encode(),
                        "application/json; charset=utf-8",
                    )
                elif parsed.path == "/api/stop":
                    app.emergency_stop()
                    self.reply("STOP sent; return to HOME and reset MCU".encode())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                body = str(exc).encode("utf-8")
                self.send_response(HTTPStatus.CONFLICT)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def synthetic_frame(
    paper_mode: str = "white",
    divider_ratio: float = 0.5,
    strong_perspective: bool = False,
) -> np.ndarray:
    black = paper_mode == "black"
    paper_shade = 22 if black else 250
    line_shade = 245 if black else 15
    canonical = np.full(
        (WARP_HEIGHT, WARP_WIDTH, 3), paper_shade, np.uint8
    )
    divider = int(round(WARP_HEIGHT * divider_ratio))
    cv2.line(
        canonical,
        (0, divider),
        (WARP_WIDTH - 1, divider),
        (line_shade, line_shade, line_shade),
        8,
    )
    polygons = (
        np.asarray([[70, 80], [270, 70], [190, 230]], np.int32),
        np.asarray([[360, 80], [620, 120], [580, 300], [410, 250]], np.int32),
        np.asarray([[90, 320], [270, 280], [330, 450], [120, 430]], np.int32),
        np.asarray([[500, 360], [680, 320], [650, 500], [540, 470]], np.int32),
    )
    for index, polygon in enumerate(polygons):
        shade = (130 + index * 15) if black else (115 + index * 12)
        cv2.fillPoly(canonical, [polygon], (shade, shade, shade))
    frame_shade = 145 if black else 65
    frame = np.full((720, 960, 3), frame_shade, np.uint8)
    source = np.asarray(
        [[0, 0], [WARP_WIDTH - 1, 0], [WARP_WIDTH - 1, WARP_HEIGHT - 1], [0, WARP_HEIGHT - 1]],
        np.float32,
    )
    if strong_perspective:
        destination = np.asarray(
            [[330, 35], [760, 145], [650, 685], [145, 570]],
            np.float32,
        )
    else:
        destination = np.asarray(
            [[275, 35], [700, 55], [720, 680], [245, 660]],
            np.float32,
        )
    matrix = cv2.getPerspectiveTransform(source, destination)
    projected = cv2.warpPerspective(canonical, matrix, (frame.shape[1], frame.shape[0]))
    valid = cv2.warpPerspective(
        np.full((WARP_HEIGHT, WARP_WIDTH), 255, np.uint8),
        matrix, (frame.shape[1], frame.shape[0]),
    )
    frame[valid > 0] = projected[valid > 0]
    return frame


def synthetic_solver_frame() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    canonical = np.full(
        (WARP_HEIGHT, WARP_WIDTH, 3), 22, np.uint8
    )
    divider_y = int(round(172.0 * WARP_PX_PER_MM))
    cv2.line(
        canonical,
        (0, divider_y),
        (WARP_WIDTH - 1, divider_y),
        (245, 245, 245),
        8,
    )
    poses = {
        "G1_BIG_TRIANGLE": ((125.0, 48.0), 12.0),
        "G4_BOTTOM": ((148.0, 128.0), -10.0),
        "G3_MIDDLE": ((52.0, 126.0), 75.0),
        "G2_SMALL_TOP_LEFT": ((30.0, 46.0), -35.0),
    }
    expected_rotations: dict[str, float] = {}
    for name, template in TARGET_PIECE_TEMPLATES:
        center, angle = poses[name]
        _area, centroid = polygon_area_centroid(template)
        radians = math.radians(angle)
        rotation = np.asarray(
            [
                [math.cos(radians), math.sin(radians)],
                [-math.sin(radians), math.cos(radians)],
            ],
            dtype=np.float64,
        )
        transformed = (
            (template - centroid) @ rotation
            + np.asarray(center, dtype=np.float64)
        )
        polygon_px = np.rint(transformed * WARP_PX_PER_MM).astype(np.int32)
        cv2.fillPoly(canonical, [polygon_px], (155, 155, 155))
        expected_rotations[name] = -angle

    frame = np.full((720, 960, 3), 145, np.uint8)
    source = np.asarray(
        [
            [0, 0],
            [WARP_WIDTH - 1, 0],
            [WARP_WIDTH - 1, WARP_HEIGHT - 1],
            [0, WARP_HEIGHT - 1],
        ],
        np.float32,
    )
    destination = np.asarray(
        [[275, 35], [700, 55], [720, 680], [245, 660]],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    projected = cv2.warpPerspective(
        canonical, matrix, (frame.shape[1], frame.shape[0])
    )
    valid = cv2.warpPerspective(
        np.full((WARP_HEIGHT, WARP_WIDTH), 255, np.uint8),
        matrix,
        (frame.shape[1], frame.shape[0]),
    )
    frame[valid > 0] = projected[valid > 0]
    return frame, destination, expected_rotations


def self_test() -> int:
    cases = (
        ("white", 0.64, False),
        ("black", 0.73, False),
        ("black_perspective", 0.68, True),
    )
    for case_name, divider_ratio, strong_perspective in cases:
        paper_mode = "black" if case_name.startswith("black") else "white"
        detector = PuzzleDetector(
            DetectorConfig(stable_frames=1, source_region="upper")
        )
        frame = synthetic_frame(
            paper_mode, divider_ratio, strong_perspective
        )
        status, _original, _warped, _mask = detector.detect(frame, None)
        print(
            case_name,
            json.dumps(status, ensure_ascii=False, indent=2),
        )
        if not status["a4_found"]:
            raise RuntimeError(f"synthetic {case_name} A4 was not detected")
        if status["paper_mode_used"] != paper_mode:
            raise RuntimeError(
                f"expected {paper_mode} mode, got {status['paper_mode_used']}"
            )
        expected_divider_mm = A4_HEIGHT_MM * divider_ratio
        if abs(float(status["divider_y_mm"]) - expected_divider_mm) > 2.0:
            raise RuntimeError(
                f"expected divider near {expected_divider_mm:.1f} mm, "
                f"got {status['divider_y_mm']}"
            )
        if status["piece_count"] != 4:
            raise RuntimeError(
                f"synthetic {case_name} test expected 4 pieces, "
                f"got {status['piece_count']}"
            )

    solver_frame, solver_corners, expected_rotations = synthetic_solver_frame()
    detector = PuzzleDetector(
        DetectorConfig(stable_frames=5, source_region="upper")
    )
    solver_status: dict[str, Any] = {}
    for _frame_index in range(12):
        solver_status, _original, _warped, _mask = detector.detect(
            solver_frame, solver_corners
        )
    plan = solver_status.get("assembly_plan")
    if not plan or not plan.get("ready", False):
        raise RuntimeError(f"synthetic assembly plan did not stabilize: {plan}")
    for move in plan["moves"]:
        expected = expected_rotations[move["target_piece"]]
        measured = float(move["rotate_deg_clockwise"])
        error = (measured - expected + 180.0) % 360.0 - 180.0
        if abs(error) > 2.0:
            raise RuntimeError(
                f"{move['target_piece']} rotation expected {expected:.1f}, "
                f"got {measured:.1f}"
            )
    print("solver_precision", json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


def process_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError(f"cannot read image: {args.image}")
    detector = PuzzleDetector(
        DetectorConfig(
            args.contrast, args.min_area, args.max_area, 1, 2.0,
            args.paper_mode, 0.0, args.source_region,
        )
    )
    status, original, warped, mask = detector.detect(frame, None)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "camera_annotated.jpg"), original)
    cv2.imwrite(str(output / "a4_annotated.jpg"), warped)
    cv2.imwrite(str(output / "piece_mask.png"), mask)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E题第一问A4灰色铁片识别")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8081)
    parser.add_argument("--camera-device", default=DEFAULT_CAMERA_DEVICE)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--contrast", type=float, default=28.0)
    parser.add_argument(
        "--paper-mode", choices=("auto", "white", "black"), default="auto"
    )
    parser.add_argument(
        "--source-region", choices=("lower", "upper"), default="lower",
        help="physical A4 half containing the four loose pieces",
    )
    parser.add_argument(
        "--divider-y-mm", type=float, default=0.0,
        help="0=auto; otherwise divider Y in A4 millimetres",
    )
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--max-area", type=float, default=4500.0)
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--stable-center-mm", type=float, default=2.0)
    parser.add_argument(
        # Raspberry Pi 5 maps /dev/serial0 to the dedicated debug header
        # (ttyAMA10).  The 40-pin header GPIO14/15 used by this project is
        # RP1 UART0, exposed as /dev/ttyAMA0.
        "--serial-device", default="/dev/ttyAMA0",
        help="UART connected to MSPM0 PB12/PB13",
    )
    parser.add_argument("--serial-baud", type=int, default=115200)
    parser.add_argument(
        "--vision-wait-seconds", type=float, default=15.0,
        help="time after KEY to wait for one stable reachable four-piece plan",
    )
    parser.add_argument(
        "--rotation-sign", type=float, choices=(-1.0, 1.0), default=1.0,
        help="motor-5 sign for positive clockwise image rotation",
    )
    parser.add_argument(
        "--a4-calibration-file", default="a4_calibration.json",
        help="persistent physical A4 corner order for the fixed camera",
    )
    parser.add_argument(
        "--camera-calibration-file", default="camera_calibration.npz",
        help="optional output from calibrate_camera.py",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("puzzle_output"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.image is not None:
        return process_image(args)
    app = PuzzleVisionApp(args)
    server = ThreadingHTTPServer((args.host, args.web_port), make_handler(app))

    def shutdown(_signum=None, _frame=None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Puzzle UI: http://<raspberry-pi-ip>:{args.web_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
