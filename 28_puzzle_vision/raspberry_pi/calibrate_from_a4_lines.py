#!/usr/bin/env python3
"""Build a *candidate* lens calibration from straight lines on the A4 scene.

This is a conservative, offline fallback for a fixed camera when a checkerboard
capture is not immediately available.  It follows the four black/white A4
boundaries and the thin white divider, then searches for radial coefficients
that make those five curves straight (the classical ``plumb-line`` method).

The result is deliberately named ``camera_calibration_candidate.npz``.  This
script refuses to write ``camera_calibration.npz`` so it cannot silently enable
an approximate one-frame calibration in the live puzzle service.  Inspect the
side-by-side preview and only promote a candidate after a physical scale check.

Recommended Raspberry Pi command (use an unannotated ``/raw.jpg`` frame)::

    python3 calibrate_from_a4_lines.py \
      --image raw.jpg --a4-calibration a4_calibration.json \
      --output camera_calibration_candidate.npz \
      --preview a4_line_calibration_preview.jpg

Only OpenCV, NumPy and the Python standard library are required.  A single
plane cannot uniquely determine focal length and principal point, so this tool
fixes them to conservative values and estimates only k1/k2.  A multi-view
checkerboard calibration with ``calibrate_camera.py`` remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Sequence

import cv2
import numpy as np


FORMAL_CALIBRATION_NAME = "camera_calibration.npz"
DEFAULT_OUTPUT = "camera_calibration_candidate.npz"
DEFAULT_PREVIEW = "a4_line_calibration_preview.jpg"


class CalibrationRejected(RuntimeError):
    """Raised when extraction or the independent quality gate is unsafe."""


def read_image(path: Path) -> np.ndarray:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise CalibrationRejected("cannot read image: {} ({})".format(path, exc))
    if frame is None or frame.ndim != 3:
        raise CalibrationRejected("cannot decode image: {}".format(path))
    return frame


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError("OpenCV could not encode preview {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def load_a4_corners(path: Path) -> np.ndarray:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        points = np.asarray(document["physical_a4_corners_px"], np.float64)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationRejected(
            "cannot load physical_a4_corners_px from {}: {}".format(path, exc)
        )
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise CalibrationRejected("A4 calibration must contain four finite points")
    return order_cyclic(points)


def parse_corners(text: str) -> np.ndarray:
    values = [float(token.strip()) for token in text.split(",") if token.strip()]
    if len(values) != 8:
        raise CalibrationRejected("--corners requires x1,y1,...,x4,y4")
    return order_cyclic(np.asarray(values, np.float64).reshape(4, 2))


def order_cyclic(points: np.ndarray) -> np.ndarray:
    """Return a cyclic geometric order; physical A4 direction is irrelevant here."""
    points = np.asarray(points, np.float64).reshape(4, 2)
    centre = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    area2 = float(
        np.sum(
            ordered[:, 0] * np.roll(ordered[:, 1], -1)
            - ordered[:, 1] * np.roll(ordered[:, 0], -1)
        )
    )
    if abs(area2) < 1000.0:
        raise CalibrationRejected("A4 corner quadrilateral is degenerate")
    return ordered


def _remap_strip(
    gray: np.ndarray, bases: np.ndarray, normal: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    coordinates = bases[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    map_x = coordinates[:, :, 0].astype(np.float32)
    map_y = coordinates[:, :, 1].astype(np.float32)
    return cv2.remap(
        gray.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _dynamic_trace(evidence: np.ndarray, max_step: int = 5) -> np.ndarray:
    """Pick a strong continuous ridge through a station-by-offset score image."""
    evidence = np.asarray(evidence, np.float64)
    rows, columns = evidence.shape
    if rows < 4 or columns < 5:
        raise CalibrationRejected("line search grid is too small")
    score = np.full((rows, columns), -np.inf, np.float64)
    back = np.zeros((rows, columns), np.int16)
    score[0] = evidence[0]
    offsets = np.arange(-max_step, max_step + 1, dtype=np.int16)
    penalty = 1.2 * np.square(offsets.astype(np.float64))
    for row in range(1, rows):
        previous = score[row - 1]
        for column in range(columns):
            candidates = column + offsets
            valid = (candidates >= 0) & (candidates < columns)
            candidate_columns = candidates[valid]
            candidate_scores = previous[candidate_columns] - penalty[valid]
            best_local = int(np.argmax(candidate_scores))
            best_column = int(candidate_columns[best_local])
            score[row, column] = evidence[row, column] + candidate_scores[best_local]
            back[row, column] = best_column
    path = np.empty(rows, np.int32)
    path[-1] = int(np.argmax(score[-1]))
    for row in range(rows - 1, 0, -1):
        path[row - 1] = int(back[row, path[row]])
    return path


def _subpixel_peak(scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    refined = indices.astype(np.float64)
    for row, index in enumerate(indices):
        if index <= 0 or index >= scores.shape[1] - 1:
            continue
        left, middle, right = (
            float(scores[row, index - 1]),
            float(scores[row, index]),
            float(scores[row, index + 1]),
        )
        denominator = left - 2.0 * middle + right
        if denominator < -1e-6:
            refined[row] += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return refined


def _smooth_path_mask(path_offsets: np.ndarray) -> np.ndarray:
    """Reject isolated ridge jumps without forcing the measured curve straight."""
    count = len(path_offsets)
    local = np.empty(count, np.float64)
    for index in range(count):
        lo, hi = max(0, index - 4), min(count, index + 5)
        local[index] = float(np.median(path_offsets[lo:hi]))
    residual = np.abs(path_offsets - local)
    mad = float(np.median(residual))
    return residual <= max(2.5, 4.0 * mad + 0.5)


def trace_outer_edge(
    gray: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    polygon_centre: np.ndarray,
    search_radius: int,
    sample_count: int,
    minimum_contrast: float,
) -> tuple[np.ndarray, dict[str, float]]:
    vector = np.asarray(end - start, np.float64)
    length = float(np.linalg.norm(vector))
    if length < 80.0:
        raise CalibrationRejected("A4 edge seed is too short")
    tangent = vector / length
    normal = np.asarray([-tangent[1], tangent[0]], np.float64)
    midpoint = 0.5 * (start + end)
    if float(np.dot(polygon_centre - midpoint, normal)) < 0.0:
        normal = -normal
    stations = np.linspace(0.045, 0.955, sample_count, dtype=np.float64)
    bases = start[None, :] + stations[:, None] * vector[None, :]
    offsets = np.arange(-search_radius, search_radius + 1, dtype=np.float64)
    samples = _remap_strip(gray, bases, normal, offsets)
    samples = cv2.GaussianBlur(samples, (5, 1), 0.0)
    # Moving from negative (outside/white) toward positive (inside/black), the
    # A4 border produces a strong negative derivative.  A six-pixel baseline
    # suppresses texture and the thin coloured annotation overlay.
    evidence = np.zeros_like(samples, np.float64)
    evidence[:, 3:-3] = samples[:, :-6] - samples[:, 6:]
    evidence[:, :3] = -1000.0
    evidence[:, -3:] = -1000.0
    path = _dynamic_trace(evidence)
    refined_index = _subpixel_peak(evidence, path)
    path_offsets = offsets[0] + refined_index
    selected_evidence = evidence[np.arange(sample_count), path]
    keep = (selected_evidence >= minimum_contrast) & _smooth_path_mask(path_offsets)
    if int(np.count_nonzero(keep)) < max(45, int(sample_count * 0.65)):
        raise CalibrationRejected(
            "outer edge support too weak: {}/{} samples, median contrast {:.1f}".format(
                int(np.count_nonzero(keep)), sample_count, float(np.median(selected_evidence))
            )
        )
    points = bases + path_offsets[:, None] * normal[None, :]
    return points[keep], {
        "samples": float(np.count_nonzero(keep)),
        "coverage": float(np.count_nonzero(keep)) / float(sample_count),
        "median_contrast": float(np.median(selected_evidence[keep])),
    }


def _long_opposite_edges(corners: np.ndarray) -> tuple[np.ndarray, ...]:
    p0, p1, p2, p3 = np.asarray(corners, np.float64)
    pair_a = 0.5 * (np.linalg.norm(p1 - p0) + np.linalg.norm(p3 - p2))
    pair_b = 0.5 * (np.linalg.norm(p2 - p1) + np.linalg.norm(p0 - p3))
    if pair_a >= pair_b:
        return p0, p1, p3, p2
    return p1, p2, p0, p3


def find_divider_seed(gray: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the bright divider without assuming an exactly equal partition."""
    a0, a1, b0, b1 = _long_opposite_edges(corners)
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for fraction in np.linspace(0.22, 0.78, 113):
        start = a0 + fraction * (a1 - a0)
        end = b0 + fraction * (b1 - b0)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 80.0:
            continue
        tangent = vector / length
        normal = np.asarray([-tangent[1], tangent[0]], np.float64)
        stations = np.linspace(0.08, 0.92, 80)
        bases = start[None, :] + stations[:, None] * vector[None, :]
        strip = _remap_strip(gray, bases, normal, np.asarray([-12.0, 0.0, 12.0]))
        contrast = strip[:, 1] - 0.5 * (strip[:, 0] + strip[:, 2])
        # A real divider is bright through most of its length; a metal piece is
        # bright over only a short interval.  The 30th percentile is deliberate.
        score = float(np.percentile(contrast, 30.0) + 0.15 * np.median(contrast))
        if best is None or score > best[0]:
            best = (score, start, end, float(fraction))
    if best is None or best[0] < 18.0:
        raise CalibrationRejected(
            "white divider was not found (persistent contrast {:.1f})".format(
                -1.0 if best is None else best[0]
            )
        )
    return best[1], best[2], best[3]


def trace_divider(
    gray: np.ndarray,
    seed_start: np.ndarray,
    seed_end: np.ndarray,
    search_radius: int,
    sample_count: int,
    minimum_contrast: float,
) -> tuple[np.ndarray, dict[str, float]]:
    vector = np.asarray(seed_end - seed_start, np.float64)
    length = float(np.linalg.norm(vector))
    tangent = vector / max(1.0, length)
    normal = np.asarray([-tangent[1], tangent[0]], np.float64)
    stations = np.linspace(0.055, 0.945, sample_count, dtype=np.float64)
    bases = seed_start[None, :] + stations[:, None] * vector[None, :]
    offsets = np.arange(-search_radius, search_radius + 1, dtype=np.float64)
    samples = _remap_strip(gray, bases, normal, offsets)
    smooth = cv2.GaussianBlur(samples, (7, 1), 0.0)
    evidence = np.full_like(smooth, -1000.0, np.float64)
    flank = 10
    evidence[:, flank:-flank] = smooth[:, flank:-flank] - 0.5 * (
        smooth[:, :-2 * flank] + smooth[:, 2 * flank :]
    )
    path = _dynamic_trace(evidence, max_step=4)
    refined_index = _subpixel_peak(evidence, path)
    path_offsets = offsets[0] + refined_index
    selected_evidence = evidence[np.arange(sample_count), path]
    keep = (selected_evidence >= minimum_contrast) & _smooth_path_mask(path_offsets)
    if int(np.count_nonzero(keep)) < max(40, int(sample_count * 0.62)):
        raise CalibrationRejected(
            "divider support too weak: {}/{} samples, median contrast {:.1f}".format(
                int(np.count_nonzero(keep)), sample_count, float(np.median(selected_evidence))
            )
        )
    points = bases + path_offsets[:, None] * normal[None, :]
    return points[keep], {
        "samples": float(np.count_nonzero(keep)),
        "coverage": float(np.count_nonzero(keep)) / float(sample_count),
        "median_contrast": float(np.median(selected_evidence[keep])),
    }


def extract_plumb_lines(
    frame: np.ndarray,
    corners: np.ndarray,
    edge_search_radius: int = 72,
    divider_search_radius: int = 38,
    sample_count: int = 112,
) -> tuple[list[np.ndarray], list[str], dict[str, dict[str, float]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mild denoising retains the broad black/white transitions.
    gray = cv2.GaussianBlur(gray, (3, 3), 0.0)
    corners = order_cyclic(corners)
    centre = np.mean(corners, axis=0)
    groups: list[np.ndarray] = []
    names: list[str] = []
    diagnostics: dict[str, dict[str, float]] = {}
    for index in range(4):
        name = "outer_edge_{}".format(index + 1)
        points, quality = trace_outer_edge(
            gray,
            corners[index],
            corners[(index + 1) % 4],
            centre,
            edge_search_radius,
            sample_count,
            minimum_contrast=28.0,
        )
        groups.append(points)
        names.append(name)
        diagnostics[name] = quality
    seed_start, seed_end, fraction = find_divider_seed(gray, corners)
    divider, quality = trace_divider(
        gray,
        seed_start,
        seed_end,
        divider_search_radius,
        sample_count,
        minimum_contrast=24.0,
    )
    quality["seed_fraction"] = fraction
    groups.append(divider)
    names.append("divider")
    diagnostics["divider"] = quality
    return groups, names, diagnostics


def line_fit(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    centre = np.mean(points, axis=0)
    _u, _s, vt = np.linalg.svd(points - centre, full_matrices=False)
    direction = vt[0]
    normal = np.asarray([-direction[1], direction[0]], np.float64)
    distances = (points - centre) @ normal
    rms = float(math.sqrt(float(np.mean(np.square(distances)))))
    length = float(np.ptp((points - centre) @ direction))
    return centre, direction, rms, length


def combined_line_rms(groups: Sequence[np.ndarray]) -> tuple[float, list[float]]:
    squared_total = 0.0
    sample_total = 0
    per_group: list[float] = []
    for points in groups:
        _centre, _direction, rms, _length = line_fit(points)
        per_group.append(rms)
        squared_total += rms * rms * len(points)
        sample_total += len(points)
    return math.sqrt(squared_total / max(1, sample_total)), per_group


def make_camera_matrix(image_size: tuple[int, int], focal_scale: float) -> np.ndarray:
    width, height = image_size
    focal = float(max(width, height)) * float(focal_scale)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        np.float64,
    )


def radial_model_is_safe(
    camera_matrix: np.ndarray, image_size: tuple[int, int], k1: float, k2: float
) -> bool:
    width, height = image_size
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    radii = []
    for x, y in ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)):
        radii.append(math.hypot((x - cx) / fx, (y - cy) / fy))
    rmax = max(radii)
    samples = np.linspace(0.0, rmax, 128)
    r2 = np.square(samples)
    scale = 1.0 + k1 * r2 + k2 * np.square(r2)
    derivative = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * np.square(r2)
    return bool(np.min(scale) > 0.30 and np.min(derivative) > 0.12)


def undistort_groups(
    groups: Sequence[np.ndarray], camera_matrix: np.ndarray, k1: float, k2: float,
    projection_matrix: np.ndarray | None = None,
) -> list[np.ndarray]:
    lengths = [len(group) for group in groups]
    joined = np.concatenate(groups, axis=0).astype(np.float64).reshape(-1, 1, 2)
    distortion = np.asarray([k1, k2, 0.0, 0.0, 0.0], np.float64)
    corrected = cv2.undistortPoints(
        joined,
        camera_matrix,
        distortion,
        P=camera_matrix if projection_matrix is None else projection_matrix,
    ).reshape(-1, 2)
    result: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        result.append(corrected[cursor : cursor + length])
        cursor += length
    return result


def _objective(
    groups: Sequence[np.ndarray], camera_matrix: np.ndarray,
    image_size: tuple[int, int], k1: float, k2: float,
) -> float:
    if not radial_model_is_safe(camera_matrix, image_size, k1, k2):
        return float("inf")
    try:
        corrected = undistort_groups(groups, camera_matrix, k1, k2)
        rms, _per_group = combined_line_rms(corrected)
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return float("inf")
    # k2 is weakly observable from one frame; this tiny regularizer favours the
    # simpler radial curve when two solutions straighten the lines equally.
    return float(rms + 0.025 * abs(k2) + 0.004 * abs(k1))


def estimate_radial_coefficients(
    groups: Sequence[np.ndarray], image_size: tuple[int, int], focal_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = make_camera_matrix(image_size, focal_scale)
    best = (float("inf"), 0.0, 0.0)
    # Broad but bounded coarse search; ~1500 evaluations runs comfortably on a Pi.
    for k1 in np.linspace(-0.80, 0.20, 41):
        for k2 in np.linspace(-0.30, 0.60, 37):
            value = _objective(groups, camera_matrix, image_size, float(k1), float(k2))
            if value < best[0]:
                best = (value, float(k1), float(k2))
    _value, k1, k2 = best
    step1, step2 = 0.025, 0.025
    for _iteration in range(45):
        improved = False
        for delta1, delta2 in (
            (-step1, 0.0), (step1, 0.0), (0.0, -step2), (0.0, step2),
            (-step1, -step2), (-step1, step2), (step1, -step2), (step1, step2),
        ):
            candidate1 = float(np.clip(k1 + delta1, -0.85, 0.25))
            candidate2 = float(np.clip(k2 + delta2, -0.35, 0.65))
            value = _objective(groups, camera_matrix, image_size, candidate1, candidate2)
            if value + 1e-9 < best[0]:
                best = (value, candidate1, candidate2)
                k1, k2 = candidate1, candidate2
                improved = True
        if not improved:
            step1 *= 0.55
            step2 *= 0.55
        if max(step1, step2) < 2e-5:
            break
    distortion = np.asarray([k1, k2, 0.0, 0.0, 0.0], np.float64)
    return camera_matrix, distortion


def split_groups(groups: Sequence[np.ndarray], parity: int) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for group in groups:
        subset = np.asarray(group)[parity::2]
        if len(subset) < 12:
            raise CalibrationRejected("not enough samples for train/validation split")
        result.append(subset)
    return result


def evaluate_candidate(
    groups: Sequence[np.ndarray], camera_matrix: np.ndarray, distortion: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, object]:
    k1, k2 = float(distortion[0]), float(distortion[1])
    raw_rms, raw_per = combined_line_rms(groups)
    corrected = undistort_groups(groups, camera_matrix, k1, k2)
    corrected_rms, corrected_per = combined_line_rms(corrected)
    validation_raw = split_groups(groups, 1)
    validation_corrected = undistort_groups(validation_raw, camera_matrix, k1, k2)
    validation_raw_rms, validation_raw_per = combined_line_rms(validation_raw)
    validation_corrected_rms, validation_corrected_per = combined_line_rms(
        validation_corrected
    )
    ratio = validation_corrected_rms / max(1e-9, validation_raw_rms)
    absolute_drop = validation_raw_rms - validation_corrected_rms
    improved_groups = sum(
        after <= before * 0.82
        for before, after in zip(validation_raw_per, validation_corrected_per)
        if before >= 0.8
    )
    badly_worse = sum(
        after > before * 1.35 + 0.35
        for before, after in zip(validation_raw_per, validation_corrected_per)
    )
    safety = radial_model_is_safe(camera_matrix, image_size, k1, k2)
    reasons: list[str] = []
    if validation_raw_rms < 1.8:
        reasons.append("raw lines are already too straight to identify distortion")
    if absolute_drop < 1.20:
        reasons.append("validation RMS drop {:.2f}px is below 1.20px".format(absolute_drop))
    if ratio > 0.72:
        reasons.append("validation RMS ratio {:.3f} is above 0.72".format(ratio))
    if improved_groups < 3:
        reasons.append("fewer than three independently curved lines improved")
    if badly_worse:
        reasons.append("{} line group(s) became materially worse".format(badly_worse))
    if not safety:
        reasons.append("radial mapping is non-monotonic inside the image")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "raw_rms": raw_rms,
        "corrected_rms": corrected_rms,
        "raw_per_group": raw_per,
        "corrected_per_group": corrected_per,
        "validation_raw_rms": validation_raw_rms,
        "validation_corrected_rms": validation_corrected_rms,
        "validation_raw_per_group": validation_raw_per,
        "validation_corrected_per_group": validation_corrected_per,
        "validation_ratio": ratio,
        "validation_drop_px": absolute_drop,
        "improved_groups": improved_groups,
        "badly_worse_groups": badly_worse,
        "radial_mapping_safe": safety,
    }


def _draw_line_group(image: np.ndarray, points: np.ndarray, colour: tuple[int, int, int], name: str) -> None:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    rounded = np.rint(points).astype(np.int32)
    for point in rounded[:: max(1, len(rounded) // 35)]:
        cv2.circle(image, tuple(point), 2, colour, -1, cv2.LINE_AA)
    centre, direction, rms, length = line_fit(points)
    half = max(10.0, length * 0.52)
    a = tuple(np.rint(centre - half * direction).astype(int))
    b = tuple(np.rint(centre + half * direction).astype(int))
    cv2.line(image, a, b, colour, 1, cv2.LINE_AA)
    anchor = tuple(np.rint(points[len(points) // 2]).astype(int))
    cv2.putText(
        image, "{} {:.2f}px".format(name, rms), anchor,
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, colour, 1, cv2.LINE_AA,
    )


def make_preview(
    frame: np.ndarray, groups: Sequence[np.ndarray], names: Sequence[str],
    camera_matrix: np.ndarray, distortion: np.ndarray, report: dict[str, object],
) -> np.ndarray:
    height, width = frame.shape[:2]
    image_size = (width, height)
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, 1.0, image_size
    )
    corrected_image = cv2.undistort(frame, camera_matrix, distortion, None, new_matrix)
    corrected_groups = undistort_groups(
        groups, camera_matrix, float(distortion[0]), float(distortion[1]), new_matrix
    )
    raw = frame.copy()
    corrected = corrected_image.copy()
    colours = [(0, 255, 0), (255, 150, 0), (0, 180, 255), (255, 0, 220), (0, 255, 255)]
    for name, points, fixed, colour in zip(names, groups, corrected_groups, colours):
        _draw_line_group(raw, points, colour, name)
        _draw_line_group(corrected, fixed, colour, name)
    overlay_height = 58
    cv2.rectangle(raw, (0, 0), (width, overlay_height), (15, 15, 15), -1)
    cv2.rectangle(corrected, (0, 0), (width, overlay_height), (15, 15, 15), -1)
    cv2.putText(raw, "RAW plumb lines", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(
        raw, "RMS {:.2f}px".format(float(report["validation_raw_rms"])),
        (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA,
    )
    cv2.putText(corrected, "CANDIDATE UNDISTORTED", (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(
        corrected,
        "validation RMS {:.2f}px  ratio {:.3f}".format(
            float(report["validation_corrected_rms"]), float(report["validation_ratio"])
        ),
        (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA,
    )
    return np.concatenate([raw, corrected], axis=1)


def save_candidate(
    output: Path, camera_matrix: np.ndarray, distortion: np.ndarray,
    image_size: tuple[int, int], report: dict[str, object], names: Sequence[str],
    diagnostics: dict[str, dict[str, float]], image_path: Path, focal_scale: float,
) -> None:
    if output.name.lower() == FORMAL_CALIBRATION_NAME:
        raise CalibrationRejected(
            "refusing to write {}; choose a candidate filename".format(FORMAL_CALIBRATION_NAME)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        camera_matrix=np.asarray(camera_matrix, np.float64),
        dist_coeffs=np.asarray(distortion, np.float64),
        image_size=np.asarray(image_size, np.int32),
        calibration_method=np.asarray("single_frame_a4_plumb_line_candidate"),
        quality_gate_passed=np.asarray(True, np.bool_),
        mean_reprojection_error=np.asarray(report["validation_corrected_rms"], np.float64),
        line_rms_raw=np.asarray(report["raw_rms"], np.float64),
        line_rms_corrected=np.asarray(report["corrected_rms"], np.float64),
        validation_line_rms_raw=np.asarray(report["validation_raw_rms"], np.float64),
        validation_line_rms_corrected=np.asarray(report["validation_corrected_rms"], np.float64),
        validation_rms_ratio=np.asarray(report["validation_ratio"], np.float64),
        per_line_rms_raw=np.asarray(report["validation_raw_per_group"], np.float64),
        per_line_rms_corrected=np.asarray(report["validation_corrected_per_group"], np.float64),
        line_names=np.asarray(list(names), dtype=np.str_),
        line_sample_counts=np.asarray(
            [int(diagnostics[name]["samples"]) for name in names], np.int32
        ),
        line_median_contrast=np.asarray(
            [diagnostics[name]["median_contrast"] for name in names], np.float64
        ),
        focal_scale=np.asarray(focal_scale, np.float64),
        principal_point_fixed=np.asarray(True, np.bool_),
        source_files=np.asarray([str(image_path)], dtype=np.str_),
    )


def calibrate_frame(
    frame: np.ndarray, corners: np.ndarray, focal_scale: float = 0.55,
) -> tuple[list[np.ndarray], list[str], dict[str, dict[str, float]], np.ndarray, np.ndarray, dict[str, object]]:
    height, width = frame.shape[:2]
    groups, names, diagnostics = extract_plumb_lines(frame, corners)
    training = split_groups(groups, 0)
    camera_matrix, distortion = estimate_radial_coefficients(
        training, (width, height), focal_scale
    )
    report = evaluate_candidate(groups, camera_matrix, distortion, (width, height))
    return groups, names, diagnostics, camera_matrix, distortion, report


def synthetic_groups(
    image_size: tuple[int, int] = (1280, 720), k1: float = -0.31, k2: float = 0.09,
    noise_px: float = 0.12,
) -> tuple[list[np.ndarray], np.ndarray]:
    matrix = make_camera_matrix(image_size, 0.55)
    width, height = image_size
    lines = [
        np.column_stack([np.linspace(170, 910, 120), np.full(120, 90.0)]),
        np.column_stack([np.full(120, 170.0), np.linspace(90, 610, 120)]),
        np.column_stack([np.linspace(170, 910, 120), np.full(120, 610.0)]),
        np.column_stack([np.full(120, 910.0), np.linspace(90, 610, 120)]),
        np.column_stack([np.full(120, 520.0), np.linspace(90, 610, 120)]),
    ]
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    rng = np.random.default_rng(20260730)
    distorted: list[np.ndarray] = []
    for points in lines:
        x = (points[:, 0] - cx) / fx
        y = (points[:, 1] - cy) / fy
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        observed = np.column_stack([cx + fx * x * radial, cy + fy * y * radial])
        observed += rng.normal(0.0, noise_px, observed.shape)
        distorted.append(observed.astype(np.float64))
    return distorted, matrix


def self_test() -> int:
    groups, expected_matrix = synthetic_groups()
    training = split_groups(groups, 0)
    matrix, distortion = estimate_radial_coefficients(training, (1280, 720), 0.55)
    report = evaluate_candidate(groups, matrix, distortion, (1280, 720))
    if not bool(report["accepted"]):
        raise RuntimeError("synthetic distorted-line candidate was rejected: {}".format(report["reasons"]))
    if float(report["validation_ratio"]) > 0.20:
        raise RuntimeError("synthetic correction was too weak: {}".format(report["validation_ratio"]))
    if not np.allclose(matrix, expected_matrix):
        raise RuntimeError("fixed synthetic camera matrix changed")
    straight, _matrix = synthetic_groups(k1=0.0, k2=0.0, noise_px=0.08)
    straight_matrix, straight_distortion = estimate_radial_coefficients(
        split_groups(straight, 0), (1280, 720), 0.55
    )
    rejected = evaluate_candidate(straight, straight_matrix, straight_distortion, (1280, 720))
    if bool(rejected["accepted"]):
        raise RuntimeError("quality gate accepted already-straight synthetic lines")
    with tempfile.TemporaryDirectory(prefix="a4_plumb_line_test_") as directory:
        output = Path(directory) / DEFAULT_OUTPUT
        diagnostics = {
            name: {"samples": float(len(group)), "median_contrast": 100.0}
            for name, group in zip(
                ["outer_edge_1", "outer_edge_2", "outer_edge_3", "outer_edge_4", "divider"],
                groups,
            )
        }
        save_candidate(
            output, matrix, distortion, (1280, 720), report,
            list(diagnostics), diagnostics, Path("synthetic.png"), 0.55,
        )
        with np.load(str(output), allow_pickle=False) as saved:
            required = {"camera_matrix", "dist_coeffs", "image_size", "quality_gate_passed"}
            if not required.issubset(set(saved.files)):
                raise RuntimeError("candidate NPZ is missing compatible arrays")
    print("SELF_TEST OK")
    print("estimated k1={:.5f} k2={:.5f}".format(float(distortion[0]), float(distortion[1])))
    print(
        "validation RMS {:.3f}px -> {:.3f}px (ratio {:.3f})".format(
            float(report["validation_raw_rms"]),
            float(report["validation_corrected_rms"]),
            float(report["validation_ratio"]),
        )
    )
    print("straight-line rejection: {}".format("; ".join(rejected["reasons"])))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative single-frame A4 plumb-line lens calibration"
    )
    parser.add_argument("--image", type=Path, help="unannotated raw camera JPEG")
    parser.add_argument(
        "--a4-calibration", type=Path, default=Path("a4_calibration.json"),
        help="JSON containing physical_a4_corners_px",
    )
    parser.add_argument(
        "--corners", help="override JSON with x1,y1,...,x4,y4 (any cyclic/physical order)",
    )
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--preview", type=Path, default=Path(DEFAULT_PREVIEW))
    parser.add_argument(
        "--focal-scale", type=float, default=0.55,
        help="fixed fx=fy=max(image dimension)*scale; do not tune to chase RMS",
    )
    parser.add_argument("--force", action="store_true", help="replace existing candidate/preview")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.image is None:
        print("ERROR: --image is required (prefer the service /raw.jpg frame)", file=sys.stderr)
        return 2
    if not 0.35 <= args.focal_scale <= 0.85:
        print("ERROR: --focal-scale must be within [0.35, 0.85]", file=sys.stderr)
        return 2
    if args.output.name.lower() == FORMAL_CALIBRATION_NAME:
        print(
            "REJECTED: this fallback never writes {}; use {}".format(
                FORMAL_CALIBRATION_NAME, DEFAULT_OUTPUT
            ),
            file=sys.stderr,
        )
        return 3
    for path in (args.output, args.preview):
        if path.exists() and not args.force:
            print("ERROR: {} exists; use --force for candidate artifacts".format(path), file=sys.stderr)
            return 2
    try:
        frame = read_image(args.image)
        corners = parse_corners(args.corners) if args.corners else load_a4_corners(args.a4_calibration)
        groups, names, diagnostics, matrix, distortion, report = calibrate_frame(
            frame, corners, args.focal_scale
        )
        print("A4 plumb-line extraction")
        for name in names:
            quality = diagnostics[name]
            print(
                "  {}: {} samples, coverage {:.1%}, contrast {:.1f}".format(
                    name, int(quality["samples"]), quality["coverage"], quality["median_contrast"]
                )
            )
        print("Candidate k1={:.6f}, k2={:.6f}".format(float(distortion[0]), float(distortion[1])))
        print(
            "Validation straight-line RMS: {:.3f}px -> {:.3f}px; drop {:.3f}px; ratio {:.3f}".format(
                float(report["validation_raw_rms"]),
                float(report["validation_corrected_rms"]),
                float(report["validation_drop_px"]),
                float(report["validation_ratio"]),
            )
        )
        if not bool(report["accepted"]):
            raise CalibrationRejected("; ".join(str(reason) for reason in report["reasons"]))
        preview = make_preview(frame, groups, names, matrix, distortion, report)
        # Save only after every quality gate passes, and commit the NPZ last so
        # an interrupted run never leaves a candidate without its visual audit.
        write_image(args.preview, preview)
        save_candidate(
            args.output, matrix, distortion, (frame.shape[1], frame.shape[0]),
            report, names, diagnostics, args.image, args.focal_scale,
        )
    except CalibrationRejected as exc:
        print("REJECTED: {}".format(exc), file=sys.stderr)
        return 3
    print("ACCEPTED candidate: {}".format(args.output))
    print("Preview: {}".format(args.preview))
    print("Do not rename/promote it until the preview and A4 metric scale are checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
