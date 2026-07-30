#!/usr/bin/env python3
"""Offline audit and A4-corner conversion for a lens-calibration candidate.

This utility mirrors the exact production transform used by
``CameraUndistorter`` without importing or changing the running service:

1. ``cv2.getOptimalNewCameraMatrix(K, D, size, 1.0, size)``
2. ``cv2.initUndistortRectifyMap(..., new_matrix, size, ...)``
3. high-accuracy ``cv2.undistortPointsIter(..., P=new_matrix)`` to invert the
   production remap at the old corner locations

The old physical A4 corner order is preserved exactly.  Outputs are deliberately
named as candidates, and this script refuses to overwrite ``a4_calibration.json``
or its input JSON.  It also re-extracts all four A4 borders plus the white divider
from the raw source frame and verifies their geometry after conversion.

Example on the Raspberry Pi::

    python3 validate_a4_calibration_candidate.py \
      --candidate camera_calibration_candidate.npz \
      --a4-calibration a4_calibration.json \
      --image current_raw.jpg \
      --output-a4 a4_calibration_candidate_undistorted.json \
      --report a4_candidate_validation_report.json \
      --preview a4_candidate_conversion_preview.jpg

Nothing produced by this command is loaded by the live service automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from calibrate_from_a4_lines import (
    CalibrationRejected,
    combined_line_rms,
    extract_plumb_lines,
    line_fit,
    radial_model_is_safe,
    read_image,
    write_image,
)


FORMAL_A4_NAME = "a4_calibration.json"
EXPECTED_METHOD = "single_frame_a4_plumb_line_candidate"
DEFAULT_OUTPUT_A4 = "a4_calibration_candidate_undistorted.json"
DEFAULT_REPORT = "a4_candidate_validation_report.json"
DEFAULT_PREVIEW = "a4_candidate_conversion_preview.jpg"
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def _scalar(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise CalibrationRejected("{} must be a scalar".format(name))
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise CalibrationRejected("{} is not finite".format(name))
    return result


def _bool_scalar(value: np.ndarray, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise CalibrationRejected("{} must be a scalar".format(name))
    return bool(array.reshape(-1)[0])


def _string_scalar(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise CalibrationRejected("{} must be a scalar".format(name))
    return str(array.reshape(-1)[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_candidate(path: Path) -> dict[str, Any]:
    required = {
        "camera_matrix",
        "dist_coeffs",
        "image_size",
        "calibration_method",
        "quality_gate_passed",
        "validation_line_rms_raw",
        "validation_line_rms_corrected",
        "validation_rms_ratio",
        "per_line_rms_raw",
        "per_line_rms_corrected",
        "line_names",
        "line_sample_counts",
        "line_median_contrast",
        "focal_scale",
        "principal_point_fixed",
    }
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise CalibrationRejected(
                    "candidate NPZ is missing: {}".format(", ".join(sorted(missing)))
                )
            camera_matrix = np.asarray(archive["camera_matrix"], np.float64)
            distortion = np.asarray(archive["dist_coeffs"], np.float64).reshape(-1)
            image_size_values = np.asarray(archive["image_size"]).reshape(-1)
            method = _string_scalar(archive["calibration_method"], "calibration_method")
            quality_passed = _bool_scalar(archive["quality_gate_passed"], "quality_gate_passed")
            raw_rms = _scalar(archive["validation_line_rms_raw"], "validation_line_rms_raw")
            corrected_rms = _scalar(
                archive["validation_line_rms_corrected"],
                "validation_line_rms_corrected",
            )
            ratio = _scalar(archive["validation_rms_ratio"], "validation_rms_ratio")
            per_raw = np.asarray(archive["per_line_rms_raw"], np.float64).reshape(-1)
            per_corrected = np.asarray(
                archive["per_line_rms_corrected"], np.float64
            ).reshape(-1)
            names = [str(item) for item in np.asarray(archive["line_names"]).reshape(-1)]
            counts = np.asarray(archive["line_sample_counts"], np.int64).reshape(-1)
            contrasts = np.asarray(
                archive["line_median_contrast"], np.float64
            ).reshape(-1)
            focal_scale = _scalar(archive["focal_scale"], "focal_scale")
            principal_fixed = _bool_scalar(
                archive["principal_point_fixed"], "principal_point_fixed"
            )
            source_files = (
                [str(item) for item in np.asarray(archive["source_files"]).reshape(-1)]
                if "source_files" in archive
                else []
            )
    except CalibrationRejected:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise CalibrationRejected("cannot read candidate {}: {}".format(path, exc))

    reasons: list[str] = []
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        reasons.append("camera_matrix must be finite 3x3")
    if distortion.shape != (5,) or not np.all(np.isfinite(distortion)):
        reasons.append("plumb-line dist_coeffs must be five finite values")
    if image_size_values.size != 2:
        reasons.append("image_size must contain width,height")
        image_size = (0, 0)
    else:
        image_size = tuple(int(value) for value in image_size_values)
        if image_size[0] < 320 or image_size[1] < 240:
            reasons.append("image_size is implausibly small")
    if method != EXPECTED_METHOD:
        reasons.append("unexpected calibration_method {}".format(method))
    if not quality_passed:
        reasons.append("quality_gate_passed is false")
    if not principal_fixed:
        reasons.append("single-frame candidate must declare fixed principal point")
    if not 0.35 <= focal_scale <= 0.85:
        reasons.append("focal_scale is outside audited bounds")
    if camera_matrix.shape == (3, 3):
        if not np.allclose(camera_matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
            reasons.append("camera_matrix last row is invalid")
        if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
            reasons.append("focal lengths must be positive")
        if image_size != (0, 0):
            if abs(float(camera_matrix[0, 2]) - image_size[0] * 0.5) > 1e-6:
                reasons.append("principal x is not the fixed image centre")
            if abs(float(camera_matrix[1, 2]) - image_size[1] * 0.5) > 1e-6:
                reasons.append("principal y is not the fixed image centre")
    if distortion.shape == (5,):
        if np.max(np.abs(distortion[2:])) > 1e-10:
            reasons.append("single-frame candidate may only contain k1/k2")
        if camera_matrix.shape == (3, 3) and image_size != (0, 0):
            if not radial_model_is_safe(
                camera_matrix, image_size, float(distortion[0]), float(distortion[1])
            ):
                reasons.append("radial mapping is non-monotonic")
    if not (raw_rms >= 1.8 and corrected_rms <= raw_rms - 1.2):
        reasons.append("stored validation RMS lacks the required absolute improvement")
    if not (0.0 < ratio <= 0.72):
        reasons.append("stored validation_rms_ratio is outside (0,0.72]")
    calculated_ratio = corrected_rms / max(raw_rms, 1e-12)
    if abs(calculated_ratio - ratio) > 1e-6:
        reasons.append("stored validation ratio is internally inconsistent")
    expected_names = {
        "outer_edge_1", "outer_edge_2", "outer_edge_3", "outer_edge_4", "divider"
    }
    if set(names) != expected_names or len(names) != 5:
        reasons.append("candidate must contain four outer lines plus divider")
    if not (
        len(per_raw) == len(per_corrected) == len(counts) == len(contrasts) == len(names)
    ):
        reasons.append("per-line metadata lengths differ")
    elif not (
        np.all(np.isfinite(per_raw))
        and np.all(np.isfinite(per_corrected))
        and np.all(np.isfinite(contrasts))
    ):
        reasons.append("per-line metadata contains non-finite values")
    else:
        if int(np.min(counts)) < 40:
            reasons.append("a line has fewer than 40 supporting samples")
        if float(np.min(contrasts)) < 24.0:
            reasons.append("a line has insufficient black/white contrast")
        improved = int(np.count_nonzero(per_corrected <= per_raw * 0.82))
        if improved < 3:
            reasons.append("fewer than three stored line groups improved")
        if float(np.max(per_corrected)) > 2.5:
            reasons.append("a corrected line RMS exceeds 2.5px")
    if reasons:
        raise CalibrationRejected("; ".join(reasons))
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": distortion,
        "image_size": image_size,
        "calibration_method": method,
        "quality_gate_passed": quality_passed,
        "validation_line_rms_raw": raw_rms,
        "validation_line_rms_corrected": corrected_rms,
        "validation_rms_ratio": ratio,
        "per_line_rms_raw": per_raw,
        "per_line_rms_corrected": per_corrected,
        "line_names": names,
        "line_sample_counts": counts,
        "line_median_contrast": contrasts,
        "focal_scale": focal_scale,
        "principal_point_fixed": principal_fixed,
        "source_files": source_files,
    }


def load_a4_document(path: Path, expected_size: tuple[int, int]) -> tuple[dict[str, Any], np.ndarray]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        corners = np.asarray(document["physical_a4_corners_px"], np.float64)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CalibrationRejected("cannot load A4 JSON {}: {}".format(path, exc))
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise CalibrationRejected("physical_a4_corners_px must be finite 4x2")
    capture_size = tuple(int(value) for value in document.get("capture_size_px", []))
    if capture_size != expected_size:
        raise CalibrationRejected(
            "A4 capture_size_px {} != candidate image_size {}".format(
                capture_size, expected_size
            )
        )
    if document.get("camera_undistortion_enabled", False):
        raise CalibrationRejected(
            "input A4 corners already claim undistorted coordinates; refusing double conversion"
        )
    if document.get("order") != "A4_TOP_LEFT,TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT":
        raise CalibrationRejected("A4 physical corner order is missing or unexpected")
    return document, corners


def production_new_camera_matrix(candidate: dict[str, Any]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    matrix, roi = cv2.getOptimalNewCameraMatrix(
        candidate["camera_matrix"],
        candidate["dist_coeffs"],
        candidate["image_size"],
        1.0,
        candidate["image_size"],
    )
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise CalibrationRejected("getOptimalNewCameraMatrix returned invalid values")
    return np.asarray(matrix, np.float64), tuple(int(value) for value in roi)


def transform_points(
    points: np.ndarray, candidate: dict[str, Any], new_matrix: np.ndarray
) -> np.ndarray:
    source = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if hasattr(cv2, "undistortPointsIter"):
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            80,
            1e-12,
        )
        corrected = cv2.undistortPointsIter(
            source,
            candidate["camera_matrix"],
            candidate["dist_coeffs"],
            None,
            new_matrix,
            criteria,
        ).reshape(-1, 2)
    else:
        # Older OpenCV builds lack the public iterative overload.  The fallback
        # is still checked against the actual float remap below; a peripheral
        # approximation error greater than 0.15px is rejected.
        corrected = cv2.undistortPoints(
            source,
            candidate["camera_matrix"],
            candidate["dist_coeffs"],
            P=new_matrix,
        ).reshape(-1, 2)
    if not np.all(np.isfinite(corrected)):
        raise CalibrationRejected("corner undistortion returned non-finite coordinates")
    return corrected


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def verify_quad(corners: np.ndarray, image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    area = polygon_area(corners)
    cross_products: list[float] = []
    for index in range(4):
        a = corners[(index + 1) % 4] - corners[index]
        b = corners[(index + 2) % 4] - corners[(index + 1) % 4]
        cross_products.append(float(a[0] * b[1] - a[1] * b[0]))
    convex = all(value > 0.0 for value in cross_products) or all(
        value < 0.0 for value in cross_products
    )
    edge_lengths = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    margins = np.column_stack(
        [corners[:, 0], corners[:, 1], width - 1 - corners[:, 0], height - 1 - corners[:, 1]]
    )
    minimum_margin = float(np.min(margins))
    area_fraction = abs(area) / float(width * height)
    reasons: list[str] = []
    if not convex:
        reasons.append("transformed A4 quadrilateral is not convex")
    if area_fraction < 0.18:
        reasons.append("transformed A4 covers less than 18% of the frame")
    if min(edge_lengths) < 220.0:
        reasons.append("a transformed A4 edge is shorter than 220px")
    if minimum_margin < 2.0:
        reasons.append("a transformed A4 corner lies outside the safe image interior")
    if reasons:
        raise CalibrationRejected("; ".join(reasons))
    return {
        "signed_area_px2": area,
        "area_fraction": area_fraction,
        "convex": convex,
        "edge_lengths_px": edge_lengths,
        "minimum_frame_margin_px": minimum_margin,
    }


def remap_roundtrip_errors(
    source_points: np.ndarray,
    corrected_points: np.ndarray,
    candidate: dict[str, Any],
    new_matrix: np.ndarray,
) -> list[float]:
    width, height = candidate["image_size"]
    map_x, map_y = cv2.initUndistortRectifyMap(
        candidate["camera_matrix"],
        candidate["dist_coeffs"],
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    query_x = corrected_points[:, 0].astype(np.float32).reshape(-1, 1)
    query_y = corrected_points[:, 1].astype(np.float32).reshape(-1, 1)
    mapped_x = cv2.remap(map_x, query_x, query_y, cv2.INTER_LINEAR).reshape(-1)
    mapped_y = cv2.remap(map_y, query_x, query_y, cv2.INTER_LINEAR).reshape(-1)
    mapped = np.column_stack([mapped_x, mapped_y])
    errors = np.linalg.norm(mapped - source_points, axis=1)
    if float(np.max(errors)) > 0.15:
        raise CalibrationRejected(
            "corner transform disagrees with production remap by {:.3f}px".format(
                float(np.max(errors))
            )
        )
    return [float(value) for value in errors]


def physical_homography(corrected_corners: np.ndarray) -> np.ndarray:
    destination = np.asarray(
        [[0.0, 0.0], [A4_WIDTH_MM, 0.0], [A4_WIDTH_MM, A4_HEIGHT_MM], [0.0, A4_HEIGHT_MM]],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.asarray(corrected_corners, np.float32), destination
    )
    if not np.all(np.isfinite(matrix)):
        raise CalibrationRejected("A4 metric homography is non-finite")
    return matrix


def _line_intersection(
    centre_a: np.ndarray,
    direction_a: np.ndarray,
    centre_b: np.ndarray,
    direction_b: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack([direction_a, -direction_b])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 0.08:
        raise CalibrationRejected("adjacent corrected A4 edges are nearly parallel")
    parameters = np.linalg.solve(matrix, centre_b - centre_a)
    return centre_a + float(parameters[0]) * direction_a


def refine_corners_from_outer_lines(
    corrected_groups: Sequence[np.ndarray], direct_physical_corners: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """Intersect corrected borders, then preserve old TL/TR/BR/BL by proximity."""
    fits = [line_fit(points) for points in corrected_groups[:4]]
    geometric: list[np.ndarray] = []
    for index in range(4):
        previous = fits[(index - 1) % 4]
        current = fits[index]
        geometric.append(
            _line_intersection(previous[0], previous[1], current[0], current[1])
        )
    geometric_array = np.asarray(geometric, np.float64)
    best_cost = float("inf")
    best: np.ndarray | None = None
    second_cost = float("inf")
    for permutation in itertools.permutations(range(4)):
        assigned = geometric_array[list(permutation)]
        distances = np.linalg.norm(assigned - direct_physical_corners, axis=1)
        cost = float(np.sum(np.square(distances)))
        if cost < best_cost:
            second_cost = best_cost
            best_cost = cost
            best = assigned
        elif cost < second_cost:
            second_cost = cost
    if best is None:
        raise CalibrationRejected("could not assign corrected border intersections")
    shifts = np.linalg.norm(best - direct_physical_corners, axis=1)
    if float(np.max(shifts)) > 48.0:
        raise CalibrationRejected(
            "old A4 lock is too stale for safe line refinement (max {:.1f}px)".format(
                float(np.max(shifts))
            )
        )
    if second_cost < best_cost * 1.8:
        raise CalibrationRejected("physical corner assignment is ambiguous")
    return best, [float(value) for value in shifts]


def transform_perspective(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(-1, 1, 2), matrix
    ).reshape(-1, 2).astype(np.float64)


def image_geometry_audit(
    frame: np.ndarray,
    source_corners: np.ndarray,
    corrected_corners: np.ndarray,
    candidate: dict[str, Any],
    new_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], list[str], np.ndarray]:
    groups, names, extraction = extract_plumb_lines(frame, source_corners)
    corrected_groups = [
        transform_points(points, candidate, new_matrix) for points in groups
    ]
    corrected_rms, per_line_corrected = combined_line_rms(corrected_groups)
    raw_rms, per_line_raw = combined_line_rms(groups)
    stored = float(candidate["validation_line_rms_corrected"])
    # Re-extraction uses all samples rather than the saved odd holdout, so allow
    # a small deterministic difference while still detecting a mismatched frame.
    if corrected_rms > max(1.35, stored * 1.65):
        raise CalibrationRejected(
            "fresh corrected-line RMS {:.3f}px does not reproduce stored {:.3f}px".format(
                corrected_rms, stored
            )
        )
    metric_matrix = physical_homography(corrected_corners)
    metric_groups = [transform_perspective(points, metric_matrix) for points in corrected_groups]
    outer_boundary_rms: list[float] = []
    outer_boundary_names: list[str] = []
    for points in metric_groups[:4]:
        candidates = {
            "x=0": np.abs(points[:, 0]),
            "x=210": np.abs(points[:, 0] - A4_WIDTH_MM),
            "y=0": np.abs(points[:, 1]),
            "y=297": np.abs(points[:, 1] - A4_HEIGHT_MM),
        }
        boundary, distances = min(
            candidates.items(), key=lambda item: float(np.sqrt(np.mean(np.square(item[1]))))
        )
        outer_boundary_names.append(boundary)
        outer_boundary_rms.append(float(np.sqrt(np.mean(np.square(distances)))))
    if len(set(outer_boundary_names)) != 4:
        raise CalibrationRejected(
            "outer curves do not map one-to-one to all four physical A4 boundaries"
        )
    if max(outer_boundary_rms) > 2.0:
        raise CalibrationRejected(
            "outer-boundary metric RMS exceeds 2.0mm: {}".format(outer_boundary_rms)
        )
    divider = metric_groups[4]
    divider_y_mean = float(np.mean(divider[:, 1]))
    divider_y_rms = float(np.std(divider[:, 1]))
    divider_x_span = float(np.ptp(divider[:, 0]))
    if divider_y_rms > 0.75:
        raise CalibrationRejected(
            "divider is not horizontal in A4 coordinates (RMS {:.3f}mm)".format(
                divider_y_rms
            )
        )
    # The tracker deliberately excludes both outer-border junctions and may
    # reject a few specular/occluded end samples.  Seventy percent of 210mm is
    # still a strong full-width-line discriminator versus any metal piece.
    if divider_x_span < 147.0:
        raise CalibrationRejected(
            "divider covers only {:.1f}mm of the A4 width".format(divider_x_span)
        )
    corrected_frame = cv2.undistort(
        frame,
        candidate["camera_matrix"],
        candidate["dist_coeffs"],
        None,
        new_matrix,
    )
    return (
        {
            "fresh_line_rms_raw_px": raw_rms,
            "fresh_line_rms_corrected_px": corrected_rms,
            "fresh_line_rms_ratio": corrected_rms / max(raw_rms, 1e-12),
            "fresh_per_line_rms_raw_px": per_line_raw,
            "fresh_per_line_rms_corrected_px": per_line_corrected,
            "outer_boundary_assignment": outer_boundary_names,
            "outer_boundary_rms_mm": outer_boundary_rms,
            "divider_y_mean_mm": divider_y_mean,
            "divider_y_rms_mm": divider_y_rms,
            "divider_x_span_mm": divider_x_span,
            "extraction": extraction,
        },
        groups,
        corrected_groups,
        names,
        corrected_frame,
    )


def make_preview(
    raw: np.ndarray,
    corrected: np.ndarray,
    old_corners: np.ndarray,
    new_corners: np.ndarray,
    corrected_groups: Sequence[np.ndarray],
    names: Sequence[str],
    geometry: dict[str, Any],
) -> np.ndarray:
    left, right = raw.copy(), corrected.copy()
    physical_labels = ["TL", "TR", "BR", "BL"]
    for index, (old, new, label) in enumerate(zip(old_corners, new_corners, physical_labels)):
        colour = [(0, 0, 255), (0, 255, 0), (255, 80, 0), (0, 220, 255)][index]
        cv2.circle(left, tuple(np.rint(old).astype(int)), 7, colour, -1, cv2.LINE_AA)
        cv2.putText(left, label, tuple(np.rint(old + [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
        cv2.circle(right, tuple(np.rint(new).astype(int)), 7, colour, -1, cv2.LINE_AA)
        cv2.putText(right, label, tuple(np.rint(new + [8, -8]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
    cv2.polylines(left, [np.rint(old_corners).astype(np.int32)], True, (0,255,255), 2, cv2.LINE_AA)
    cv2.polylines(right, [np.rint(new_corners).astype(np.int32)], True, (0,255,255), 2, cv2.LINE_AA)
    colours = [(0,255,0), (255,150,0), (0,180,255), (255,0,220), (0,255,255)]
    for points, name, colour in zip(corrected_groups, names, colours):
        polyline = np.rint(points).astype(np.int32)
        cv2.polylines(right, [polyline], False, colour, 1, cv2.LINE_AA)
        anchor = tuple(polyline[len(polyline)//2])
        cv2.putText(right, name, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.40, colour, 1, cv2.LINE_AA)
    for image in (left, right):
        cv2.rectangle(image, (0,0), (image.shape[1], 62), (15,15,15), -1)
    cv2.putText(left, "RAW + OLD PHYSICAL A4 CORNERS", (15,27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(right, "PRODUCTION UNDISTORT + CONVERTED CORNERS", (15,27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(
        right,
        "fresh RMS {:.2f}px; divider y {:.2f}+/-{:.2f}mm".format(
            float(geometry["fresh_line_rms_corrected_px"]),
            float(geometry["divider_y_mean_mm"]),
            float(geometry["divider_y_rms_mm"]),
        ),
        (15,52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220,220,220), 1, cv2.LINE_AA,
    )
    return np.concatenate([left, right], axis=1)


def guard_output_path(path: Path, input_a4: Path) -> None:
    if path.name.lower() == FORMAL_A4_NAME:
        raise CalibrationRejected(
            "refusing to write {}; choose a candidate filename".format(FORMAL_A4_NAME)
        )
    try:
        if path.resolve() == input_a4.resolve():
            raise CalibrationRejected("refusing to overwrite the input A4 calibration")
    except OSError:
        pass


def build_candidate_a4_document(
    old_document: dict[str, Any],
    corrected_corners: np.ndarray,
    candidate_path: Path,
    candidate_sha256: str,
    input_a4: Path,
    input_a4_sha256: str,
    raw_image: Path,
    raw_image_sha256: str,
    new_matrix: np.ndarray,
) -> dict[str, Any]:
    return {
        "physical_a4_corners_px": [
            [round(float(point[0]), 4), round(float(point[1]), 4)]
            for point in corrected_corners
        ],
        "order": "A4_TOP_LEFT,TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT",
        "capture_size_px": [int(value) for value in old_document["capture_size_px"]],
        "source_region": old_document.get("source_region", "lower"),
        # This flag is required by PuzzleVisionApp._load_a4_calibration when a
        # CameraUndistorter is enabled.  The candidate remains offline until a
        # human deliberately promotes both files together.
        "camera_undistortion_enabled": True,
        "undistort_alpha": 1.0,
        "candidate_only": True,
        "camera_calibration_candidate": str(candidate_path),
        "camera_calibration_sha256": candidate_sha256,
        "source_a4_calibration": str(input_a4),
        "source_a4_calibration_sha256": input_a4_sha256,
        "source_raw_image": str(raw_image),
        "source_raw_image_sha256": raw_image_sha256,
        "production_new_camera_matrix": [
            [round(float(value), 10) for value in row] for row in new_matrix
        ],
        "corner_conversion": (
            "undistortPointsIter(old,80iter,alpha=1.0), then intersections of 4 freshly tracked corrected A4 borders"
        ),
        "note": "CANDIDATE ONLY: promote camera and A4 files together after preview/mm verification",
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def audit_and_convert(
    candidate_path: Path,
    a4_path: Path,
    image_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    candidate = load_candidate(candidate_path)
    old_document, old_corners = load_a4_document(a4_path, candidate["image_size"])
    frame = read_image(image_path)
    if (frame.shape[1], frame.shape[0]) != candidate["image_size"]:
        raise CalibrationRejected(
            "raw image size {} != candidate {}".format(
                (frame.shape[1], frame.shape[0]), candidate["image_size"]
            )
        )
    new_matrix, roi = production_new_camera_matrix(candidate)
    direct_corners = transform_points(old_corners, candidate, new_matrix)
    roundtrip = remap_roundtrip_errors(
        old_corners, direct_corners, candidate, new_matrix
    )
    # The saved lock can be a few pixels stale even with a fixed camera.  Use it
    # only to preserve physical TL/TR/BR/BL identity; obtain the actual corner
    # locations from intersections of the freshly tracked, corrected borders.
    raw_groups, _names, _extraction = extract_plumb_lines(frame, old_corners)
    first_corrected_groups = [
        transform_points(points, candidate, new_matrix) for points in raw_groups
    ]
    corrected_corners, refinement_shift = refine_corners_from_outer_lines(
        first_corrected_groups, direct_corners
    )
    quad_report = verify_quad(corrected_corners, candidate["image_size"])
    geometry, _raw_groups, corrected_groups, names, corrected_frame = image_geometry_audit(
        frame, old_corners, corrected_corners, candidate, new_matrix
    )
    displacement = np.linalg.norm(direct_corners - old_corners, axis=1)
    candidate_hash = sha256_file(candidate_path)
    a4_hash = sha256_file(a4_path)
    image_hash = sha256_file(image_path)
    output_a4 = build_candidate_a4_document(
        old_document,
        corrected_corners,
        candidate_path,
        candidate_hash,
        a4_path,
        a4_hash,
        image_path,
        image_hash,
        new_matrix,
    )
    report = {
        "accepted": True,
        "offline_only": True,
        "candidate_file": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "source_a4_file": str(a4_path),
        "source_a4_sha256": a4_hash,
        "source_raw_image": str(image_path),
        "source_raw_image_sha256": image_hash,
        "runtime_schema": {
            "camera_matrix_shape": list(candidate["camera_matrix"].shape),
            "dist_coeffs_shape": list(candidate["dist_coeffs"].shape),
            "image_size": list(candidate["image_size"]),
            "calibration_method": candidate["calibration_method"],
            "quality_gate_passed": candidate["quality_gate_passed"],
            "validation_line_rms_raw_px": candidate["validation_line_rms_raw"],
            "validation_line_rms_corrected_px": candidate[
                "validation_line_rms_corrected"
            ],
            "validation_rms_ratio": candidate["validation_rms_ratio"],
            "focal_scale": candidate["focal_scale"],
            "principal_point_fixed": candidate["principal_point_fixed"],
        },
        "camera_matrix": candidate["camera_matrix"].tolist(),
        "dist_coeffs": candidate["dist_coeffs"].tolist(),
        "production_new_camera_matrix": new_matrix.tolist(),
        "production_valid_roi": list(roi),
        "source_physical_a4_corners_px": old_corners.tolist(),
        "direct_undistorted_physical_a4_corners_px": direct_corners.tolist(),
        "converted_physical_a4_corners_px": corrected_corners.tolist(),
        "corner_displacement_px": displacement.tolist(),
        "max_corner_displacement_px": float(np.max(displacement)),
        "border_intersection_refinement_px": refinement_shift,
        "max_border_intersection_refinement_px": max(refinement_shift),
        "production_map_roundtrip_error_px": roundtrip,
        "max_production_map_roundtrip_error_px": max(roundtrip),
        "converted_quad": quad_report,
        "fresh_image_geometry": geometry,
        "promotion_preconditions": [
            "visually inspect preview: all physical TL/TR/BR/BL labels must remain correct",
            "verify divider_y_mean_mm and a known ruler/feature in the rectified A4 view",
            "promote camera candidate and converted A4 candidate together, never one alone",
            "restart service and require four stable pieces plus plausible A4 coordinates before motor power",
        ],
    }
    preview = make_preview(
        frame,
        corrected_frame,
        old_corners,
        corrected_corners,
        corrected_groups,
        names,
        geometry,
    )
    return output_a4, report, preview


def synthetic_self_test() -> int:
    image_size = (1280, 720)
    camera_matrix = np.asarray(
        [[704.0, 0.0, 640.0], [0.0, 704.0, 360.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    distortion = np.asarray([-0.285, 0.077, 0.0, 0.0, 0.0], np.float64)
    new_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, 1.0, image_size
    )
    ideal_normalized = np.asarray(
        [[-0.65, 0.42], [-0.63, -0.41], [0.38, -0.40], [0.40, 0.43]],
        np.float64,
    )
    r2 = np.sum(np.square(ideal_normalized), axis=1)
    radial = 1.0 + distortion[0] * r2 + distortion[1] * np.square(r2)
    distorted = np.column_stack(
        [
            camera_matrix[0, 2] + camera_matrix[0, 0] * ideal_normalized[:, 0] * radial,
            camera_matrix[1, 2] + camera_matrix[1, 1] * ideal_normalized[:, 1] * radial,
        ]
    )
    expected = np.column_stack(
        [
            new_matrix[0, 2] + new_matrix[0, 0] * ideal_normalized[:, 0],
            new_matrix[1, 2] + new_matrix[1, 1] * ideal_normalized[:, 1],
        ]
    )
    candidate = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": distortion,
        "image_size": image_size,
    }
    converted = transform_points(distorted, candidate, new_matrix)
    maximum = float(np.max(np.linalg.norm(converted - expected, axis=1)))
    if maximum > 0.002:
        raise RuntimeError("synthetic corner conversion error {:.6f}px".format(maximum))
    roundtrip = remap_roundtrip_errors(distorted, converted, candidate, new_matrix)
    if max(roundtrip) > 0.15:
        raise RuntimeError("synthetic production-map roundtrip failed")
    print("SELF_TEST OK")
    print("max corner conversion error {:.6f}px".format(maximum))
    print("max production-map roundtrip error {:.6f}px".format(max(roundtrip)))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline audit and old-A4-corner conversion for a lens candidate"
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--a4-calibration", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-a4", type=Path, default=Path(DEFAULT_OUTPUT_A4))
    parser.add_argument("--report", type=Path, default=Path(DEFAULT_REPORT))
    parser.add_argument("--preview", type=Path, default=Path(DEFAULT_PREVIEW))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return synthetic_self_test()
    if args.candidate is None or args.a4_calibration is None or args.image is None:
        print("ERROR: --candidate, --a4-calibration and --image are required", file=sys.stderr)
        return 2
    try:
        guard_output_path(args.output_a4, args.a4_calibration)
        for path in (args.output_a4, args.report, args.preview):
            if path.exists() and not args.force:
                raise CalibrationRejected(
                    "{} exists; use --force only for candidate artifacts".format(path)
                )
        output_a4, report, preview = audit_and_convert(
            args.candidate, args.a4_calibration, args.image
        )
        # All checks run before any output is created.  Commit the human-audit
        # preview/report first and the candidate A4 JSON last.
        write_image(args.preview, preview)
        write_json(args.report, report)
        write_json(args.output_a4, output_a4)
    except CalibrationRejected as exc:
        print("REJECTED: {}".format(exc), file=sys.stderr)
        return 3
    print("ACCEPTED OFFLINE CONVERSION")
    print("  candidate A4: {}".format(args.output_a4))
    print("  report: {}".format(args.report))
    print("  preview: {}".format(args.preview))
    print(
        "  corners: {}".format(
            json.dumps(output_a4["physical_a4_corners_px"], ensure_ascii=False)
        )
    )
    print("No live-service file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
