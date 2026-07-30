#!/usr/bin/env python3
"""Offline end-to-end audit of a camera/A4 candidate against the puzzle solver.

The script never changes ``camera_calibration.npz`` or ``a4_calibration.json``.
It applies the exact production remap, runs enough identical-frame iterations
to exercise all multi-frame stability gates, builds both motion modes, and
writes candidate-only images/report for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from puzzle_motion import StageCalibration, build_task_plan, estimate_plan_seconds
from puzzle_vision import CameraUndistorter, DetectorConfig, PuzzleDetector


class PipelineRejected(RuntimeError):
    pass


def read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(str(path), np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise PipelineRejected(f"cannot read {path}: {exc}") from exc
    if frame is None or frame.ndim != 3:
        raise PipelineRejected(f"cannot decode {path}")
    return frame


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise PipelineRejected(f"cannot encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def load_candidate_a4(path: Path, expected_hash: str) -> np.ndarray:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        corners = np.asarray(payload["physical_a4_corners_px"], np.float32)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PipelineRejected(f"cannot load candidate A4 {path}: {exc}") from exc
    reasons: list[str] = []
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        reasons.append("candidate A4 corners must be finite 4x2")
    if payload.get("order") != "A4_TOP_LEFT,TOP_RIGHT,BOTTOM_RIGHT,BOTTOM_LEFT":
        reasons.append("physical A4 corner order is missing")
    if payload.get("camera_undistortion_enabled") is not True:
        reasons.append("candidate A4 is not marked as undistorted")
    if payload.get("camera_calibration_sha256") != expected_hash:
        reasons.append("candidate A4 hash does not match camera candidate")
    if tuple(payload.get("capture_size_px", [])) != (1280, 720):
        reasons.append("candidate A4 capture size is not 1280x720")
    if payload.get("source_region") != "lower":
        reasons.append("source_region must remain lower")
    if reasons:
        raise PipelineRejected("; ".join(reasons))
    return corners


def add_motion_plans(status: dict[str, Any]) -> dict[str, Any]:
    # PuzzleVisionApp.vision_status() adds this runtime-only guard after the
    # detector returns.  Mirror it here before invoking the same planner.
    status["a4_physical_orientation_locked"] = bool(
        status.get("a4_orientation") == "locked_physical_a4"
    )
    calibration = StageCalibration()
    plans: dict[str, Any] = {}
    for mode in (1, 2):
        plan = build_task_plan(mode, status, calibration)
        seconds = estimate_plan_seconds(plan, calibration)
        plan["estimated_seconds"] = seconds
        plan["within_120_seconds"] = bool(plan.get("ready") and seconds <= 120.0)
        plan["safe_time_budget"] = bool(plan.get("ready") and seconds <= 105.0)
        plan["execution_ready"] = bool(plan.get("ready") and seconds <= 105.0)
        plans[str(mode)] = plan
    status["motion_plans"] = plans
    return status


def validate_status(status: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not status.get("a4_found"):
        problems.append("A4_NOT_FOUND")
    if status.get("a4_orientation") != "locked_physical_a4":
        problems.append("A4_PHYSICAL_ORIENTATION_NOT_LOCKED")
    if not status.get("divider_found"):
        problems.append("DIVIDER_NOT_FOUND")
    if int(status.get("divider_samples", 0)) < 12:
        problems.append("DIVIDER_NOT_STABLE_FOR_12_FRAMES")
    if float(status.get("divider_spread_px", 999.0)) > 1.0:
        problems.append("DIVIDER_SPREAD_OVER_1PX")
    if int(status.get("piece_count", 0)) != 4:
        problems.append("NEED_EXACTLY_FOUR_PIECES")
    if not status.get("stable_four_pieces"):
        problems.append("FOUR_PIECES_NOT_STABLE")
    if not (status.get("piece_set_quality") or {}).get("valid"):
        problems.append("PIECE_SET_QUALITY_FAILED")

    divider_y = float(status.get("divider_y_mm") or -1.0)
    for piece in status.get("pieces") or []:
        center = piece.get("center_mm") or [0.0, 0.0]
        if float(center[1]) <= divider_y:
            problems.append(f"{piece.get('id', '?')}_NOT_IN_LOWER_SOURCE")

    assembly = status.get("assembly_plan") or {}
    if not assembly.get("ready"):
        problems.append("ASSEMBLY_SOLUTION_NOT_READY")
    if int(assembly.get("measurement_samples", 0)) < 8:
        problems.append("ASSEMBLY_HAS_FEWER_THAN_8_SAMPLES")
    if float(assembly.get("position_spread_mm", 999.0)) > 0.8:
        problems.append("ASSEMBLY_POSITION_SPREAD_OVER_0.8MM")
    if float(assembly.get("angle_spread_deg", 999.0)) > 1.5:
        problems.append("ASSEMBLY_ANGLE_SPREAD_OVER_1.5DEG")
    for move in assembly.get("moves") or []:
        if float(move.get("shape_residual", 999.0)) > 0.35:
            problems.append(f"{move.get('piece_id', '?')}_SHAPE_RESIDUAL_OVER_0.35")

    for mode in ("1", "2"):
        plan = (status.get("motion_plans") or {}).get(mode) or {}
        if not plan.get("execution_ready"):
            problems.append(f"MODE_{mode}_NOT_EXECUTION_READY:{plan.get('error')}")
        if len(plan.get("moves") or []) != 4:
            problems.append(f"MODE_{mode}_DOES_NOT_HAVE_4_MOVES")
        if plan.get("unreachable"):
            problems.append(f"MODE_{mode}_HAS_UNREACHABLE_POINTS")
        if float(plan.get("estimated_seconds", 999.0)) > 105.0:
            problems.append(f"MODE_{mode}_EXCEEDS_105_SECONDS")
    return problems


def audit(
    image_path: Path,
    camera_path: Path,
    a4_path: Path,
    iterations: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = read_image(image_path)
    height, width = frame.shape[:2]
    undistorter = CameraUndistorter(str(camera_path), width, height)
    if not undistorter.enabled:
        raise PipelineRejected(f"camera candidate rejected: {undistorter.error}")
    if undistorter.file_sha256 is None:
        raise PipelineRejected("camera candidate hash is unavailable")
    corners = load_candidate_a4(a4_path, undistorter.file_sha256)
    corrected = undistorter.apply(frame)
    detector = PuzzleDetector(
        DetectorConfig(
            contrast=28.0,
            min_area_mm2=100.0,
            max_area_mm2=4500.0,
            stable_frames=5,
            stable_center_mm=2.0,
            paper_mode="auto",
            divider_y_mm=0.0,
            source_region="lower",
        )
    )
    status: dict[str, Any] = {}
    camera_view = corrected
    warped = np.empty((1, 1, 3), np.uint8)
    mask = np.empty((1, 1), np.uint8)
    for _index in range(iterations):
        status, camera_view, warped, mask = detector.detect(corrected, corners)
    status = add_motion_plans(status)
    status["camera_calibration"] = undistorter.status()
    status["offline_iterations"] = iterations
    problems = validate_status(status)
    status["pipeline_validation"] = {
        "accepted": not problems,
        "problems": problems,
        "offline_only": True,
    }
    if problems:
        raise PipelineRejected("; ".join(problems))
    return status, frame, camera_view, warped, mask


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline full puzzle-pipeline validation for calibration candidates"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--camera-candidate", type=Path, required=True)
    parser.add_argument("--a4-candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations < 12 or args.iterations > 60:
        print("REJECTED: --iterations must be in [12,60]", file=sys.stderr)
        return 3
    output = args.output_dir
    artifacts = {
        "report": output / "candidate_pipeline_report.json",
        "raw": output / "sensor_raw.jpg",
        "camera": output / "camera_undistorted_annotated.jpg",
        "a4": output / "a4_rectified_annotated.jpg",
        "mask": output / "piece_mask.png",
    }
    if not args.force and any(path.exists() for path in artifacts.values()):
        print("REJECTED: output exists; use --force for offline artifacts", file=sys.stderr)
        return 3
    try:
        status, raw, camera, a4, mask = audit(
            args.image,
            args.camera_candidate,
            args.a4_candidate,
            args.iterations,
        )
        output.mkdir(parents=True, exist_ok=True)
        write_image(artifacts["raw"], raw)
        write_image(artifacts["camera"], camera)
        write_image(artifacts["a4"], a4)
        write_image(artifacts["mask"], mask)
        artifacts["report"].write_text(
            json.dumps(status, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except PipelineRejected as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 3
    print("ACCEPTED OFFLINE PUZZLE PIPELINE")
    print(json.dumps({
        "divider_y_mm": status["divider_y_mm"],
        "pieces": [piece["center_mm"] for piece in status["pieces"]],
        "mode1_seconds": status["motion_plans"]["1"]["estimated_seconds"],
        "mode2_seconds": status["motion_plans"]["2"]["estimated_seconds"],
        "output": str(output),
    }, ensure_ascii=False))
    print("No live-service file was modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
