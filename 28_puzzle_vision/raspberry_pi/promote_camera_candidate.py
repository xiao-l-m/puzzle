#!/usr/bin/env python3
"""Atomically promote an already accepted camera/A4 candidate pair.

This tool is intentionally separate from calibration and validation.  It
requires an accepted full-pipeline report, verifies hashes again, backs up the
live files, stages both replacements, then replaces camera first and A4 second
so any interrupted operation fails closed on the next service start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

from puzzle_vision import CameraUndistorter
from validate_candidate_puzzle_pipeline import PipelineRejected, load_candidate_a4


def load_report(path: Path, camera_hash: str) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineRejected(f"cannot load pipeline report {path}: {exc}") from exc
    validation = report.get("pipeline_validation") or {}
    if validation.get("accepted") is not True or validation.get("problems"):
        raise PipelineRejected("full puzzle pipeline report is not accepted")
    calibration = report.get("camera_calibration") or {}
    if calibration.get("loaded") is not True:
        raise PipelineRejected("report did not load the camera candidate")
    if calibration.get("file_sha256") != camera_hash:
        raise PipelineRejected("report belongs to a different camera candidate")
    for mode in ("1", "2"):
        plan = (report.get("motion_plans") or {}).get(mode) or {}
        if plan.get("execution_ready") is not True:
            raise PipelineRejected(f"report mode {mode} is not execution-ready")
        if len(plan.get("moves") or []) != 4:
            raise PipelineRejected(f"report mode {mode} does not have four moves")
    return report


def stage_bytes(path: Path, data: bytes) -> Path:
    temporary = path.with_name(path.name + ".promote-tmp")
    temporary.write_bytes(data)
    return temporary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a validated camera/A4 pair")
    parser.add_argument("--camera-candidate", type=Path, required=True)
    parser.add_argument("--a4-candidate", type=Path, required=True)
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--live-camera", type=Path, default=Path("camera_calibration.npz"))
    parser.add_argument("--live-a4", type=Path, default=Path("a4_calibration.json"))
    parser.add_argument("--backup-root", type=Path, default=Path("calibration_backups"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.camera_candidate.resolve() == args.live_camera.resolve():
            raise PipelineRejected("camera candidate is already the live file")
        if args.a4_candidate.resolve() == args.live_a4.resolve():
            raise PipelineRejected("A4 candidate is already the live file")
        loader = CameraUndistorter(str(args.camera_candidate), 1280, 720)
        if not loader.enabled or loader.file_sha256 is None:
            raise PipelineRejected(f"camera candidate rejected: {loader.error}")
        load_candidate_a4(args.a4_candidate, loader.file_sha256)
        report = load_report(args.pipeline_report, loader.file_sha256)
        a4_payload = json.loads(args.a4_candidate.read_text(encoding="utf-8"))
        a4_payload["candidate_only"] = False
        a4_payload["camera_calibration_file"] = str(args.live_camera)
        a4_payload["promoted_at_utc"] = datetime.now(timezone.utc).isoformat()
        a4_payload["promotion_report"] = str(args.pipeline_report)
        a4_data = (json.dumps(a4_payload, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        camera_data = args.camera_candidate.read_bytes()
        if args.dry_run:
            print("DRY RUN ACCEPTED")
            print(f"camera sha256: {loader.file_sha256}")
            print(f"divider y: {report.get('divider_y_mm')} mm")
            return 0

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.backup_root / stamp
        backup.mkdir(parents=True, exist_ok=False)
        if args.live_camera.exists():
            shutil.copy2(args.live_camera, backup / args.live_camera.name)
        if args.live_a4.exists():
            shutil.copy2(args.live_a4, backup / args.live_a4.name)
        shutil.copy2(args.camera_candidate, backup / args.camera_candidate.name)
        shutil.copy2(args.a4_candidate, backup / args.a4_candidate.name)
        shutil.copy2(args.pipeline_report, backup / args.pipeline_report.name)

        args.live_camera.parent.mkdir(parents=True, exist_ok=True)
        args.live_a4.parent.mkdir(parents=True, exist_ok=True)
        staged_camera = stage_bytes(args.live_camera, camera_data)
        staged_a4 = stage_bytes(args.live_a4, a4_data)
        # Camera first: if interrupted before A4 replacement, the production
        # hash guard rejects old A4 corners and motion remains disabled.
        os.replace(staged_camera, args.live_camera)
        os.replace(staged_a4, args.live_a4)
    except (PipelineRejected, OSError, ValueError, KeyError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 3
    print("PROMOTED CAMERA/A4 PAIR")
    print(f"backup: {backup}")
    print(f"camera sha256: {loader.file_sha256}")
    print("Restart the vision service, then verify four stable pieces before motor power.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
