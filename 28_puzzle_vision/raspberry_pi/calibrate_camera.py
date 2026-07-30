#!/usr/bin/env python3
"""Calibrate the fixed puzzle camera with a planar checkerboard.

Two acquisition modes are supported:

1. Batch images (recommended when the Raspberry Pi has no desktop):

   python3 calibrate_camera.py --images 'calibration_photos/*.jpg' \
       --cols 9 --rows 6 --square-mm 20

2. Live camera capture:

   python3 calibrate_camera.py --live --camera-device /dev/video0 \
       --cols 9 --rows 6 --square-mm 20

``--cols`` and ``--rows`` are the numbers of *inner corners*, not squares.
In the preview, SPACE records a detected board and Q finishes calibration.
For an SSH/headless session add ``--headless``; a sufficiently different valid
view is then recorded automatically.  Move/tilt the board between captures and
cover the centre, four sides and four corners of the image.

The output ``camera_calibration.npz`` contains OpenCV ``camera_matrix`` and
``dist_coeffs`` arrays plus image-size and error metadata.  The puzzle vision
pipeline should undistort every full-size frame before A4 corner detection and
perspective rectification.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable, Sequence

import cv2
import numpy as np


DEFAULT_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-DHZJ-240229-XH_Integrated_Webcam_HD-video-index0"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def checkerboard_object_points(
    cols: int, rows: int, square_mm: float
) -> np.ndarray:
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(square_mm)
    return points


def find_checkerboard(
    frame: np.ndarray, board_size: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners: np.ndarray | None = None

    # The SB detector is considerably more reliable near image edges and under
    # perspective.  Debian's older OpenCV builds may not provide it, so retain
    # the classic detector as a compatible fallback.
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
        flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
        found, corners = cv2.findChessboardCornersSB(gray, board_size, flags)
        if found and corners is not None:
            return True, np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found or corners is None:
        return False, None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray, np.asarray(corners, dtype=np.float32), (11, 11), (-1, -1), criteria
    )
    return True, refined


def view_descriptor(
    corners: np.ndarray, image_size: tuple[int, int]
) -> np.ndarray:
    """Describe position, size and orientation for live-view diversity checks."""
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    width, height = image_size
    center = np.mean(points, axis=0)
    span = np.ptp(points, axis=0)
    first_row_vector = points[-1] - points[0]
    angle = math.atan2(float(first_row_vector[1]), float(first_row_vector[0]))
    return np.asarray(
        [
            center[0] / max(1.0, width),
            center[1] / max(1.0, height),
            span[0] / max(1.0, width),
            span[1] / max(1.0, height),
            math.sin(angle),
            math.cos(angle),
        ],
        dtype=np.float64,
    )


def view_is_diverse(
    descriptor: np.ndarray,
    accepted: Sequence[np.ndarray],
    minimum_distance: float,
) -> bool:
    if not accepted:
        return True
    # Centre and board coverage are the most important terms.  The sine/cosine
    # representation avoids a discontinuity at +/-180 degrees.
    weights = np.asarray([1.4, 1.4, 1.0, 1.0, 0.35, 0.35], np.float64)
    return min(
        float(np.linalg.norm((descriptor - previous) * weights))
        for previous in accepted
    ) >= minimum_distance


def resolve_image_paths(inputs: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for token in inputs:
        candidate = Path(token).expanduser()
        matches: list[Path]
        if candidate.is_dir():
            matches = sorted(
                path
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = sorted(Path(path) for path in glob.glob(token, recursive=True))
        for path in matches:
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            resolved.append(path)
    return resolved


def read_image(path: Path) -> np.ndarray | None:
    # imdecode/fromfile also supports non-ASCII Windows paths, while behaving
    # like imread on the Raspberry Pi.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def collect_from_images(
    paths: Sequence[Path], board_size: tuple[int, int]
) -> tuple[list[np.ndarray], tuple[int, int], list[str]]:
    image_points: list[np.ndarray] = []
    sources: list[str] = []
    image_size: tuple[int, int] | None = None
    for path in paths:
        frame = read_image(path)
        if frame is None:
            print("SKIP unreadable: {}".format(path), file=sys.stderr)
            continue
        size = (int(frame.shape[1]), int(frame.shape[0]))
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(
                "SKIP size mismatch: {} is {}, expected {}".format(
                    path, size, image_size
                ),
                file=sys.stderr,
            )
            continue
        found, corners = find_checkerboard(frame, board_size)
        if not found or corners is None:
            print("MISS checkerboard: {}".format(path), file=sys.stderr)
            continue
        image_points.append(corners)
        sources.append(str(path))
        print("FOUND {}/{}: {}".format(len(image_points), len(paths), path))
    if image_size is None:
        raise RuntimeError("no readable calibration images")
    return image_points, image_size, sources


def open_camera(device: str, width: int, height: int, fps: int):
    source: int | str = int(device) if device.isdigit() else device
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(source)
    if not capture.isOpened() and source != 0:
        capture.release()
        print(
            "Camera {} unavailable; trying index 0".format(device),
            file=sys.stderr,
        )
        capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        raise RuntimeError("cannot open camera: {}".format(device))
    return capture


def collect_live(
    args: argparse.Namespace, board_size: tuple[int, int]
) -> tuple[list[np.ndarray], tuple[int, int], list[str]]:
    capture = open_camera(
        args.camera_device, args.width, args.height, args.camera_fps
    )
    capture_dir = Path(args.capture_dir).expanduser()
    capture_dir.mkdir(parents=True, exist_ok=True)
    image_points: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    sources: list[str] = []
    image_size: tuple[int, int] | None = None
    last_auto_capture = 0.0
    started = time.monotonic()
    print(
        "Live calibration: show a {}x{} inner-corner board; ".format(
            board_size[0], board_size[1]
        )
        + ("automatic headless capture enabled" if args.headless else "SPACE=capture, Q=finish")
    )
    try:
        while len(image_points) < args.samples:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera frame read failed")
            size = (int(frame.shape[1]), int(frame.shape[0]))
            if image_size is None:
                image_size = size
            elif size != image_size:
                raise RuntimeError(
                    "camera resolution changed from {} to {}".format(image_size, size)
                )
            found, corners = find_checkerboard(frame, board_size)
            descriptor = (
                None
                if not found or corners is None
                else view_descriptor(corners, image_size)
            )
            diverse = bool(
                descriptor is not None
                and view_is_diverse(descriptor, descriptors, args.diversity)
            )
            now = time.monotonic()
            capture_requested = False
            stop_requested = False

            if args.headless:
                capture_requested = bool(
                    found
                    and diverse
                    and now - last_auto_capture >= args.capture_interval
                )
                if args.max_seconds > 0.0 and now - started >= args.max_seconds:
                    stop_requested = True
            else:
                preview = frame.copy()
                if found and corners is not None:
                    cv2.drawChessboardCorners(preview, board_size, corners, True)
                message = "views {}/{}  {}".format(
                    len(image_points),
                    args.samples,
                    "READY - press SPACE" if found and diverse else (
                        "move/tilt board" if found else "board not found"
                    ),
                )
                cv2.putText(
                    preview,
                    message,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 0) if found and diverse else (0, 170, 255),
                    2,
                )
                try:
                    cv2.imshow("camera calibration", preview)
                except cv2.error as exc:
                    raise RuntimeError(
                        "OpenCV has no GUI support; rerun with --headless"
                    ) from exc
                key = cv2.waitKey(1) & 0xFF
                capture_requested = key == ord(" ")
                stop_requested = key in (ord("q"), ord("Q"), 27)

            if capture_requested:
                if not found or corners is None or descriptor is None:
                    print("Capture ignored: checkerboard not found")
                elif not diverse:
                    print("Capture ignored: view too similar; move or tilt the board")
                else:
                    index = len(image_points) + 1
                    destination = capture_dir / "calibration_{:02d}.jpg".format(index)
                    if not cv2.imwrite(str(destination), frame):
                        raise RuntimeError("cannot save {}".format(destination))
                    image_points.append(corners.copy())
                    descriptors.append(descriptor)
                    sources.append(str(destination))
                    last_auto_capture = now
                    print(
                        "CAPTURED {}/{}: {}".format(index, args.samples, destination),
                        flush=True,
                    )
            if stop_requested:
                break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    if image_size is None:
        raise RuntimeError("camera produced no frames")
    return image_points, image_size, sources


def calibrate(
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    cols: int,
    rows: int,
    square_mm: float,
    rational_model: bool = False,
) -> dict[str, np.ndarray | float | list[float]]:
    if len(image_points) < 3:
        raise RuntimeError("at least 3 valid views are required for calibration")
    object_template = checkerboard_object_points(cols, rows, square_mm)
    object_points = [object_template.copy() for _ in image_points]
    flags = cv2.CALIB_RATIONAL_MODEL if rational_model else 0
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        list(image_points),
        image_size,
        None,
        None,
        flags=flags,
    )
    per_view_errors: list[float] = []
    total_squared_error = 0.0
    total_points = 0
    for object_view, observed, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _jacobian = cv2.projectPoints(
            object_view, rvec, tvec, camera_matrix, dist_coeffs
        )
        difference = (
            np.asarray(observed, np.float64).reshape(-1, 2)
            - np.asarray(projected, np.float64).reshape(-1, 2)
        )
        squared = np.sum(difference * difference, axis=1)
        per_view_errors.append(float(math.sqrt(float(np.mean(squared)))))
        total_squared_error += float(np.sum(squared))
        total_points += int(len(squared))
    overall_error = math.sqrt(total_squared_error / max(1, total_points))
    if not (
        np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(dist_coeffs))
        and math.isfinite(overall_error)
    ):
        raise RuntimeError("calibration produced non-finite parameters")
    return {
        "camera_matrix": np.asarray(camera_matrix, np.float64),
        "dist_coeffs": np.asarray(dist_coeffs, np.float64),
        "rms_error": float(rms),
        "mean_reprojection_error": float(overall_error),
        "per_view_errors": per_view_errors,
    }


def save_calibration(
    output: Path,
    result: dict[str, np.ndarray | float | list[float]],
    image_size: tuple[int, int],
    cols: int,
    rows: int,
    square_mm: float,
    sources: Sequence[str],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        camera_matrix=np.asarray(result["camera_matrix"], np.float64),
        dist_coeffs=np.asarray(result["dist_coeffs"], np.float64),
        image_size=np.asarray(image_size, np.int32),
        board_size=np.asarray([cols, rows], np.int32),
        square_size_mm=np.asarray(square_mm, np.float64),
        rms_error=np.asarray(result["rms_error"], np.float64),
        mean_reprojection_error=np.asarray(
            result["mean_reprojection_error"], np.float64
        ),
        per_view_errors=np.asarray(result["per_view_errors"], np.float64),
        source_files=np.asarray(list(sources), dtype=np.str_),
    )


def print_report(
    output: Path,
    result: dict[str, np.ndarray | float | list[float]],
    image_size: tuple[int, int],
    view_count: int,
) -> None:
    camera_matrix = np.asarray(result["camera_matrix"])
    dist_coeffs = np.asarray(result["dist_coeffs"])
    errors = np.asarray(result["per_view_errors"], np.float64)
    print("\nCalibration complete")
    print("  output: {}".format(output))
    print("  image size: {} x {}".format(image_size[0], image_size[1]))
    print("  accepted views: {}".format(view_count))
    print("  OpenCV RMS: {:.4f} px".format(float(result["rms_error"])))
    print(
        "  reprojection RMSE: {:.4f} px (worst view {:.4f} px)".format(
            float(result["mean_reprojection_error"]),
            float(np.max(errors)) if len(errors) else float("nan"),
        )
    )
    print("  camera_matrix:\n{}".format(camera_matrix))
    print("  dist_coeffs: {}".format(dist_coeffs.reshape(-1)))
    if float(result["mean_reprojection_error"]) > 0.8:
        print(
            "WARNING: error is high. Retake sharp views covering the whole image "
            "with more board tilt.",
            file=sys.stderr,
        )


def synthetic_self_test() -> int:
    """Exercise calibration and NPZ serialization without images or a camera."""
    rng = np.random.default_rng(20260729)
    cols, rows, square_mm = 9, 6, 20.0
    image_size = (1280, 720)
    object_points = checkerboard_object_points(cols, rows, square_mm)
    expected_matrix = np.asarray(
        [[920.0, 0.0, 638.0], [0.0, 915.0, 356.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    expected_distortion = np.asarray([-0.18, 0.045, 0.001, -0.0008, 0.0])
    image_points: list[np.ndarray] = []
    for index in range(16):
        rvec = np.asarray(
            [
                -0.28 + 0.035 * index,
                0.20 * math.sin(index * 0.7),
                -0.18 + 0.025 * index,
            ],
            np.float64,
        )
        tvec = np.asarray(
            [
                -85.0 + 12.0 * (index % 5),
                -45.0 + 15.0 * (index % 4),
                640.0 + 18.0 * index,
            ],
            np.float64,
        )
        projected, _jacobian = cv2.projectPoints(
            object_points, rvec, tvec, expected_matrix, expected_distortion
        )
        noise = rng.normal(0.0, 0.06, projected.shape)
        image_points.append((projected + noise).astype(np.float32))
    result = calibrate(
        image_points, image_size, cols, rows, square_mm, rational_model=False
    )
    error = float(result["mean_reprojection_error"])
    if error > 0.25:
        raise RuntimeError("synthetic reprojection error too high: {:.4f}".format(error))
    with tempfile.TemporaryDirectory(prefix="camera_calibration_test_") as directory:
        destination = Path(directory) / "camera_calibration.npz"
        save_calibration(
            destination,
            result,
            image_size,
            cols,
            rows,
            square_mm,
            ["synthetic_{:02d}".format(i) for i in range(len(image_points))],
        )
        with np.load(str(destination)) as saved:
            required = {
                "camera_matrix",
                "dist_coeffs",
                "image_size",
                "mean_reprojection_error",
            }
            if not required.issubset(set(saved.files)):
                raise RuntimeError("NPZ output is missing required arrays")
            if tuple(int(value) for value in saved["image_size"]) != image_size:
                raise RuntimeError("NPZ image size round-trip failed")
    print("SELF_TEST OK")
    print("synthetic reprojection RMSE: {:.4f} px".format(error))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the fixed puzzle camera with a checkerboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Batch photos. Quote the glob so this script expands it consistently.
  python3 calibrate_camera.py --images 'calibration_photos/*.jpg' --cols 9 --rows 6 --square-mm 20

  # Live desktop capture: SPACE records a view, Q starts calibration.
  python3 calibrate_camera.py --live --camera-device /dev/video0

  # SSH/headless capture: move the checkerboard between automatic captures.
  python3 calibrate_camera.py --live --headless --samples 20 --max-seconds 180

  # No camera required; checks math, CLI-importability and NPZ serialization.
  python3 calibrate_camera.py --self-test
""",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--images",
        nargs="+",
        metavar="PATH_OR_GLOB",
        help="calibration image files, directories or glob patterns",
    )
    source.add_argument("--live", action="store_true", help="capture from camera")
    source.add_argument(
        "--self-test", action="store_true", help="run synthetic test without a camera"
    )
    parser.add_argument("--cols", type=int, default=9, help="inner corners per row")
    parser.add_argument("--rows", type=int, default=6, help="inner corners per column")
    parser.add_argument(
        "--square-mm", type=float, default=20.0, help="checker square side in mm"
    )
    parser.add_argument("--output", type=Path, default=Path("camera_calibration.npz"))
    parser.add_argument("--min-views", type=int, default=10)
    parser.add_argument("--rational-model", action="store_true")
    parser.add_argument("--camera-device", default=DEFAULT_CAMERA_DEVICE)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--capture-dir", default="calibration_captures")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture-interval", type=float, default=1.2)
    parser.add_argument(
        "--diversity",
        type=float,
        default=0.055,
        help="minimum normalized pose difference between live captures",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="headless capture timeout; 0 waits indefinitely",
    )
    args = parser.parse_args(argv)
    if args.cols < 3 or args.rows < 3:
        parser.error("--cols and --rows must both be at least 3 inner corners")
    if args.square_mm <= 0.0:
        parser.error("--square-mm must be positive")
    if args.min_views < 3:
        parser.error("--min-views must be at least 3")
    if args.samples < args.min_views:
        parser.error("--samples must be at least --min-views")
    if args.width <= 0 or args.height <= 0 or args.camera_fps <= 0:
        parser.error("camera width, height and FPS must be positive")
    if args.capture_interval < 0.0 or args.diversity < 0.0:
        parser.error("capture interval and diversity must be non-negative")
    if args.headless and not args.live:
        parser.error("--headless is only valid with --live")
    if not (args.images or args.live or args.self_test):
        parser.error("choose --images, --live or --self-test")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return synthetic_self_test()

    board_size = (int(args.cols), int(args.rows))
    if args.images:
        paths = resolve_image_paths(args.images)
        if not paths:
            raise RuntimeError("no image files matched --images")
        image_points, image_size, sources = collect_from_images(paths, board_size)
    else:
        image_points, image_size, sources = collect_live(args, board_size)
    if len(image_points) < args.min_views:
        raise RuntimeError(
            "only {} valid views; need at least {}".format(
                len(image_points), args.min_views
            )
        )
    result = calibrate(
        image_points,
        image_size,
        args.cols,
        args.rows,
        args.square_mm,
        rational_model=args.rational_model,
    )
    output = Path(args.output).expanduser()
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    save_calibration(
        output,
        result,
        image_size,
        args.cols,
        args.rows,
        args.square_mm,
        sources,
    )
    print_report(output, result, image_size, len(image_points))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCalibration cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
