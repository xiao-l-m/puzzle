#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
cd "$APP_DIR"

if [ -x "$APP_DIR/.venv/bin/python" ]; then
    PYTHON="$APP_DIR/.venv/bin/python"
else
    PYTHON=python3
fi

exec "$PYTHON" integrated_puzzle_vision.py \
    --service-mode 2 \
    --mode2-white-only \
    --require-saved-a4 \
    --divider-y-mm 151.8 \
    --stable-frames 3 \
    --vision-wait-seconds 30 \
    --a4-calibration-file "$APP_DIR/a4_calibration.json" \
    --camera-calibration-file "$APP_DIR/camera_calibration.npz"
