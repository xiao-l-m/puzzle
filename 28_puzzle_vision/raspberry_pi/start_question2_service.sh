#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
PID_FILE="$APP_DIR/puzzle_vision.pid"
MODE_FILE="$APP_DIR/puzzle_vision.mode"
LOG_FILE="$APP_DIR/puzzle_question2.log"

sh "$APP_DIR/stop_puzzle_vision.sh"
# Use the current fixed-template detector.  The older clean checkout does not
# restore the fixed-camera A4 calibration at startup and its automatic divider
# detector mistakes a nearby piece edge for the physical centre line.  Both
# errors were reproduced on the live camera and make the four contours flicker.
nohup sh "$APP_DIR/run_question2_service.sh" \
    >"$LOG_FILE" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$PID_FILE"
printf '%s\n' '2' >"$MODE_FILE"
sleep 1
if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "question-2 service failed; inspect $LOG_FILE" >&2
    exit 1
fi
echo "question-2 fixed-template service started: PID $new_pid, port 8081"
