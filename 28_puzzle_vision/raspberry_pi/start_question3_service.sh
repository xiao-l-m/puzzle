#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
PID_FILE="$APP_DIR/puzzle_vision.pid"
MODE_FILE="$APP_DIR/puzzle_vision.mode"
LOG_FILE="$APP_DIR/puzzle_question3.log"

sh "$APP_DIR/stop_puzzle_vision.sh"
cd "$APP_DIR"
nohup sh ./run_puzzle_vision.sh \
    --service-mode 3 \
    --divider-y-mm 151.8 \
    --stable-frames 3 \
    --vision-wait-seconds 30 \
    >"$LOG_FILE" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$PID_FILE"
printf '%s\n' '3' >"$MODE_FILE"
sleep 1
if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "question-3 service failed; inspect $LOG_FILE" >&2
    exit 1
fi
echo "question-3 unknown-shape service started: PID $new_pid, port 8081"
