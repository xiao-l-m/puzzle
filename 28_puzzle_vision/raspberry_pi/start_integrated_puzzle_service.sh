#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
PID_FILE="$APP_DIR/puzzle_vision.pid"
MODE_FILE="$APP_DIR/puzzle_vision.mode"
LOG_FILE="$APP_DIR/puzzle_integrated.log"

sh "$APP_DIR/stop_puzzle_vision.sh"
cd "$APP_DIR"
nohup sh ./run_integrated_puzzle_vision.sh \
    >"$LOG_FILE" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$PID_FILE"
printf '%s\n' 'integrated-1-2-3' >"$MODE_FILE"
sleep 1
if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "integrated service failed; inspect $LOG_FILE" >&2
    exit 1
fi
echo "integrated question 1/2/3 service started: PID $new_pid, port 8081"
