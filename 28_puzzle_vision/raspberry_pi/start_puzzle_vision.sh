#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
PID_FILE="$APP_DIR/puzzle_vision.pid"
LOG_FILE="$APP_DIR/puzzle_vision.log"

if [ -f "$PID_FILE" ]; then
    old_pid=$(sed -n '1p' "$PID_FILE")
    case "$old_pid" in
        *[!0-9]*|'') old_pid=0 ;;
    esac
    if [ "$old_pid" -gt 0 ] && kill -0 "$old_pid" 2>/dev/null; then
        echo "puzzle vision already running: PID $old_pid"
        exit 0
    fi
fi

cd "$APP_DIR"
nohup ./run_puzzle_vision.sh >"$LOG_FILE" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$PID_FILE"
echo "puzzle vision started: PID $new_pid"

