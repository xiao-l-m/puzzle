#!/bin/sh
set -eu

APP_DIR=/home/wxm/28_puzzle_vision/raspberry_pi
PID_FILE="$APP_DIR/puzzle_vision.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "puzzle vision PID file is absent"
    exit 0
fi

pid=$(sed -n '1p' "$PID_FILE")
case "$pid" in
    *[!0-9]*|'') echo "invalid PID file" >&2; exit 1 ;;
esac

if ! kill -0 "$pid" 2>/dev/null; then
    echo "puzzle vision is not running"
    exit 0
fi

command_line=$(tr '\000' ' ' <"/proc/$pid/cmdline")
case "$command_line" in
    *puzzle_vision.py*) ;;
    *) echo "refusing to stop unrelated PID $pid: $command_line" >&2; exit 1 ;;
esac

kill "$pid"
i=0
while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 50 ]; do
    sleep 0.1
    i=$((i + 1))
done
if kill -0 "$pid" 2>/dev/null; then
    echo "puzzle vision is still stopping: PID $pid" >&2
    exit 1
fi
rm -f "$PID_FILE"
rm -f "$APP_DIR/puzzle_vision.mode"
echo "puzzle vision stopped: PID $pid"
