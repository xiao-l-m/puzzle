#!/bin/sh
set -eu

cd /home/wxm/28_puzzle_vision/raspberry_pi
if [ -x .venv/bin/python ]; then
    exec .venv/bin/python puzzle_vision.py "$@"
fi
exec python3 puzzle_vision.py "$@"
