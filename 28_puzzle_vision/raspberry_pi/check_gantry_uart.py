#!/usr/bin/env python3
"""Safe, no-motion UART diagnostic for the Raspberry Pi/gantry MCU link.

The script sends only ``PING`` lines.  It never sends a task, plan, motion,
magnet, or stop command, so it can be used while the motor power is disabled.
Run it only after the long-running vision service has released the serial port.
"""

from __future__ import annotations

import argparse
import json
import time

import serial


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyAMA0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--reply-window", type=float, default=0.45)
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="accept an echoed PING line when Pi pins 8 and 10 are shorted",
    )
    args = parser.parse_args()

    result = {
        "device": args.device,
        "baudrate": args.baudrate,
        "attempts": args.attempts,
        "bytes_received": 0,
        "lines": [],
    }

    with serial.Serial(
        args.device,
        args.baudrate,
        timeout=0.05,
        write_timeout=0.5,
        exclusive=True,
    ) as port:
        port.reset_input_buffer()
        for _ in range(args.attempts):
            port.write(b"PING\n")
            port.flush()
            deadline = time.monotonic() + args.reply_window
            while time.monotonic() < deadline:
                raw = port.readline()
                if not raw:
                    continue
                result["bytes_received"] += len(raw)
                result["lines"].append(raw.decode("ascii", errors="replace").strip())
            if result["lines"]:
                break

    protocol_reply = any(
        line.startswith(("READY ", "PONG READY", "PONG BUSY"))
        for line in result["lines"]
    )
    loopback_echo = args.loopback and any(line == "PING" for line in result["lines"])
    result["protocol_reply"] = protocol_reply
    result["loopback_echo"] = loopback_echo
    result["link_ok"] = protocol_reply or loopback_echo
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["link_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
