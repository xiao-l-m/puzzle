#!/usr/bin/env python3

from copy import deepcopy
import time
import tempfile
import unittest
from unittest.mock import patch

import gantry_serial
from gantry_serial import GantryTaskController


READY_STATUS = {
    "config": {"a4_locked": True},
    "a4_found": True,
    "divider_found": True,
    "divider_y_mm": 150.0,
    "stable_four_pieces": True,
    "pieces": [
        {
            "id": "P1",
            "center_mm": [30.0, 80.0],
            "vertices_mm": [[25.0, 75.0], [35.0, 75.0], [30.0, 85.0]],
        },
        {
            "id": "P2",
            "center_mm": [70.0, 90.0],
            "vertices_mm": [[65.0, 85.0], [75.0, 85.0], [70.0, 95.0]],
        },
        {
            "id": "P3",
            "center_mm": [110.0, 100.0],
            "vertices_mm": [[105.0, 95.0], [115.0, 95.0], [110.0, 105.0]],
        },
        {
            "id": "P4",
            "center_mm": [150.0, 110.0],
            "vertices_mm": [[145.0, 105.0], [155.0, 105.0], [150.0, 115.0]],
        },
    ],
}


class FakeSerialLink:
    """Deterministic, connected link with no background serial thread."""

    def __init__(self, device, baudrate, on_line):
        self.device = device
        self.baudrate = baudrate
        self.on_line = on_line
        self.sent = []
        self.closed = False

    def wait_connected(self, timeout):
        return not self.closed

    def send(self, line):
        if self.closed:
            raise RuntimeError("fake link is closed")
        self.sent.append(line)

    def status(self):
        return {
            "device": self.device,
            "baudrate": self.baudrate,
            "port_open": not self.closed,
            "connected": not self.closed,
            "error": None,
            "last_rx": None,
            "last_tx": None if not self.sent else self.sent[-1],
            "last_rx_age_seconds": 0.0,
        }

    def close(self):
        self.closed = True


class CountingSerialModule:
    """Records attempts to open the real UART."""

    def __init__(self):
        self.open_calls = 0

    def Serial(self, *args, **kwargs):
        self.open_calls += 1
        raise AssertionError("serial port must not be opened")


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class UartConsoleConflictTests(unittest.TestCase):
    def test_detects_serial0_alias_of_concrete_tty(self):
        def resolve(path):
            normalized = str(path).replace("\\", "/")
            if normalized in {"/dev/serial0", "/dev/ttyAMA10"}:
                return "/dev/ttyAMA10"
            return normalized

        with patch.object(gantry_serial.sys, "platform", "linux"):
            with patch.object(
                gantry_serial.Path,
                "read_text",
                return_value="console=serial0,115200 root=/dev/mmcblk0p2 rw",
            ):
                with patch.object(
                    gantry_serial.os.path, "realpath", side_effect=resolve
                ):
                    self.assertTrue(
                        gantry_serial.detect_uart_console_conflict(
                            "/dev/ttyAMA10"
                        )
                    )

    def test_windows_does_not_read_proc_or_report_conflict(self):
        with patch.object(gantry_serial.sys, "platform", "win32"):
            with patch.object(
                gantry_serial.Path,
                "read_text",
                side_effect=AssertionError("must not read /proc on Windows"),
            ):
                self.assertFalse(
                    gantry_serial.detect_uart_console_conflict("COM7")
                )

    def test_missing_proc_keeps_existing_opening_behavior(self):
        with patch.object(gantry_serial.sys, "platform", "linux"):
            with patch.object(
                gantry_serial.Path,
                "read_text",
                side_effect=FileNotFoundError,
            ):
                self.assertFalse(
                    gantry_serial.detect_uart_console_conflict(
                        "/dev/serial0"
                    )
                )

    def test_link_reports_conflict_without_opening_uart(self):
        fake_serial = CountingSerialModule()
        with patch.object(gantry_serial, "serial", fake_serial):
            with patch.object(
                gantry_serial,
                "detect_uart_console_conflict",
                return_value=True,
            ):
                link = gantry_serial.GantrySerialLink(
                    "/dev/serial0", 115200, lambda line: None
                )
                link.thread.join(timeout=1.0)
                try:
                    self.assertFalse(link.thread.is_alive())
                    self.assertEqual(fake_serial.open_calls, 0)
                    status = link.status()
                    self.assertFalse(status["port_open"])
                    self.assertFalse(status["connected"])
                    self.assertEqual(
                        status["error"],
                        gantry_serial.UART_CONSOLE_CONFLICT,
                    )
                finally:
                    link.close()


class SerialHeartbeatValidationTests(unittest.TestCase):
    def test_accepts_supported_mcu_responses(self):
        decode = gantry_serial.GantrySerialLink._decode_protocol_line
        for raw in (
            b"READY GANTRY_V4_WEB_PROBE\r\n",
            b"PONG READY\n",
            b"STATE 7 MOVE\n",
            b"ACK 7 COMMIT\n",
            b"DONE 7\n",
            b"ERR 0 BUSY\n",
            b"KEY 1 2\n",
            b"STATUS READY\n",
        ):
            self.assertIsNotNone(decode(raw), raw)

    def test_rejects_noise_and_foreign_lines(self):
        decode = gantry_serial.GantrySerialLink._decode_protocol_line
        for raw in (
            b"\xff\n",
            b"\x00\n",
            b"garbage\n",
            b"\r\n",
        ):
            self.assertIsNone(decode(raw), raw)


class GantryProtocolTests(unittest.TestCase):
    def test_hardware_request_waits_for_frames_after_button_edge(self):
        live_status = deepcopy(READY_STATUS)
        live_status["vision_frame_index"] = 40

        def status_provider():
            return deepcopy(live_status)

        with patch.object(gantry_serial, "GantrySerialLink", FakeSerialLink):
            controller = GantryTaskController(
                status_provider,
                "/dev/fake-gantry",
                115200,
                vision_wait_seconds=1.0,
            )
            try:
                controller.request_task(1, 7010, "TEST_KEY")
                self.assertTrue(
                    wait_until(
                        lambda: controller.status()["state"]
                        == "WAITING_VISION"
                    )
                )
                self.assertEqual(controller.link.sent, [])
                self.assertIn("0/8", controller.status()["message"])

                # Seven new frames are deliberately insufficient.
                live_status["vision_frame_index"] = 47
                time.sleep(0.08)
                self.assertEqual(controller.link.sent, [])

                live_status["vision_frame_index"] = 48
                self.assertTrue(
                    wait_until(
                        lambda: controller.status()["state"]
                        == "WAITING_MCU"
                    ),
                    controller.status(),
                )
                snapshot = controller.status()["frozen_plan"]["vision_snapshot"]
                self.assertEqual(snapshot["request_frame_index"], 40)
                self.assertEqual(snapshot["frozen_frame_index"], 48)
                self.assertEqual(snapshot["fresh_frames_after_request"], 8)
            finally:
                controller.close()

    def test_click_probe_sends_metric_target_then_can_return_home(self):
        with patch.object(gantry_serial, "GantrySerialLink", FakeSerialLink):
            controller = GantryTaskController(
                lambda: {}, "/dev/fake-gantry", 115200
            )
            try:
                request_id = controller.request_probe(100.04, 200.06)
                self.assertEqual(
                    controller.link.sent[-1],
                    f"PROBE {request_id} 1000 2001",
                )
                self.assertEqual(controller.status()["active_mode"], 4)
                self.assertEqual(
                    controller.status()["frozen_plan"]["z_down_mm"], 12.0
                )
                controller._on_serial_line(f"DONE {request_id}")
                home_request = controller.request_probe_home()
                self.assertEqual(
                    controller.link.sent[-1], f"PROBEHOME {home_request}"
                )
            finally:
                controller.close()

    def test_click_probe_rejects_unreachable_point_without_sending(self):
        with patch.object(gantry_serial, "GantrySerialLink", FakeSerialLink):
            controller = GantryTaskController(
                lambda: {}, "/dev/fake-gantry", 115200
            )
            try:
                with self.assertRaisesRegex(ValueError, "outside reachable"):
                    controller.request_probe(3.9, 200.0)
                self.assertEqual(controller.link.sent, [])
            finally:
                controller.close()

    def test_item_line_uses_tenth_mm_and_millidegree_integers(self):
        move = {
            "pick_a4_mm": [12.34, 56.78],
            "place_a4_mm": [90.06, 123.44],
            "motor5_rotate_deg": -17.8916,
        }
        line = GantryTaskController._item_line(42, 3, move)
        self.assertEqual(
            line,
            "ITEM 42 3 123 568 901 1234 -17892",
        )
        fields = line.split()
        self.assertEqual(len(fields), 8)
        self.assertTrue(all(field.lstrip("-").isdigit() for field in fields[1:]))

    def test_trigger_freezes_plan_and_duplicate_request_is_idempotent(self):
        live_status = deepcopy(READY_STATUS)

        def status_provider():
            return deepcopy(live_status)

        with patch.object(gantry_serial, "GantrySerialLink", FakeSerialLink):
            controller = GantryTaskController(
                status_provider,
                "/dev/fake-gantry",
                115200,
                vision_wait_seconds=0.5,
            )
            try:
                with tempfile.TemporaryDirectory() as audit_dir:
                    controller.audit_log_dir = gantry_serial.Path(audit_dir)
                    request_id = controller.request_task(1, 7001, "TEST_KEY")
                    self.assertEqual(request_id, 7001)
                    self.assertTrue(
                        wait_until(
                            lambda: controller.status()["state"] == "WAITING_MCU"
                        ),
                        controller.status(),
                    )

                    first_status = controller.status()
                    frozen = deepcopy(first_status["frozen_plan"])
                    self.assertIsNotNone(frozen)
                    self.assertEqual(frozen["request_id"], 7001)
                    self.assertEqual(len(frozen["moves"]), 4)
                    sent_before_duplicate = list(controller.link.sent)
                    queue_size_before = controller.requests.qsize()

                    # A live camera update after motion starts must not alter the
                    # immutable plan that was already transmitted to the MCU.
                    live_status["pieces"][0]["center_mm"] = [190.0, 140.0]
                    live_status["pieces"][0]["vertices_mm"] = [
                        [185.0, 135.0],
                        [195.0, 135.0],
                        [190.0, 145.0],
                    ]
                    live_status["stable_four_pieces"] = False
                    time.sleep(0.08)
                    self.assertEqual(controller.status()["frozen_plan"], frozen)

                    duplicate = controller.request_task(2, 7001, "REPEATED_KEY")
                    self.assertEqual(duplicate, 7001)
                    time.sleep(0.08)
                    self.assertEqual(controller.requests.qsize(), queue_size_before)
                    self.assertEqual(controller.link.sent, sent_before_duplicate)
                    self.assertEqual(controller.status()["frozen_plan"], frozen)
                    self.assertTrue(
                        any(
                            "duplicate KEY 7001 ignored" in entry
                            for entry in controller.status()["history"]
                        )
                    )
            finally:
                controller.close()

    def test_mode3_sends_one_to_four_items(self):
        with patch.object(gantry_serial, "GantrySerialLink", FakeSerialLink):
            controller = GantryTaskController(
                lambda: {}, "/dev/fake-gantry", 115200
            )
            try:
                moves = [
                    {
                        "pick_a4_mm": [50.0 + index, 90.0],
                        "place_a4_mm": [80.0 + index, 180.0],
                        "motor5_rotate_deg": 10.0 * index,
                    }
                    for index in range(2)
                ]
                controller._send_plan(9001, 3, {"moves": moves})
                self.assertEqual(controller.link.sent[0], "PLAN 9001 3 2")
                self.assertEqual(controller.link.sent[-1], "COMMIT 9001")
                self.assertEqual(
                    len([line for line in controller.link.sent if line.startswith("ITEM ")]),
                    2,
                )
            finally:
                controller.close()


if __name__ == "__main__":
    unittest.main()
