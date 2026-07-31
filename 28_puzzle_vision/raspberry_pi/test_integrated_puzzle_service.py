#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

from integrated_puzzle_vision import (
    INTEGRATED_HTML,
    IntegratedPuzzleVisionApp,
    parse_hardware_key,
)


class IntegratedPuzzleServiceTests(unittest.TestCase):
    def test_three_hardware_keys_map_to_three_modes(self) -> None:
        self.assertEqual(parse_hardware_key("KEY 17 1"), (17, 1))
        self.assertEqual(parse_hardware_key("KEY 18 2"), (18, 2))
        self.assertEqual(parse_hardware_key("KEY 19 3"), (19, 3))
        self.assertIsNone(parse_hardware_key("PONG READY"))

    def test_invalid_hardware_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_hardware_key("KEY 20 4")

    def test_dashboard_exposes_both_mode_previews(self) -> None:
        page = INTEGRATED_HTML.decode("utf-8")
        self.assertIn("综合服务", page)
        self.assertIn("view2Button.disabled=false", page)
        self.assertIn("view3Button.disabled=false", page)
        self.assertIn("mode3.disabled=!ser.connected||robotBusy", page)

    def test_physical_key_switches_vision_before_controller_queue(self) -> None:
        events = []
        app = IntegratedPuzzleVisionApp.__new__(IntegratedPuzzleVisionApp)
        app._arm_hardware_mode = lambda mode: events.append(("vision", mode))
        app.controller = SimpleNamespace(
            _record=lambda message: events.append(("record", message)),
            _on_serial_line=lambda line: events.append(("controller", line)),
        )
        app._on_integrated_serial_line("KEY 31 3")
        self.assertEqual(
            events,
            [("vision", 3), ("controller", "KEY 31 3")],
        )


if __name__ == "__main__":
    unittest.main()
