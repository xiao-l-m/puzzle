#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class FirmwareProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = (ROOT / "30_stepper_5motor_magnet_test" / "gantry_config.h").read_text(
            encoding="utf-8"
        )
        cls.controller = (
            ROOT / "30_stepper_5motor_magnet_test" / "gantry_controller.c"
        ).read_text(encoding="utf-8")

    def test_plan_capacity_is_one_to_four(self) -> None:
        self.assertIn("GANTRY_MIN_PLAN_ITEMS         1U", self.config)
        self.assertIn("GANTRY_MAX_PLAN_ITEMS         4U", self.config)
        self.assertIn("itemCount < GANTRY_MIN_PLAN_ITEMS", self.controller)
        self.assertIn("itemCount > GANTRY_MAX_PLAN_ITEMS", self.controller)

    def test_mode3_and_protocol_rejections_are_present(self) -> None:
        self.assertIn("mode > 3U", self.controller)
        self.assertIn("ITEM_DUPLICATE", self.controller)
        self.assertIn("MISSING_ITEM", self.controller)
        self.assertIn('"ERR %lu RANGE"', self.controller)

    def test_carried_xy_and_rotation_are_concurrent_and_share_deadline(self) -> None:
        self.assertIn("start_place_xy_and_rotation", self.controller)
        self.assertIn('send_state("MOVE_PLACE_ROTATE")', self.controller)
        self.assertIn("rotationDeadline", self.controller)
        self.assertIn("xyDeadline", self.controller)
        self.assertIn("STATE_WAIT_PLACE_XY_ROTATE", self.controller)


if __name__ == "__main__":
    unittest.main()
