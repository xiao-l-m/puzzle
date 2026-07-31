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
        cls.main = (
            ROOT / "30_stepper_5motor_magnet_test" / "empty.c"
        ).read_text(encoding="utf-8")
        cls.syscfg = (
            ROOT / "30_stepper_5motor_magnet_test" / "empty.syscfg"
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

    def test_key2_starts_mode2_and_pb21_remains_emergency_stop(self) -> None:
        self.assertIn("KEYS_KEY2_PIN", self.main)
        self.assertIn("(uint8_t)(index + 1U)", self.main)
        self.assertIn("KEY_PIN_21_PIN", self.main)
        self.assertIn('GantryController_EmergencyStop("PB21")', self.main)
        self.assertIn(
            'KEYS.associatedPins[1].pin.$assign      = "PB18"',
            self.syscfg,
        )

    def test_board_led_distinguishes_active_complete_and_fault(self) -> None:
        self.assertIn("LED1_PIN_22_PIN", self.main)
        self.assertIn("GantryController_IsIdle()", self.main)
        self.assertIn("COMPLETE_LED_HALF_PERIOD_MS 500U", self.main)
        self.assertIn("REJECT_LED_HALF_PERIOD_MS   200U", self.main)
        self.assertIn("GantryController_TakeCompletionEvent()", self.main)


if __name__ == "__main__":
    unittest.main()
