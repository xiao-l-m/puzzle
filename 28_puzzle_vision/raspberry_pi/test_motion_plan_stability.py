#!/usr/bin/env python3

from copy import deepcopy
import unittest

from puzzle_motion import MotionPlanStabilityGate


def plan(angle: float = 10.0) -> dict:
    return {
        "mode": 1,
        "strategy": "POLYGON_GRID_PACKING",
        "ready": True,
        "moves": [
            {
                "order": index,
                "piece_id": f"P{index}",
                "pick_a4_mm": [20.0 * index, 170.0 + index],
                "place_a4_mm": [20.0 * index, 80.0 + index],
                "motor5_rotate_deg": angle if index == 1 else 0.0,
            }
            for index in range(1, 5)
        ],
    }


class MotionPlanStabilityTests(unittest.TestCase):
    def test_requires_three_independent_consistent_samples(self) -> None:
        gate = MotionPlanStabilityGate(required_samples=3)
        first = gate.update(1, plan())
        second_plan = plan()
        second_plan["moves"][0]["pick_a4_mm"][0] += 0.2
        second = gate.update(1, second_plan)
        third = gate.update(1, plan(angle=10.4))
        self.assertFalse(first["stable"])
        self.assertFalse(second["stable"])
        self.assertTrue(third["stable"], third)
        self.assertEqual(third["samples"], 3)
        self.assertLessEqual(third["position_spread_mm"], 0.8)
        self.assertLessEqual(third["angle_spread_deg"], 1.5)

    def test_large_target_jump_must_age_out_before_ready_again(self) -> None:
        gate = MotionPlanStabilityGate(required_samples=3)
        for _ in range(3):
            self.assertEqual(gate.update(1, plan())["samples"], _ + 1)
        jumped = plan()
        jumped["moves"][2]["place_a4_mm"][0] += 4.0
        self.assertFalse(gate.update(1, jumped)["stable"])
        self.assertFalse(gate.update(1, deepcopy(jumped))["stable"])
        self.assertTrue(gate.update(1, deepcopy(jumped))["stable"])

    def test_assignment_change_resets_history(self) -> None:
        gate = MotionPlanStabilityGate(required_samples=3)
        gate.update(1, plan())
        gate.update(1, plan())
        changed = plan()
        changed["moves"][0]["piece_id"] = "P4"
        status = gate.update(1, changed)
        self.assertFalse(status["stable"])
        self.assertEqual(status["samples"], 1)

    def test_rotation_wraparound_is_treated_as_two_degrees(self) -> None:
        gate = MotionPlanStabilityGate(
            required_samples=3, angle_tolerance_deg=1.5
        )
        gate.update(1, plan(angle=179.0))
        gate.update(1, plan(angle=-179.0))
        result = gate.update(1, plan(angle=180.0))
        self.assertTrue(result["stable"], result)
        self.assertEqual(result["angle_spread_deg"], 1.0)

    def test_invalid_plan_clears_existing_history(self) -> None:
        gate = MotionPlanStabilityGate(required_samples=3)
        gate.update(1, plan())
        gate.update(1, plan())
        result = gate.update(1, {"ready": False, "moves": []})
        self.assertFalse(result["stable"])
        self.assertEqual(result["samples"], 0)
        self.assertEqual(gate.update(1, plan())["samples"], 1)


if __name__ == "__main__":
    unittest.main()
