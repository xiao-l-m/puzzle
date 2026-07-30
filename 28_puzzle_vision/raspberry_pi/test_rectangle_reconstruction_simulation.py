#!/usr/bin/env python3

from __future__ import annotations

import unittest

from simulate_rectangle_reconstruction import (
    generate_legal_puzzle,
    run_simulation,
    validate_generated_puzzle,
)


class RectangleReconstructionSimulationTests(unittest.TestCase):
    def test_generators_obey_all_field_constraints(self) -> None:
        for piece_count in range(1, 5):
            for seed in range(20):
                puzzle = generate_legal_puzzle(piece_count, seed)
                validate_generated_puzzle(puzzle)

    def test_solver_reconstructs_every_supported_piece_count(self) -> None:
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                report = run_simulation(
                    piece_count=piece_count,
                    seed=2026,
                    noise_mm=0.0,
                    seam_gap_mm=0.0,
                )
                self.assertTrue(report["reconstruction"].get("ready"), report)
                self.assertTrue(report["evaluation"]["geometric_success"])
                self.assertLessEqual(
                    report["evaluation"]["dimension_max_error_mm"], 0.02
                )

    def test_solver_reconstructs_varied_slanted_four_piece_cases(self) -> None:
        for seed in range(2026, 2031):
            with self.subTest(seed=seed):
                report = run_simulation(
                    piece_count=4,
                    seed=seed,
                    noise_mm=0.0,
                    seam_gap_mm=0.0,
                    four_piece_template="slanted",
                )
                self.assertTrue(report["reconstruction"].get("ready"), report)
                self.assertTrue(report["evaluation"]["geometric_success"])
                self.assertLessEqual(
                    report["evaluation"]["dimension_max_error_mm"], 0.02
                )

    def test_noisy_slanted_case_uses_global_pose_refinement(self) -> None:
        report = run_simulation(
            piece_count=4,
            seed=2026,
            noise_mm=0.2,
            seam_gap_mm=0.0,
            four_piece_template="slanted",
        )
        reconstruction = report["reconstruction"]
        self.assertTrue(reconstruction.get("ready"), report)
        quality = reconstruction["global_quality"]
        self.assertEqual(quality["connected_components"], 1)
        self.assertGreater(quality["raster_fill_ratio"], 0.94)
        optimization = reconstruction["pose_graph_optimization"]
        self.assertTrue(optimization["accepted"])
        self.assertLess(
            optimization["residual_after_mm"],
            optimization["residual_before_mm"],
        )
        self.assertLessEqual(
            report["evaluation"]["dimension_max_error_mm"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
