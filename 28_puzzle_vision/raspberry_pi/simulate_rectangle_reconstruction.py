#!/usr/bin/env python3
"""Generate legal rectangle fragments and reconstruct them from scrambled poses.

This is a geometry-only simulation.  It deliberately gives the solver only the
observed polygon vertices: the original rectangle size, cut locations, piece
poses, and piece order are not used during reconstruction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from arbitrary_puzzle_solver import (
    canonical_polygon,
    edge_length,
    polygon_area,
    polygon_centroid,
    reconstruct_rectangle,
    rotate_points,
)


COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
MIN_EDGE_MM = 20.0


@dataclass(frozen=True)
class GeneratedPuzzle:
    width_mm: float
    height_mm: float
    target_pieces: list[np.ndarray]


def _random_cut(rng: random.Random, total: float, margin: float = 25.0) -> float:
    """Choose a cut that keeps both resulting boundary segments >= margin."""
    low = int(math.ceil(margin))
    high = int(math.floor(total - margin))
    if low > high:
        return total * 0.5
    return float(rng.randint(low, high))


def generate_legal_puzzle(
    piece_count: int,
    seed: int = 2026,
    four_piece_template: str = "slanted",
) -> GeneratedPuzzle:
    """Generate one of four guaranteed-legal rectangular subdivisions.

    The templates are intentionally simple and physically manufacturable:
    one rectangle, two diagonal triangles, a strip plus two triangles, or a
    four-rectangle grid.  Every piece touches the target outline and has 3--4
    edges, so it lies inside the official maximum of five.
    """
    if not 1 <= piece_count <= 4:
        raise ValueError("piece_count must be between 1 and 4")
    rng = random.Random(seed)
    width = float(rng.randint(90, 120))
    height = float(rng.randint(60 if piece_count == 4 else 50, 90))

    if piece_count == 1:
        pieces = [
            np.asarray([[0, 0], [width, 0], [width, height], [0, height]], np.float64)
        ]
    elif piece_count == 2:
        pieces = [
            np.asarray([[0, 0], [width, 0], [0, height]], np.float64),
            np.asarray([[width, 0], [width, height], [0, height]], np.float64),
        ]
    elif piece_count == 3:
        cut_x = _random_cut(rng, width)
        pieces = [
            np.asarray([[0, 0], [cut_x, 0], [cut_x, height], [0, height]], np.float64),
            np.asarray([[cut_x, 0], [width, 0], [cut_x, height]], np.float64),
            np.asarray([[width, 0], [width, height], [cut_x, height]], np.float64),
        ]
    elif four_piece_template == "grid":
        cut_x = _random_cut(rng, width)
        cut_y = _random_cut(rng, height)
        pieces = [
            np.asarray([[0, 0], [cut_x, 0], [cut_x, cut_y], [0, cut_y]], np.float64),
            np.asarray(
                [[cut_x, 0], [width, 0], [width, cut_y], [cut_x, cut_y]],
                np.float64,
            ),
            np.asarray(
                [[0, cut_y], [cut_x, cut_y], [cut_x, height], [0, height]],
                np.float64,
            ),
            np.asarray(
                [
                    [cut_x, cut_y],
                    [width, cut_y],
                    [width, height],
                    [cut_x, height],
                ],
                np.float64,
            ),
        ]
    elif four_piece_template == "slanted":
        # Same topology as the user's reference: a large right triangle and
        # three polygons fanning into its diagonal.  The reference drawing has
        # a 10 mm left boundary segment, which is illegal for this problem, so
        # the three left boundary segments here share the available slack on
        # top of a guaranteed 20 mm minimum.
        pieces = []
        for _attempt in range(1000):
            slack_y = height - 60.0
            slack_breaks = sorted((rng.random(), rng.random()))
            first_left = 20.0 + slack_breaks[0] * slack_y
            second_left = 20.0 + (
                slack_breaks[1] - slack_breaks[0]
            ) * slack_y
            third_left = height - first_left - second_left
            point_b = np.asarray([0.0, first_left], np.float64)
            point_c = np.asarray(
                [0.0, first_left + second_left], np.float64
            )
            point_a = np.asarray(
                [rng.uniform(0.24, 0.34) * width, 0.0], np.float64
            )
            bottom_right = np.asarray([width, height], np.float64)
            diagonal = bottom_right - point_a
            first_fraction = rng.uniform(0.25, 0.30)
            second_fraction = rng.uniform(0.60, 0.69)
            junction_p = point_a + first_fraction * diagonal
            junction_q = point_a + second_fraction * diagonal

            internal_lengths = [
                float(np.linalg.norm(junction_p - point_a)),
                float(np.linalg.norm(junction_p - point_b)),
                float(np.linalg.norm(junction_q - junction_p)),
                float(np.linalg.norm(junction_q - point_c)),
                float(np.linalg.norm(bottom_right - junction_q)),
            ]
            unrelated_differences = [
                abs(first - second)
                for index, first in enumerate(internal_lengths)
                for second in internal_lengths[index + 1 :]
            ]
            if min(internal_lengths) < MIN_EDGE_MM - 1.0e-6:
                continue
            if min(unrelated_differences) <= 3.5:
                continue
            pieces = [
                np.asarray(
                    [[0.0, 0.0], point_a, junction_p, point_b],
                    np.float64,
                ),
                np.asarray(
                    [point_b, junction_p, junction_q, point_c],
                    np.float64,
                ),
                np.asarray(
                    [point_c, junction_q, bottom_right, [0.0, height]],
                    np.float64,
                ),
                np.asarray(
                    # Keep the two collinear T-junctions as contour vertices.
                    # This makes the large piece a legal five-edge polygon and
                    # exposes all three one-to-one seam segments to the matcher.
                    [
                        point_a,
                        [width, 0.0],
                        bottom_right,
                        junction_q,
                        junction_p,
                    ],
                    np.float64,
                ),
            ]
            break
        if not pieces:
            raise ValueError("could not generate distinguishable slanted seams")
    else:
        raise ValueError("four_piece_template must be 'slanted' or 'grid'")
    puzzle = GeneratedPuzzle(width, height, pieces)
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_target_boundary_edge(
    polygon: np.ndarray, width: float, height: float, tolerance: float = 1.0e-6
) -> bool:
    for index in range(len(polygon)):
        start = polygon[index]
        end = polygon[(index + 1) % len(polygon)]
        if (
            abs(float(start[0])) <= tolerance
            and abs(float(end[0])) <= tolerance
            or abs(float(start[0] - width)) <= tolerance
            and abs(float(end[0] - width)) <= tolerance
            or abs(float(start[1])) <= tolerance
            and abs(float(end[1])) <= tolerance
            or abs(float(start[1] - height)) <= tolerance
            and abs(float(end[1] - height)) <= tolerance
        ):
            return True
    return False


def validate_generated_puzzle(puzzle: GeneratedPuzzle) -> None:
    """Raise if a generated puzzle violates any stated field constraint."""
    long_side = max(puzzle.width_mm, puzzle.height_mm)
    short_side = min(puzzle.width_mm, puzzle.height_mm)
    if not (90.0 <= long_side <= 120.0 and 50.0 <= short_side <= 90.0):
        raise ValueError("target rectangle dimensions are outside the official range")
    if not 1 <= len(puzzle.target_pieces) <= 4:
        raise ValueError("piece count is outside 1..4")
    for index, polygon in enumerate(puzzle.target_pieces):
        if not 3 <= len(polygon) <= 5:
            raise ValueError("piece {} has an illegal edge count".format(index + 1))
        minimum_edge = min(edge_length(polygon, edge) for edge in range(len(polygon)))
        if minimum_edge < MIN_EDGE_MM - 1.0e-6:
            raise ValueError("piece {} has an edge shorter than 20 mm".format(index + 1))
        if not _has_target_boundary_edge(
            polygon, puzzle.width_mm, puzzle.height_mm
        ):
            raise ValueError("piece {} does not touch the target outline".format(index + 1))
    total_area = sum(polygon_area(polygon) for polygon in puzzle.target_pieces)
    if abs(total_area - puzzle.width_mm * puzzle.height_mm) > 1.0e-5:
        raise ValueError("pieces do not exactly cover the target rectangle")


def scramble_pieces(
    puzzle: GeneratedPuzzle,
    seed: int = 2026,
    vertex_noise_mm: float = 0.0,
) -> list[dict[str, Any]]:
    """Apply unknown rigid poses and optional camera-like vertex noise."""
    rng = random.Random(seed + 7919)
    pieces: list[dict[str, Any]] = []
    columns = 2 if len(puzzle.target_pieces) > 1 else 1
    for index, target in enumerate(puzzle.target_pieces):
        center = polygon_centroid(target)
        angle_deg = rng.uniform(-170.0, 170.0)
        scattered_center = np.asarray(
            [90.0 + 155.0 * (index % columns), 90.0 + 125.0 * (index // columns)],
            np.float64,
        )
        observed = rotate_points(target - center, math.radians(angle_deg)) + scattered_center
        if vertex_noise_mm > 0.0:
            noise = np.asarray(
                [
                    [
                        rng.gauss(0.0, vertex_noise_mm),
                        rng.gauss(0.0, vertex_noise_mm),
                    ]
                    for _ in range(len(observed))
                ],
                np.float64,
            )
            observed = observed + noise
        pick = polygon_centroid(observed)
        pieces.append(
            {
                "id": "P{}".format(index + 1),
                "vertices_mm": np.round(observed, 6).tolist(),
                "center_mm": np.round(pick, 6).tolist(),
                "pick_point_mm": np.round(pick, 6).tolist(),
                "pick_method": "AREA_CENTROID",
                "simulation_pose_deg": round(angle_deg, 6),
            }
        )
    rng.shuffle(pieces)
    return pieces


def _normalize_polygons(polygons: Iterable[np.ndarray]) -> list[np.ndarray]:
    values = [np.asarray(polygon, np.float64) for polygon in polygons]
    minimum = np.min(np.concatenate(values, axis=0), axis=0)
    return [polygon - minimum for polygon in values]


def evaluate_reconstruction(
    puzzle: GeneratedPuzzle, reconstruction: dict[str, Any]
) -> dict[str, Any]:
    """Compare the recovered labelled layout with truth under four rotations."""
    if not reconstruction.get("ready", False):
        return {"success": False, "error": reconstruction.get("error")}
    solved_by_id = {
        move["piece_id"]: np.asarray(move["target_vertices_local_mm"], np.float64)
        for move in reconstruction["moves"]
    }
    truth_by_id = {
        "P{}".format(index + 1): polygon
        for index, polygon in enumerate(puzzle.target_pieces)
    }
    best_rms = float("inf")
    best_max = float("inf")
    best_quarter_turn = 0
    for quarter_turn in range(4):
        angle = quarter_turn * math.pi * 0.5
        rotated_truth = {
            identifier: rotate_points(polygon, angle)
            for identifier, polygon in truth_by_id.items()
        }
        truth_normalized_list = _normalize_polygons(rotated_truth.values())
        truth_normalized = {
            identifier: polygon
            for identifier, polygon in zip(rotated_truth, truth_normalized_list)
        }
        solved_values = _normalize_polygons(solved_by_id.values())
        solved_normalized = {
            identifier: polygon
            for identifier, polygon in zip(solved_by_id, solved_values)
        }
        distances: list[float] = []
        for identifier in sorted(truth_normalized):
            truth_polygon = canonical_polygon(truth_normalized[identifier])
            solved_polygon = canonical_polygon(solved_normalized[identifier])
            if len(truth_polygon) != len(solved_polygon):
                distances.append(float("inf"))
                continue
            # Canonical starts can change slightly under noise.  Compare every
            # cyclic shift while preserving orientation (mirroring is illegal).
            candidates = []
            for shift in range(len(solved_polygon)):
                shifted = np.roll(solved_polygon, shift, axis=0)
                candidates.append(np.linalg.norm(truth_polygon - shifted, axis=1))
            piece_distances = min(
                candidates, key=lambda values: float(np.mean(values * values))
            )
            distances.extend(float(value) for value in piece_distances)
        rms = math.sqrt(sum(value * value for value in distances) / len(distances))
        maximum = max(distances)
        if rms < best_rms:
            best_rms = rms
            best_max = maximum
            best_quarter_turn = quarter_turn
    solved_width, solved_height = reconstruction["layout_size_mm"]
    dimension_error = min(
        max(
            abs(float(solved_width) - puzzle.width_mm),
            abs(float(solved_height) - puzzle.height_mm),
        ),
        max(
            abs(float(solved_width) - puzzle.height_mm),
            abs(float(solved_height) - puzzle.width_mm),
        ),
    )
    geometric_success = bool(dimension_error <= 2.0)
    return {
        "success": geometric_success,
        "geometric_success": geometric_success,
        "best_truth_rotation_deg": best_quarter_turn * 90,
        "labelled_vertex_rms_error_mm": round(best_rms, 4),
        "labelled_vertex_max_error_mm": round(best_max, 4),
        "dimension_max_error_mm": round(float(dimension_error), 4),
        "note": (
            "A nonzero labelled error can be an equally valid row/column "
            "permutation; geometry success does not require recovering an "
            "unobservable original label order."
        ),
    }


def _panel_transform(
    polygons: list[np.ndarray], x: float, y: float, width: float, height: float
) -> tuple[float, np.ndarray]:
    points = np.concatenate(polygons, axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    extent = np.maximum(maximum - minimum, 1.0)
    scale = min((width - 32.0) / extent[0], (height - 54.0) / extent[1])
    rendered_size = extent * scale
    offset = np.asarray(
        [
            x + (width - rendered_size[0]) * 0.5 - minimum[0] * scale,
            y + 34.0 + (height - 48.0 - rendered_size[1]) * 0.5 - minimum[1] * scale,
        ],
        np.float64,
    )
    return float(scale), offset


def _svg_panel(
    title: str,
    polygons_by_id: list[tuple[str, np.ndarray]],
    x: float,
    y: float,
    width: float,
    height: float,
    highlighted_edges: set[tuple[str, int]] | None = None,
) -> list[str]:
    polygons = [polygon for _identifier, polygon in polygons_by_id]
    scale, offset = _panel_transform(polygons, x, y, width, height)
    lines = [
        '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" rx="10" fill="#ffffff" stroke="#c7ced8"/>'.format(
            x, y, width, height
        ),
        '<text x="{:.1f}" y="{:.1f}" font-size="16" font-weight="600" fill="#1f2937">{}</text>'.format(
            x + 16.0, y + 24.0, html.escape(title)
        ),
    ]
    for identifier, polygon in polygons_by_id:
        numeric = int(identifier[1:]) - 1 if identifier[1:].isdigit() else 0
        rendered = polygon * scale + offset
        points = " ".join("{:.2f},{:.2f}".format(*point) for point in rendered)
        center = polygon_centroid(rendered)
        lines.append(
            '<polygon points="{}" fill="{}" fill-opacity="0.75" stroke="#172033" stroke-width="1.5"/>'.format(
                points, COLORS[numeric % len(COLORS)]
            )
        )
        lines.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="middle" dominant-baseline="middle" font-size="14" font-weight="700" fill="#111827">{}</text>'.format(
                center[0], center[1], html.escape(identifier)
            )
        )
        for edge_index in range(len(polygon)):
            if not highlighted_edges or (
                identifier, edge_index
            ) not in highlighted_edges:
                continue
            start = rendered[edge_index]
            end = rendered[(edge_index + 1) % len(rendered)]
            lines.append(
                '<line x1="{:.2f}" y1="{:.2f}" x2="{:.2f}" y2="{:.2f}" stroke="#d81b60" stroke-width="4" stroke-linecap="round" stroke-dasharray="7 4"/>'.format(
                    start[0], start[1], end[0], end[1]
                )
            )
    return lines


def write_svg(
    output_path: Path,
    puzzle: GeneratedPuzzle,
    observed: list[dict[str, Any]],
    reconstruction: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    target = [
        ("P{}".format(index + 1), polygon)
        for index, polygon in enumerate(puzzle.target_pieces)
    ]
    scattered = [
        (piece["id"], np.asarray(piece["vertices_mm"], np.float64))
        for piece in observed
    ]
    solved = [
        (move["piece_id"], np.asarray(move["target_vertices_local_mm"], np.float64))
        for move in reconstruction.get("moves", [])
    ]
    before_optimization = [
        (
            move["piece_id"],
            np.asarray(
                move.get(
                    "pre_optimization_vertices_local_mm",
                    move["target_vertices_local_mm"],
                ),
                np.float64,
            ),
        )
        for move in reconstruction.get("moves", [])
    ]
    highlighted_edges: set[tuple[str, int]] = set()
    for seam in reconstruction.get("matched_seams", []):
        highlighted_edges.add(
            (str(seam["first_piece_id"]), int(seam["first_edge"]))
        )
        highlighted_edges.add(
            (str(seam["second_piece_id"]), int(seam["second_edge"]))
        )
    pose_graph = reconstruction.get("pose_graph_optimization") or {}
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1580" height="470" viewBox="0 0 1580 470">',
        '<rect width="1580" height="470" fill="#eef2f7"/>',
        '<text x="24" y="30" font-size="21" font-weight="700" fill="#111827">矩形碎片自动重建仿真</text>',
        '<text x="1556" y="29" text-anchor="end" font-size="13" fill="#4b5563">真值 {:.0f}×{:.0f} mm · {} 片 · 尺寸误差 {:.3f} mm · 闭环残差 {:.3f}→{:.3f} mm</text>'.format(
            puzzle.width_mm,
            puzzle.height_mm,
            len(puzzle.target_pieces),
            float(evaluation.get("dimension_max_error_mm", float("nan"))),
            float(pose_graph.get("residual_before_mm", 0.0)),
            float(pose_graph.get("residual_after_mm", 0.0)),
        ),
    ]
    panel_width = 365.0
    panel_height = 365.0
    lines.extend(_svg_panel("① 合法切分真值（求解器不可见）", target, 20, 48, panel_width, panel_height))
    lines.extend(_svg_panel("② 输入：乱序、旋转、平移", scattered, 405, 48, panel_width, panel_height))
    if solved:
        lines.extend(
            _svg_panel(
                "③ 候选拼接（粉色为匹配缝）",
                before_optimization,
                790,
                48,
                panel_width,
                panel_height,
                highlighted_edges=highlighted_edges,
            )
        )
        lines.extend(
            _svg_panel(
                "④ 全局评分 + 闭环优化结果",
                solved,
                1175,
                48,
                panel_width,
                panel_height,
            )
        )
    else:
        lines.extend(
            [
                '<rect x="790" y="48" width="750" height="365" rx="10" fill="#ffffff" stroke="#c7ced8"/>',
                '<text x="806" y="72" font-size="16" font-weight="600" fill="#1f2937">③–④ 重建被安全拒绝</text>',
                '<text x="1165" y="230" text-anchor="middle" font-size="14" fill="#b91c1c">{}</text>'.format(
                    html.escape(str(reconstruction.get("error", "UNKNOWN")))
                ),
            ]
        )
    lines.append(
        '<text x="790" y="450" text-anchor="middle" font-size="12" fill="#6b7280">求解只使用毫米轮廓；不读取颜色、真值尺寸、切割类别或原始邻接关系</text>'
    )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_simulation(
    piece_count: int,
    seed: int,
    noise_mm: float,
    seam_gap_mm: float,
    four_piece_template: str = "slanted",
) -> dict[str, Any]:
    puzzle = generate_legal_puzzle(piece_count, seed, four_piece_template)
    observed = scramble_pieces(puzzle, seed, noise_mm)
    # The simulator uses the edge-pair search so the demonstration remains
    # fast and shows the general algorithm directly.  The live pipeline keeps
    # the additional outer-boundary packing fallback enabled by default.
    reconstruction = reconstruct_rectangle(
        observed,
        requested_gap_mm=seam_gap_mm,
        use_boundary_pack=False,
    )
    evaluation = evaluate_reconstruction(puzzle, reconstruction)
    return {
        "simulation": {
            "seed": seed,
            "piece_count": piece_count,
            "target_size_mm": [puzzle.width_mm, puzzle.height_mm],
            "vertex_noise_sigma_mm": noise_mm,
            "requested_seam_gap_mm": seam_gap_mm,
            "four_piece_template": four_piece_template,
        },
        "constraint_check": {
            "valid": True,
            "piece_edge_counts": [len(piece) for piece in puzzle.target_pieces],
            "piece_minimum_edges_mm": [
                round(
                    min(edge_length(piece, edge) for edge in range(len(piece))),
                    3,
                )
                for piece in puzzle.target_pieces
            ],
            "every_piece_has_target_boundary_edge": True,
        },
        "target_pieces": [
            {"id": "P{}".format(index + 1), "vertices_mm": piece.tolist()}
            for index, piece in enumerate(puzzle.target_pieces)
        ],
        "observed_pieces": observed,
        "reconstruction": reconstruction,
        "evaluation": evaluation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--template",
        choices=("slanted", "grid"),
        default="slanted",
        help="four-piece cut topology; ignored for one to three pieces",
    )
    parser.add_argument(
        "--noise-mm",
        type=float,
        default=0.0,
        help="independent Gaussian noise applied to each observed vertex",
    )
    parser.add_argument(
        "--seam-gap-mm",
        type=float,
        default=0.0,
        help="physical placement gap; use 0 for exact geometric reconstruction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_output",
    )
    args = parser.parse_args()
    if args.noise_mm < 0.0 or args.seam_gap_mm < 0.0:
        parser.error("noise and seam gap must be non-negative")

    report = run_simulation(
        args.pieces,
        args.seed,
        args.noise_mm,
        args.seam_gap_mm,
        four_piece_template=args.template,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rectangle_reconstruction.json"
    svg_path = output_dir / "rectangle_reconstruction.svg"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    puzzle = GeneratedPuzzle(
        report["simulation"]["target_size_mm"][0],
        report["simulation"]["target_size_mm"][1],
        [
            np.asarray(piece["vertices_mm"], np.float64)
            for piece in report["target_pieces"]
        ],
    )
    write_svg(
        svg_path,
        puzzle,
        report["observed_pieces"],
        report["reconstruction"],
        report["evaluation"],
    )
    summary = {
        "ready": report["reconstruction"].get("ready", False),
        "target_size_mm": report["simulation"]["target_size_mm"],
        "solved_size_mm": report["reconstruction"].get("layout_size_mm"),
        "evaluation": report["evaluation"],
        "json": str(json_path),
        "svg": str(svg_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["reconstruction"].get("ready", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
