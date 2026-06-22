#!/usr/bin/env python3
"""Variance of per-step normalized decode speed by provider.

This is a small companion to ``plot.py``. It intentionally reuses
``plot.load_observations`` so the eligible rows and normalized-decode-speed
definition stay exactly aligned with the scatter artifact:

    output_tokens / observable_generation_span_seconds

The output includes both the raw per-step ratios and the figure's display-capped
view, where values above ``--max-speed-tokens-per-second`` are clipped to the cap.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import plot as decode_speed_plot


EXP_DIR = Path(__file__).resolve().parent
PROVIDERS = ("claude", "codex")


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def variance_stats(values: list[float]) -> dict[str, float | int | None]:
    """Return count, mean, variance, stddev, and coefficient of variation."""
    vals = finite_values(values)
    n = len(vals)
    if n == 0:
        return {
            "rows": 0,
            "mean": None,
            "population_variance": None,
            "sample_variance": None,
            "population_stddev": None,
            "sample_stddev": None,
        }

    mean = math.fsum(vals) / n
    squared_diffs = [math.pow(value - mean, 2) for value in vals]
    population_variance = math.fsum(squared_diffs) / n
    sample_variance = math.fsum(squared_diffs) / (n - 1) if n > 1 else None
    population_stddev = math.sqrt(population_variance)
    sample_stddev = math.sqrt(sample_variance) if sample_variance is not None else None
    population_cv_percent = population_stddev / mean * 100 if mean else None
    sample_cv_percent = sample_stddev / mean * 100 if mean and sample_stddev is not None else None
    return {
        "rows": n,
        "mean": mean,
        "population_variance": population_variance,
        "sample_variance": sample_variance,
        "population_stddev": population_stddev,
        "sample_stddev": sample_stddev,
        "population_cv_percent": population_cv_percent,
        "sample_cv_percent": sample_cv_percent,
    }


def fmt(value: float | int | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def collect_rows(
    observations: list[decode_speed_plot.Observation], *, max_speed: float
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for provider in PROVIDERS:
        speeds = [
            row.normalized_decode_speed
            for row in observations
            if row.provider == provider and math.isfinite(row.normalized_decode_speed)
        ]
        rows_above_cap = sum(1 for speed in speeds if speed > max_speed)
        cap_share = rows_above_cap / len(speeds) if speeds else None

        for view, values in (
            ("raw", speeds),
            (f"display_capped_{max_speed:g}", [min(speed, max_speed) for speed in speeds]),
        ):
            stats = variance_stats(values)
            rows.append(
                {
                    "provider": provider,
                    "speed_view": view,
                    "rows": fmt(stats["rows"], digits=0),
                    "mean_normalized_decode_tokens_per_second": fmt(stats["mean"]),
                    "population_variance_normalized_decode_tokens_per_second_squared": fmt(
                        stats["population_variance"]
                    ),
                    "sample_variance_normalized_decode_tokens_per_second_squared": fmt(
                        stats["sample_variance"]
                    ),
                    "population_stddev_normalized_decode_tokens_per_second": fmt(
                        stats["population_stddev"]
                    ),
                    "sample_stddev_normalized_decode_tokens_per_second": fmt(
                        stats["sample_stddev"]
                    ),
                    "population_coefficient_of_variation_percent": fmt(
                        stats["population_cv_percent"]
                    ),
                    "sample_coefficient_of_variation_percent": fmt(
                        stats["sample_cv_percent"]
                    ),
                    "display_cap_tokens_per_second": fmt(max_speed),
                    "rows_above_display_cap": fmt(rows_above_cap, digits=0),
                    "share_above_display_cap": fmt(cap_share),
                }
            )
    return rows


def write_csv(rows: list[dict[str, str]], out_dir: Path) -> Path:
    out = out_dir / "context_decode_speed_variance.csv"
    fieldnames = [
        "provider",
        "speed_view",
        "rows",
        "mean_normalized_decode_tokens_per_second",
        "population_variance_normalized_decode_tokens_per_second_squared",
        "sample_variance_normalized_decode_tokens_per_second_squared",
        "population_stddev_normalized_decode_tokens_per_second",
        "sample_stddev_normalized_decode_tokens_per_second",
        "population_coefficient_of_variation_percent",
        "sample_coefficient_of_variation_percent",
        "display_cap_tokens_per_second",
        "rows_above_display_cap",
        "share_above_display_cap",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    decode_speed_plot.trace_db.add_db_args(parser, default_output_dir=EXP_DIR)
    parser.add_argument(
        "--max-speed-tokens-per-second",
        type=float,
        default=decode_speed_plot.DEFAULT_MAX_SPEED_TOKENS_PER_SECOND,
        help="display cap used for the capped variance row",
    )
    args = parser.parse_args()

    con = decode_speed_plot.trace_db.open_from_args(args)
    try:
        observations = decode_speed_plot.load_observations(con)
    finally:
        con.close()

    rows = collect_rows(observations, max_speed=args.max_speed_tokens_per_second)
    out = write_csv(rows, Path(args.output_dir))
    print(f"Saved {out}")
    for row in rows:
        if row["speed_view"] == "raw":
            print(
                f"{row['provider']}: n={row['rows']}, "
                f"population variance={row['population_variance_normalized_decode_tokens_per_second_squared']}, "
                f"sample variance={row['sample_variance_normalized_decode_tokens_per_second_squared']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
