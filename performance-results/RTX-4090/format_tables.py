#!/usr/bin/env python3
"""Read results.csv (cuda-bench CSV schema) and write consolidated.md."""

from __future__ import annotations

import argparse
import csv
import datetime
from collections import defaultdict
from pathlib import Path


def fmt_int(n: str) -> str:
    try:
        v = int(n)
        return f"{v:,}"
    except ValueError:
        return n


def fmt_float(n: str, nd: int = 4) -> str:
    try:
        return f"{float(n):.{nd}f}"
    except ValueError:
        return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "results.csv",
        help="Path to results.csv (default: alongside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "consolidated.md",
        help="Output markdown path",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    if not rows:
        raise SystemExit(f"No data rows in {args.input}")

    gpu = rows[0].get("gpu_name", "")
    cc = rows[0].get("compute_capability", "")

    order = [
        "saxpy",
        "sgemm",
        "stencil",
        "reduction",
        "montecarlo",
        "fft_1d_c2c",
        "nbody_barnes_hut",
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["benchmark"]].append(r)

    lines: list[str] = []
    today = datetime.date.today().isoformat()
    lines.append(f"# {gpu} — consolidated `cuda-bench` results\n")
    lines.append(f"_Generated {today} from `{args.input.name}`._\n")
    lines.append(
        "> **Provenance:** All figures are derived from `results.csv` in this directory "
        "(typically copied from `cuda-bench/results/results.csv` after a run). Regenerate "
        "this file with `python3 format_tables.py` whenever `results.csv` changes.\n"
    )
    lines.append("## Run context\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| GPU | {gpu} |")
    lines.append(f"| Compute capability | {cc} |")
    lines.append("| Suite | `cuda-bench` (`scripts/run_all.sh`) |")
    lines.append("| Notes | RunPod-style container (GPU clock lock often unavailable). |")
    lines.append("\n## Summary tables\n")

    for bench in order:
        lines.append(f"### {bench}\n")
        sub = grouped.get(bench, [])
        if not sub:
            lines.append("_No rows for this benchmark in the CSV._\n")
            continue
        lines.append(
            "| problem_size | variant | mean_ms | stddev_ms | gflops | gb_s | "
            "% peak F | % peak BW |"
        )
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in sub:
            ps = fmt_int(r.get("problem_size", ""))
            lines.append(
                f"| {ps} | {r.get('variant', '')} | {fmt_float(r.get('mean_ms', ''))} | "
                f"{fmt_float(r.get('stddev_ms', ''))} | {fmt_float(r.get('gflops', ''))} | "
                f"{fmt_float(r.get('gb_s', ''))} | {fmt_float(r.get('pct_peak_flops', ''))} | "
                f"{fmt_float(r.get('pct_peak_bw', ''))} |"
            )
        lines.append("")

    lines.append("## Raw CSV\n")
    lines.append("```csv")
    with args.input.open() as f:
        lines.append(f.read().rstrip())
    lines.append("```\n")

    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
