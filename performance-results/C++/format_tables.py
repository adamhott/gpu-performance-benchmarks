#!/usr/bin/env python3
"""Read results.csv (same schema as cuda-bench) and write consolidated.md for C++ host runs."""

from __future__ import annotations

import argparse
import csv
import datetime
from collections import defaultdict
from pathlib import Path

# Folder name (under performance-results/C++/) -> human title for markdown
ARCHIVE_TITLES: dict[str, str] = {
    "RTX-A4000": "NVIDIA RTX A4000 — C++ host ports",
    "RTX-4090": "NVIDIA GeForce RTX 4090 — C++ host ports",
    "H100-80GB-HBM3": "NVIDIA H100 80GB HBM3 — C++ host ports",
}


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
        default=None,
        help="Path to results.csv (default: ./results.csv under current working directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: ./consolidated.md next to input)",
    )
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    if args.input is None:
        args.input = cwd / "results.csv"
    if args.output is None:
        args.output = args.input.parent / "consolidated.md"

    if not args.input.is_file():
        raise SystemExit(
            f"Missing {args.input}. Copy `cuda-bench/results/results_cpp.csv` to "
            f"`results.csv` in this directory (see README), or pass --input explicitly."
        )

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))

    if not rows:
        raise SystemExit(f"No data rows in {args.input}")

    gpu = rows[0].get("gpu_name", "")
    cc = rows[0].get("compute_capability", "")
    archive_key = args.input.parent.name
    title = ARCHIVE_TITLES.get(archive_key, f"{archive_key} — C++ host ports")

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
    lines.append(f"# {title}\n")
    lines.append(f"_Generated {today} from `{args.input.name}`._\n")
    lines.append(
        "> **Provenance:** Figures come from `results.csv` in this directory — typically "
        "`cuda-bench/results/results_cpp.csv` renamed after a **`cuda-bench/C++`** run on "
        f"a host in the **{archive_key}** class (see paired snapshot under "
        f"`../CUDA/{archive_key}/`). Regenerate with `python3 ../format_tables.py` from this folder.\n"
    )
    lines.append("## Run context\n")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Archive folder | `{archive_key}` |")
    lines.append(f"| CSV `gpu_name` | {gpu} |")
    lines.append(f"| CSV `compute_capability` | {cc} |")
    lines.append("| Suite | `cuda-bench/C++` (CPU host timing) |")
    lines.append(f"| Paired CUDA snapshot | `../CUDA/{archive_key}/` |")
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
