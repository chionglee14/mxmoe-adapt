"""Aggregate repeated independent config-search runs into verification evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for summary_path in args.input:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        matches = [
            record
            for record in summary["all_records"]
            if record["candidate_index"] == args.candidate_index
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one candidate in {summary_path}, got {len(matches)}")
        record = matches[0]
        if not record.get("verified"):
            raise RuntimeError(f"candidate is not verified in {summary_path}")
        runs.append(
            {
                "summary_path": str(summary_path),
                "seed": summary["shape"]["seed"],
                "speedup": record["speedup"],
                "baseline_latency_us": record["baseline_latency_us"],
                "candidate_latency_us": record["candidate_latency_us"],
                "max_abs_error": record["max_abs_error"],
            }
        )

    if len(runs) < 3:
        raise RuntimeError("at least three independent runs are required for verification")
    speedups = [run["speedup"] for run in runs]
    output = {
        "schema_version": 1,
        "status": "verified" if min(speedups) > 1.0 else "measured_not_consistently_faster",
        "candidate_index": args.candidate_index,
        "independent_run_count": len(runs),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "all_runs_faster": min(speedups) > 1.0,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
