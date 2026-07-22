"""Aggregate measured C500 benchmark JSON files without hiding failed runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    records = []
    invalid = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            invalid.append({"path": str(path), "reason": type(error).__name__})
            continue
        if payload.get("status") != "measured" or "speedup" not in payload:
            invalid.append({"path": str(path), "reason": "not a measured benchmark result"})
            continue
        records.append(
            {
                "path": str(path),
                "shape": payload.get("shape"),
                "environment_id": payload.get("environment", {}).get("environment_id"),
                "speedup": float(payload["speedup"]),
                "correctness_passed": bool(payload.get("correctness", {}).get("passed")),
            }
        )
    passing = [record for record in records if record["correctness_passed"]]
    geomean = None
    if passing and all(record["speedup"] > 0 for record in passing):
        geomean = math.exp(
            sum(math.log(record["speedup"]) for record in passing) / len(passing)
        )
    return {
        "schema_version": 1,
        "total_measured": len(records),
        "correctness_passed": len(passing),
        "geomean_speedup_passing_only": geomean,
        "records": records,
        "invalid_or_nonmeasured": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = summarize(sorted(args.results_dir.glob("*.json")))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
