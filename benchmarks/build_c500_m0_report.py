"""Build the C500 M0 evidence report from measured experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fp16 = load(args.results / "validation-realshape-fp16-aggregate.json")
    bf16 = load(args.results / "validation-realshape-bf16-aggregate.json")
    search = load(
        args.results
        / "search-realshape-t4-e8-h4096-i14336-fp16-safe-baseline"
        / "summary.json"
    )
    evidence = load(args.results / "validation-realshape-fp16-seed17/candidate_04.json")
    failure_log = (
        args.results
        / "search-realshape-t4-e8-h4096-i14336-fp16"
        / "candidate_01.log"
    ).read_text(encoding="utf-8")
    expected_failure = "Required: 73728, Hardware limit: 65536" in failure_log
    if not expected_failure:
        raise RuntimeError("upstream default shared-memory failure evidence is missing")

    report = {
        "schema_version": 1,
        "milestone": "M0 C500 functional recovery and first verified tuning result",
        "status": "verified",
        "hardware": evidence["environment"]["torch_device"],
        "environment_id": evidence["environment"]["environment_id"],
        "packages": evidence["environment"]["packages"],
        "source_revisions": evidence["environment"]["source_revisions"],
        "shape": search["shape"],
        "upstream_default": {
            "status": "compile_failed_out_of_shared_memory",
            "config": search["upstream_default_config"],
            "required_shared_memory_bytes": 73728,
            "hardware_limit_bytes": 65536,
            "evidence": str(
                args.results
                / "search-realshape-t4-e8-h4096-i14336-fp16"
                / "candidate_01.log"
            ),
        },
        "c500_safe_baseline_config": search["baseline_config"],
        "best_verified_config": search["best"]["config"],
        "fp16_validation": fp16,
        "bf16_validation": bf16,
        "limitations": [
            "One synthetic decode shape has been independently verified so far.",
            "Expert count is 8 and routing uses seeded synthetic Top-K assignments.",
            "Route-class coverage, prefill shapes, E=64/128, and model-layer integration remain pending.",
            "M0 uses explicit FlagGems configuration injection; an upstream MetaX config table is pending.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
