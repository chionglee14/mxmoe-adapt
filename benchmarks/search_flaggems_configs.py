"""Run an isolated, failure-preserving C500 FlagGems MoE config sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from mxmoe_adapt.config_schema import KernelConfig, SearchConstraints


def initial_candidates(intermediate_size: int) -> list[KernelConfig]:
    default_n = 128 if intermediate_size >= 4096 else 64
    wide_n = 256 if intermediate_size >= 4096 else 128
    raw = [
        (16, default_n, 128, 1, 4, 3),  # FlagGems default decode heuristic
        (16, default_n, 64, 1, 4, 2),
        (16, default_n, 32, 1, 4, 2),
        (16, 32, 64, 1, 4, 2),
        (8, 64, 64, 1, 4, 2),
        (32, default_n, 64, 1, 4, 2),
        (16, wide_n, 64, 1, 4, 2),
        (16, default_n, 64, 1, 8, 2),
        (16, default_n, 64, 1, 2, 2),
    ]
    candidates = [
        KernelConfig(
            block_size_m=m,
            block_size_n=n,
            block_size_k=k,
            group_size_m=group,
            num_warps=warps,
            num_stages=stages,
            align_block_size=m,
        )
        for m, n, k, group, warps, stages in raw
    ]
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-index",
        type=int,
        action="append",
        help="run only selected 1-based candidate indices; may be repeated",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    benchmark = root / "benchmarks" / "benchmark_fused_moe.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    all_candidates = initial_candidates(args.intermediate_size)
    baseline_config = all_candidates[1]
    constraints = SearchConstraints()
    selected = set(args.candidate_index or range(1, len(all_candidates) + 1))
    for index, config in enumerate(all_candidates, start=1):
        if index not in selected:
            continue
        config_payload = asdict(config)
        result_path = args.output_dir / f"candidate_{index:02d}.json"
        log_path = args.output_dir / f"candidate_{index:02d}.log"
        record = {
            "candidate_index": index,
            "config": config_payload,
            "baseline_config": asdict(baseline_config),
            "result_path": str(result_path),
            "log_path": str(log_path),
            "verified": False,
        }
        if not constraints.accepts(config):
            record.update(
                {
                    "status": "filtered_by_static_resource_proxy",
                    "returncode": None,
                    "estimated_smem_bytes": config.estimated_smem_bytes(),
                    "shared_memory_budget_bytes": constraints.shared_memory_budget_bytes,
                }
            )
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue
        command = [
            sys.executable,
            str(benchmark),
            "--baseline",
            "mxmoe_adapt.adapters.flaggems_tuned:run_baseline",
            "--candidate",
            "mxmoe_adapt.adapters.flaggems_tuned:run",
            "--device",
            "cuda",
            "--dtype",
            args.dtype,
            "--tokens",
            str(args.tokens),
            "--experts",
            str(args.experts),
            "--hidden-size",
            str(args.hidden_size),
            "--intermediate-size",
            str(args.intermediate_size),
            "--top-k",
            str(args.top_k),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--rounds",
            str(args.rounds),
            "--seed",
            str(args.seed),
            "--output",
            str(result_path),
        ]
        environment = os.environ.copy()
        environment["MXMOE_KERNEL_CONFIG"] = json.dumps(config_payload)
        environment["MXMOE_BASELINE_CONFIG"] = json.dumps(asdict(baseline_config))
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        record.update({"status": "executed", "returncode": completed.returncode})
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "verified": bool(
                        result["baseline_correctness_passed"]
                        and result["correctness"]["passed"]
                    ),
                    "baseline_latency_us": result["baseline_latency_us"],
                    "candidate_latency_us": result["candidate_latency_us"],
                    "speedup": result["speedup"],
                    "candidate_first_call_ms": result["candidate_first_call_ms"],
                    "max_abs_error": result["correctness"]["max_abs"],
                }
            )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    verified = [record for record in records if record["verified"]]
    filtered = [
        record
        for record in records
        if record.get("status") == "filtered_by_static_resource_proxy"
    ]
    verified.sort(key=lambda record: record["speedup"], reverse=True)
    summary = {
        "schema_version": 1,
        "status": "measured",
        "shape": {
            "tokens": args.tokens,
            "experts": args.experts,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "top_k": args.top_k,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "candidate_count": len(records),
        "ranking_metric": "paired baseline-to-candidate speedup within each process",
        "baseline_config": asdict(baseline_config),
        "upstream_default_config": asdict(all_candidates[0]),
        "verified_count": len(verified),
        "filtered_count": len(filtered),
        "failed_count": len(records) - len(verified) - len(filtered),
        "best": verified[0] if verified else None,
        "ranked_verified": verified,
        "all_records": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not verified:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
