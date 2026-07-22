"""C500 Fused MoE correctness and latency harness."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from mxmoe_adapt.adapters.reference import run as reference_moe
from mxmoe_adapt.environment import collect_environment
from mxmoe_adapt.route_features import extract_route_features


def _load_callable(specification: str) -> Callable[..., Any]:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("candidate must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{specification} is not callable")
    return function


def _synchronize(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_batch(torch: Any, function: Callable[[], Any], repeats: int) -> float:
    _synchronize(torch)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    _synchronize(torch)
    return (time.perf_counter() - started) * 1_000_000 / repeats


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "p90_us": _percentile(samples, 0.90),
        "p99_us": _percentile(samples, 0.99),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _time_paired(
    torch: Any,
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    warmup: int,
    repeats: int,
    rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(warmup):
        baseline()
        candidate()
    baseline_samples: list[float] = []
    candidate_samples: list[float] = []
    for round_index in range(rounds):
        ordered = (
            ((baseline, baseline_samples), (candidate, candidate_samples))
            if round_index % 2 == 0
            else ((candidate, candidate_samples), (baseline, baseline_samples))
        )
        for function, samples in ordered:
            samples.append(_time_batch(torch, function, repeats))
    return _summary(baseline_samples), _summary(candidate_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="mxmoe_adapt.adapters.flaggems:run",
        help="baseline callable using module:function syntax",
    )
    parser.add_argument(
        "--candidate",
        default="mxmoe_adapt.adapters.flaggems:run",
        help="callable using module:function syntax",
    )
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--library-log-level",
        choices=("WARNING", "ERROR", "CRITICAL"),
        default="ERROR",
        help="suppress per-dispatch warnings during performance timing",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.rounds <= 0:
        raise ValueError("warmup must be non-negative; repeats and rounds must be positive")

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA-style device API is unavailable; run inside the MXMACA environment")
    if args.top_k > args.experts:
        raise ValueError("top_k cannot exceed experts")
    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    shape = {
        "tokens": args.tokens,
        "experts": args.experts,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "top_k": args.top_k,
        "dtype": args.dtype,
    }

    hidden = torch.randn(args.tokens, args.hidden_size, device=args.device, dtype=dtype) * 0.1
    w1 = torch.randn(
        args.experts,
        2 * args.intermediate_size,
        args.hidden_size,
        device=args.device,
        dtype=dtype,
    ) * 0.05
    w2 = torch.randn(
        args.experts,
        args.hidden_size,
        args.intermediate_size,
        device=args.device,
        dtype=dtype,
    ) * 0.05
    routing_logits = torch.randn(args.tokens, args.experts, device=args.device)
    topk_logits, topk_ids = torch.topk(routing_logits, args.top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).to(dtype)

    baseline = _load_callable(args.baseline)
    candidate = _load_callable(args.candidate)
    logging.getLogger("flag_gems.fused.fused_moe").setLevel(
        getattr(logging, args.library_log_level)
    )
    atol = args.atol if args.atol is not None else (0.05 if args.dtype == "fp16" else 0.1)
    rtol = args.rtol if args.rtol is not None else (0.05 if args.dtype == "fp16" else 0.1)
    with torch.no_grad():
        expected = reference_moe(hidden, w1, w2, topk_weights, topk_ids)
        _synchronize(torch)
        baseline_first_started = time.perf_counter()
        baseline_output = baseline(hidden, w1, w2, topk_weights, topk_ids)
        _synchronize(torch)
        baseline_first_call_ms = (time.perf_counter() - baseline_first_started) * 1_000
        candidate_first_started = time.perf_counter()
        actual = candidate(hidden, w1, w2, topk_weights, topk_ids)
        _synchronize(torch)
        candidate_first_call_ms = (time.perf_counter() - candidate_first_started) * 1_000
        baseline_passed = bool(torch.allclose(baseline_output, expected, atol=atol, rtol=rtol))
        candidate_passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
        difference = (actual.float() - expected.float()).abs()
        denominator = expected.float().abs().clamp_min(1e-6)
        max_abs = float(difference.max().item())
        max_rel = float((difference / denominator).max().item())
        baseline_statistics, candidate_statistics = _time_paired(
            torch,
            lambda: baseline(hidden, w1, w2, topk_weights, topk_ids),
            lambda: candidate(hidden, w1, w2, topk_weights, topk_ids),
            args.warmup,
            args.repeats,
            args.rounds,
        )
        baseline_latency = baseline_statistics["median_us"]
        candidate_latency = candidate_statistics["median_us"]

    route = extract_route_features(topk_ids.cpu().tolist(), args.experts)
    payload = {
        "schema_version": 1,
        "status": "measured",
        "environment": collect_environment(),
        "shape": shape,
        "route_features": route.as_dict(),
        "baseline": args.baseline,
        "baseline_config": json.loads(os.environ["MXMOE_BASELINE_CONFIG"])
        if "MXMOE_BASELINE_CONFIG" in os.environ
        else None,
        "baseline_latency_us": baseline_latency,
        "baseline_latency": baseline_statistics,
        "baseline_first_call_ms": baseline_first_call_ms,
        "baseline_correctness_passed": baseline_passed,
        "candidate": args.candidate,
        "candidate_config": json.loads(os.environ["MXMOE_KERNEL_CONFIG"])
        if "MXMOE_KERNEL_CONFIG" in os.environ
        else None,
        "candidate_latency_us": candidate_latency,
        "candidate_latency": candidate_statistics,
        "candidate_first_call_ms": candidate_first_call_ms,
        "speedup": baseline_latency / candidate_latency,
        "correctness": {
            "passed": candidate_passed,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "atol": atol,
            "rtol": rtol
        },
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rounds": args.rounds,
            "seed": args.seed,
            "timing_method": "paired alternating rounds; synchronized batched wall-clock",
            "library_log_level": args.library_log_level,
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not baseline_passed or not candidate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
