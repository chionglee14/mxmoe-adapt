"""Hardware-constrained joint alignment/GEMM search-space generator."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

from .config_schema import KernelConfig, SearchConstraints


def generate_search_space(
    constraints: SearchConstraints | None = None,
    dtype_bytes: int = 2,
) -> list[KernelConfig]:
    limits = constraints or SearchConstraints()
    candidates: list[KernelConfig] = []
    axes = itertools.product(
        (16, 32, 64, 128),
        (32, 64, 128, 256),
        (32, 64, 128),
        (1, 4, 8, 16),
        limits.allowed_warps,
        range(1, limits.max_stages + 1),
        (8, 16, 32, 64),
    )
    for values in axes:
        config = KernelConfig(*values)
        if limits.accepts(config, dtype_bytes=dtype_bytes):
            candidates.append(config)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smem-bytes", type=int, default=65_536)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    args = parser.parse_args()
    constraints = SearchConstraints(shared_memory_budget_bytes=args.smem_bytes)
    payload = {
        "schema_version": 1,
        "warning": "estimated shared-memory filtering only; validate with the C500 compiler",
        "constraints": asdict(constraints),
        "candidates": [asdict(config) for config in generate_search_space(constraints, args.dtype_bytes)],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
