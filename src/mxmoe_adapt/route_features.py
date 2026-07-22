"""Routing-distribution features used by tuning and dispatch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RouteFeatures:
    tokens: int
    assignments: int
    experts: int
    top_k: int
    nonempty_experts: int
    empty_ratio: float
    mean_tokens_per_expert: float
    max_tokens_per_expert: int
    coefficient_of_variation: float
    normalized_entropy: float
    counts: tuple[int, ...]

    def padding_ratio(self, block_size_m: int) -> float:
        if block_size_m <= 0:
            raise ValueError("block_size_m must be positive")
        if self.assignments == 0:
            return 0.0
        padded = sum(
            math.ceil(count / block_size_m) * block_size_m
            for count in self.counts
            if count
        )
        return padded / self.assignments - 1.0

    def route_class(self) -> str:
        if self.mean_tokens_per_expert <= 4 or self.empty_ratio >= 0.5:
            return "decode_sparse"
        if self.coefficient_of_variation >= 1.0:
            return "skewed"
        return "prefill_dense"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _flatten_ids(
    expert_ids: Iterable[int] | Iterable[Sequence[int]], top_k: int | None
) -> tuple[list[int], int, int]:
    rows = list(expert_ids)
    if not rows:
        if top_k is None or top_k <= 0:
            raise ValueError("top_k must be positive for an empty route")
        return [], 0, top_k

    first = rows[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        nested = [list(row) for row in rows]  # type: ignore[arg-type]
        inferred = len(nested[0])
        if inferred <= 0 or any(len(row) != inferred for row in nested):
            raise ValueError("all route rows must have the same positive top_k")
        if top_k is not None and top_k != inferred:
            raise ValueError(f"top_k={top_k} does not match route width {inferred}")
        return [int(item) for row in nested for item in row], len(nested), inferred

    if top_k is None or top_k <= 0:
        raise ValueError("top_k is required for a flat expert-id sequence")
    flat = [int(item) for item in rows]  # type: ignore[arg-type]
    if len(flat) % top_k:
        raise ValueError("flat expert-id count must be divisible by top_k")
    return flat, len(flat) // top_k, top_k


def extract_route_features(
    expert_ids: Iterable[int] | Iterable[Sequence[int]],
    num_experts: int,
    top_k: int | None = None,
) -> RouteFeatures:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    flat, tokens, resolved_top_k = _flatten_ids(expert_ids, top_k)
    counts = [0] * num_experts
    for expert_id in flat:
        if expert_id < 0 or expert_id >= num_experts:
            raise ValueError(f"expert id {expert_id} outside [0, {num_experts})")
        counts[expert_id] += 1

    assignments = len(flat)
    mean = assignments / num_experts
    variance = sum((count - mean) ** 2 for count in counts) / num_experts
    cv = math.sqrt(variance) / mean if mean else 0.0
    if assignments and num_experts > 1:
        probabilities = [count / assignments for count in counts if count]
        entropy = -sum(p * math.log(p) for p in probabilities)
        normalized_entropy = entropy / math.log(num_experts)
    else:
        normalized_entropy = 0.0

    nonempty = sum(count > 0 for count in counts)
    return RouteFeatures(
        tokens=tokens,
        assignments=assignments,
        experts=num_experts,
        top_k=resolved_top_k,
        nonempty_experts=nonempty,
        empty_ratio=(num_experts - nonempty) / num_experts,
        mean_tokens_per_expert=mean,
        max_tokens_per_expert=max(counts, default=0),
        coefficient_of_variation=cv,
        normalized_entropy=normalized_entropy,
        counts=tuple(counts),
    )
