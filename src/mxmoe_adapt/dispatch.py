"""Versioned configuration database and lightweight runtime selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config_schema import KernelConfig


@dataclass(frozen=True)
class Workload:
    tokens: int
    experts: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    dtype: str
    route_class: str

    def __post_init__(self) -> None:
        for field in ("tokens", "experts", "hidden_size", "intermediate_size", "top_k"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.dtype not in {"fp16", "bf16"}:
            raise ValueError("initial release supports only fp16 and bf16")


@dataclass(frozen=True)
class ConfigEntry:
    workload: Workload
    config: KernelConfig
    latency_us: float
    verified: bool
    environment_id: str


class ConfigDatabase:
    def __init__(self, entries: Iterable[ConfigEntry], schema_version: int = 1) -> None:
        if schema_version != 1:
            raise ValueError(f"unsupported config database schema: {schema_version}")
        self.entries = tuple(entries)
        self.schema_version = schema_version

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ConfigDatabase":
        entries = []
        for raw in payload.get("entries", []):
            entries.append(
                ConfigEntry(
                    workload=Workload(**raw["workload"]),
                    config=KernelConfig.from_mapping(raw["config"]),
                    latency_us=float(raw["latency_us"]),
                    verified=bool(raw.get("verified", False)),
                    environment_id=str(raw.get("environment_id", "unknown")),
                )
            )
        return cls(entries, int(payload.get("schema_version", 1)))

    @classmethod
    def load(cls, path: str | Path) -> "ConfigDatabase":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _distance(query: Workload, candidate: Workload) -> float:
        if query.dtype != candidate.dtype:
            return math.inf
        numeric = (
            (query.tokens, candidate.tokens, 3.0),
            (query.experts, candidate.experts, 1.5),
            (query.hidden_size, candidate.hidden_size, 1.0),
            (query.intermediate_size, candidate.intermediate_size, 1.0),
            (query.top_k, candidate.top_k, 2.0),
        )
        distance = sum(weight * abs(math.log2(a / b)) for a, b, weight in numeric)
        if query.route_class != candidate.route_class:
            distance += 2.5
        return distance

    def select(
        self,
        workload: Workload,
        environment_id: str | None = None,
        allow_unverified: bool = False,
        max_distance: float = 6.0,
    ) -> ConfigEntry | None:
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative")
        pool = [entry for entry in self.entries if allow_unverified or entry.verified]
        if environment_id:
            exact_environment = [
                entry for entry in pool if entry.environment_id == environment_id
            ]
            if not exact_environment:
                return None
            pool = exact_environment
        if not pool:
            return None
        selected = min(pool, key=lambda entry: self._distance(workload, entry.workload))
        distance = self._distance(workload, selected.workload)
        if math.isinf(distance) or distance > max_distance:
            return None
        return selected
