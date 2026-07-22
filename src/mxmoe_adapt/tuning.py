"""Joint objective for route alignment and expert GEMM tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config_schema import KernelConfig
from .route_features import RouteFeatures


@dataclass(frozen=True)
class Measurement:
    config: KernelConfig
    latency_us: float
    compile_ms: float
    max_abs_error: float
    verified: bool

    def __post_init__(self) -> None:
        if self.latency_us <= 0:
            raise ValueError("latency_us must be positive")
        if self.compile_ms < 0 or self.max_abs_error < 0:
            raise ValueError("compile time and error cannot be negative")


@dataclass(frozen=True)
class RankedMeasurement:
    measurement: Measurement
    padding_ratio: float
    score: float


def rank_measurements(
    measurements: Iterable[Measurement],
    route: RouteFeatures,
    padding_penalty_us: float = 1.0,
    compile_penalty: float = 0.001,
) -> list[RankedMeasurement]:
    """Rank verified candidates using measured latency plus lifecycle costs.

    Compile time is lightly weighted because tuning is offline. Padding remains
    explicit so the same GEMM tile can be compared across different alignments.
    """
    if padding_penalty_us < 0 or compile_penalty < 0:
        raise ValueError("penalties must be non-negative")
    ranked = []
    for measurement in measurements:
        if not measurement.verified:
            continue
        padding = route.padding_ratio(measurement.config.align_block_size)
        score = (
            measurement.latency_us
            + padding_penalty_us * padding
            + compile_penalty * measurement.compile_ms
        )
        ranked.append(RankedMeasurement(measurement, padding, score))
    return sorted(ranked, key=lambda item: item.score)


def select_best(
    measurements: Iterable[Measurement],
    route: RouteFeatures,
    padding_penalty_us: float = 1.0,
    compile_penalty: float = 0.001,
) -> RankedMeasurement | None:
    ranked = rank_measurements(
        measurements,
        route,
        padding_penalty_us=padding_penalty_us,
        compile_penalty=compile_penalty,
    )
    return ranked[0] if ranked else None
