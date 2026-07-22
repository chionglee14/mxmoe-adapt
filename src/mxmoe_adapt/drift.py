"""Detect stack or performance drift that invalidates tuned C500 configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DriftReport:
    environment_changed: bool
    performance_regression: bool
    latency_change_ratio: float | None
    reasons: tuple[str, ...]

    @property
    def retune_required(self) -> bool:
        return self.environment_changed or self.performance_regression


def detect_drift(
    baseline_environment: Mapping[str, Any],
    current_environment: Mapping[str, Any],
    baseline_latency_us: float | None = None,
    current_latency_us: float | None = None,
    regression_threshold: float = 0.05,
) -> DriftReport:
    reasons: list[str] = []
    baseline_id = baseline_environment.get("environment_id")
    current_id = current_environment.get("environment_id")
    environment_changed = baseline_id != current_id
    if environment_changed:
        reasons.append(f"environment_id changed: {baseline_id} -> {current_id}")

    latency_change: float | None = None
    performance_regression = False
    if baseline_latency_us is not None and current_latency_us is not None:
        if baseline_latency_us <= 0 or current_latency_us <= 0:
            raise ValueError("latencies must be positive")
        latency_change = current_latency_us / baseline_latency_us - 1.0
        performance_regression = latency_change > regression_threshold
        if performance_regression:
            reasons.append(
                f"latency regressed by {latency_change:.1%}, threshold={regression_threshold:.1%}"
            )

    return DriftReport(
        environment_changed=environment_changed,
        performance_regression=performance_regression,
        latency_change_ratio=latency_change,
        reasons=tuple(reasons),
    )
