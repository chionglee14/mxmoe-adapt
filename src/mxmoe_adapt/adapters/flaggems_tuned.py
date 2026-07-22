"""FlagGems Fused MoE adapter with an explicit experimental kernel config."""

from __future__ import annotations

import importlib
import json
import os
from functools import lru_cache
from threading import Lock
from typing import Any

from mxmoe_adapt.config_schema import KernelConfig


_PATCH_LOCK = Lock()


@lru_cache(maxsize=4)
def _config_from_environment(variable: str = "MXMOE_KERNEL_CONFIG") -> KernelConfig:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} must contain a JSON kernel configuration")
    config = KernelConfig.from_mapping(json.loads(raw))
    if config.align_block_size != config.block_size_m:
        raise ValueError(
            "FlagGems v5.0.2 couples MoE alignment to BLOCK_SIZE_M; "
            "align_block_size must equal block_size_m in the M0 adapter"
        )
    return config


def _run_with_config(
    variable: str,
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
) -> Any:
    module = importlib.import_module("flag_gems.fused.fused_moe")
    config = _config_from_environment(variable).to_flaggems()
    with _PATCH_LOCK:
        original = module.try_get_optimal_moe_config
        module.try_get_optimal_moe_config = lambda *args, **kwargs: dict(config)
        try:
            return module.outplace_fused_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation="silu",
            )
        finally:
            module.try_get_optimal_moe_config = original


def run(
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
) -> Any:
    return _run_with_config(
        "MXMOE_KERNEL_CONFIG", hidden_states, w1, w2, topk_weights, topk_ids
    )


def run_baseline(
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
) -> Any:
    return _run_with_config(
        "MXMOE_BASELINE_CONFIG", hidden_states, w1, w2, topk_weights, topk_ids
    )
