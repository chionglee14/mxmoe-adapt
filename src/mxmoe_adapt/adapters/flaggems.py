"""FlagGems public Fused MoE baseline adapter."""

from __future__ import annotations

from typing import Any


def run(
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
) -> Any:
    try:
        from flag_gems.fused.fused_moe import outplace_fused_experts
    except ImportError as error:
        raise RuntimeError(
            "FlagGems fused MoE API is unavailable. Record the FlagGems commit and "
            "update this adapter if the upstream API changed."
        ) from error
    return outplace_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
    )
