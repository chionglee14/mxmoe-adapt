"""Clear PyTorch MoE reference for correctness checks, not performance."""

from __future__ import annotations

from typing import Any


def run(
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
) -> Any:
    import torch
    import torch.nn.functional as functional

    output = torch.zeros_like(hidden_states)
    for token in range(hidden_states.shape[0]):
        token_output = torch.zeros_like(hidden_states[token])
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            gate_up = functional.linear(hidden_states[token], w1[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = functional.silu(gate) * up
            expert_output = functional.linear(intermediate, w2[expert])
            token_output.add_(expert_output * topk_weights[token, slot])
        output[token] = token_output
    return output
