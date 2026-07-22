"""Validated schemas for C500 MoE kernel configurations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class KernelConfig:
    block_size_m: int
    block_size_n: int
    block_size_k: int
    group_size_m: int
    num_warps: int
    num_stages: int
    align_block_size: int = 16

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "KernelConfig":
        aliases = {
            "block_size_m": ("block_size_m", "BLOCK_SIZE_M"),
            "block_size_n": ("block_size_n", "BLOCK_SIZE_N"),
            "block_size_k": ("block_size_k", "BLOCK_SIZE_K"),
            "group_size_m": ("group_size_m", "GROUP_SIZE_M"),
            "num_warps": ("num_warps",),
            "num_stages": ("num_stages",),
            "align_block_size": ("align_block_size", "ALIGN_BLOCK_SIZE"),
        }
        parsed: dict[str, Any] = {}
        for field, keys in aliases.items():
            match = next((value[key] for key in keys if key in value), None)
            if match is None and field == "align_block_size":
                match = value.get("BLOCK_SIZE_M", value.get("block_size_m", 16))
            if match is None:
                raise ValueError(f"missing kernel config field: {field}")
            parsed[field] = match
        return cls(**parsed)

    def to_flaggems(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_size_m,
            "BLOCK_SIZE_N": self.block_size_n,
            "BLOCK_SIZE_K": self.block_size_k,
            "GROUP_SIZE_M": self.group_size_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def estimated_smem_bytes(self, dtype_bytes: int = 2) -> int:
        """Return a conservative tile-storage proxy, not a compiler measurement."""
        if dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")
        per_stage = (
            self.block_size_m * self.block_size_k
            + self.block_size_k * self.block_size_n
        ) * dtype_bytes
        return per_stage * self.num_stages


@dataclass(frozen=True)
class SearchConstraints:
    # Configurable: the target C500 compiler/profiler remains the source of truth.
    shared_memory_budget_bytes: int = 65_536
    allowed_warps: tuple[int, ...] = (2, 4, 8)
    max_stages: int = 3

    def accepts(self, config: KernelConfig, dtype_bytes: int = 2) -> bool:
        return (
            config.num_warps in self.allowed_warps
            and config.num_stages <= self.max_stages
            and config.estimated_smem_bytes(dtype_bytes) <= self.shared_memory_budget_bytes
        )
