from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    default_risk_level: str = "medium"
    default_strategy: str = "hybrid_default"
    default_loop_interval_seconds: int = 60


def get_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        default_risk_level=os.getenv("RUNTIME_RISK_LEVEL", "medium"),
        default_strategy=os.getenv("RUNTIME_STRATEGY", "hybrid_default"),
        default_loop_interval_seconds=int(os.getenv("RUNTIME_LOOP_INTERVAL", "60")),
    )

