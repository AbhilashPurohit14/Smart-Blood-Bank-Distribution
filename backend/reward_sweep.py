from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .config import BLOOD_GROUPS, DATA_DIR, MODEL_DIR, SEED, set_reward_weights
from .env import BloodBankEnv
from .evaluate import rollout
from .train import train


def evaluate_config(shortage_cost: float, holding_cost: float, timesteps: int = 15000, label: str | None = None):
    model_name = f"ppo_sweep_{label or f's{shortage_cost}_h{holding_cost}'}"
    set_reward_weights(shortage_cost=shortage_cost, holding_cost=holding_cost)
    model = train(timesteps=timesteps, shortage_cost=shortage_cost, holding_cost=holding_cost, save_name=model_name)
    runs = [rollout("RL Policy", SEED + i, model) for i in range(5)]
    summary = {
        "config": {"shortage_cost": shortage_cost, "holding_cost": holding_cost},
        "wastage_pct": round(float(np.mean([r["wastage_pct"] for r in runs])), 2),
        "shortage_pct": round(float(np.mean([r["shortage_pct"] for r in runs])), 2),
        "fill_rate": round(float(np.mean([r["fill_rate"] for r in runs])), 2),
        "average_total_cost": round(float(np.mean([r["average_total_cost"] for r in runs])), 2),
        "cost_breakdown": {
            component: round(float(np.mean([r["cost_breakdown"][component] for r in runs])), 2)
            for component in ("wastage", "shortage", "holding", "transport")
        },
    }
    return summary


if __name__ == "__main__":
    configs = [
        (80.0, 0.12),
        (120.0, 0.20),
        (160.0, 0.28),
        (220.0, 0.35),
        (320.0, 0.45),
        (440.0, 0.65),
    ]
    results = [evaluate_config(shortage_cost, holding_cost, timesteps=20000, label=f"{shortage_cost:g}_{holding_cost:g}") for shortage_cost, holding_cost in configs]
    (DATA_DIR / "reward_sweep.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
