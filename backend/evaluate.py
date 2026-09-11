from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .baseline import baseline_action
from .config import BLOOD_GROUPS, DATA_DIR, NUM_HOSPITALS, SEED
from .env import BloodBankEnv


def rollout(policy, seed: int, model=None):
    env = BloodBankEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = (2 * baseline_action(env) - 1) if policy == "Baseline" else model.predict(obs, deterministic=True)[0]
        obs, _, done, _, _ = env.step(action)
    metrics = env.daily_metrics
    demand = sum(day["demand"] for day in metrics)
    short = sum(day["shortage"] for day in metrics)
    expired = sum(day["expired"] for day in metrics)
    breakdown = {name: float(sum(day["costs"][name] for day in metrics))
                 for name in ("wastage", "shortage", "holding", "transport")}
    return {
        "wastage_pct": 100 * expired / max(demand + expired, 1),
        "shortage_pct": 100 * short / max(demand, 1),
        "fill_rate": 100 * (demand - short) / max(demand, 1),
        "average_total_cost": float(np.mean([day["total_cost"] for day in metrics])),
        "average_holding_cost": float(np.mean([day["costs"]["holding"] for day in metrics])),
        "cost_breakdown": breakdown,
        "timeseries": metrics,
    }


def evaluate(episodes=5):
    model_path = Path(DATA_DIR.parent / "models" / "ppo_bloodbank.zip")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run python -m backend.train first.")
    model = PPO.load(model_path)
    results = {name: [rollout(name, SEED + i, model) for i in range(episodes)]
               for name in ("RL Policy", "Baseline")}
    metric_keys = ("wastage_pct", "shortage_pct", "fill_rate", "average_total_cost", "average_holding_cost")
    summary = {name: {key: round(float(np.mean([run[key] for run in runs])), 2) for key in metric_keys}
               for name, runs in results.items()}
    cost_breakdowns = {name: {component: round(float(np.mean([run["cost_breakdown"][component] for run in runs])), 2)
                               for component in ("wastage", "shortage", "holding", "transport")}
                       for name, runs in results.items()}
    comparison = {"episodes": episodes, "metrics": summary, "cost_breakdowns": cost_breakdowns}
    (DATA_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2))
    series = results["RL Policy"][0]["timeseries"]
    payload = {"nodes": ["Central Bank"] + [f"Hospital {i + 1}" for i in range(NUM_HOSPITALS)],
               "blood_groups": BLOOD_GROUPS, "series": series}
    (DATA_DIR / "inventory_timeseries.json").write_text(json.dumps(payload, indent=2))
    return comparison


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))