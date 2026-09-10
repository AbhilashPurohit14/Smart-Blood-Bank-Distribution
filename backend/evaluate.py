from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .baseline import baseline_action
from .config import BLOOD_GROUPS, DATA_DIR, DISTANCES_KM, NUM_HOSPITALS, SEED
from .env import BloodBankEnv

def rollout(policy, seed: int, model=None):
    env = BloodBankEnv(seed=seed)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = baseline_action(env) if policy == "Baseline" else model.predict(obs, deterministic=True)[0]
        obs, _, done, _, _ = env.step(action)
    m = env.daily_metrics
    demand, short, expired = sum(x['demand'] for x in m), sum(x['shortage'] for x in m), sum(x['expired'] for x in m)
    return {"wastage_pct": 100 * expired / max(demand + expired, 1), "shortage_pct": 100 * short / max(demand, 1),
            "fill_rate": 100 * (demand - short) / max(demand, 1), "average_total_cost": np.mean([x['total_cost'] for x in m]),
            "average_holding_cost": np.mean([x['holding_units'] * .35 for x in m]), "timeseries": m}

def evaluate(episodes=5):
    model_path = Path(DATA_DIR.parent / 'models' / 'ppo_bloodbank.zip')
    if not model_path.exists():
        raise FileNotFoundError('Model not found. Run python -m backend.train first.')
    model = PPO.load(model_path)
    results = {name: [rollout(name, SEED + i, model) for i in range(episodes)] for name in ('RL Policy', 'Baseline')}
    summary = {name: {k: round(float(np.mean([r[k] for r in runs])), 2) for k in runs[0] if k != 'timeseries'} for name, runs in results.items()}
    comparison = {"episodes": episodes, "metrics": summary}
    (DATA_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    series = results['RL Policy'][0]['timeseries']
    payload = {"nodes": ['Central Bank'] + [f'Hospital {i+1}' for i in range(NUM_HOSPITALS)], "blood_groups": BLOOD_GROUPS, "series": series}
    (DATA_DIR / 'inventory_timeseries.json').write_text(json.dumps(payload, indent=2))
    return comparison

if __name__ == '__main__': print(json.dumps(evaluate(), indent=2))
