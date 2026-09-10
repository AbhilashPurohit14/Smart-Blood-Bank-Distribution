from __future__ import annotations
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .config import DATA_DIR, MODEL_DIR, SEED
from .env import BloodBankEnv

class HistoryCallback(BaseCallback):
    def __init__(self): super().__init__(); self.history = []
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'total_cost' in info:
                self.history.append({"step": self.num_timesteps, "reward": round(-info['total_cost'], 2),
                  "wastage": round(100 * info['expired'] / max(info['demand'] + info['expired'], 1), 2),
                  "shortage": round(100 * info['shortage'] / max(info['demand'], 1), 2)})
        return True

def train(timesteps=30000):
    env = BloodBankEnv(seed=SEED)
    cb = HistoryCallback()
    model = PPO('MlpPolicy', env, seed=SEED, verbose=1, n_steps=512, batch_size=64, learning_rate=3e-4, gamma=.98)
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(MODEL_DIR / 'ppo_bloodbank')
    # downsample to make the dashboard payload compact
    history = cb.history[::max(1, len(cb.history)//150)]
    (DATA_DIR / 'training_history.json').write_text(json.dumps({"history": history, "timesteps": timesteps}, indent=2))
    return model
if __name__ == '__main__': train()
