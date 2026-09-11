from __future__ import annotations
import json
from collections import deque
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .config import DATA_DIR, MODEL_DIR, SEED
from .env import BloodBankEnv, set_reward_weights


class HistoryCallback(BaseCallback):
    """Records a smoothed episode-return curve rather than noisy daily costs."""
    def __init__(self):
        super().__init__()
        self.history, self.episode_rewards = [], []
        self.current_return = 0.0
        self.recent_returns = deque(maxlen=10)

    def _on_step(self):
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for reward, done, info in zip(rewards, dones, infos):
            self.current_return += float(reward)
            if done:
                self.recent_returns.append(self.current_return)
                self.history.append({
                    "step": self.num_timesteps,
                    "reward": round(float(np.mean(self.recent_returns)), 2),
                    "wastage": round(100 * info["expired"] / max(info["demand"] + info["expired"], 1), 2),
                    "shortage": round(100 * info["shortage"] / max(info["demand"], 1), 2),
                })
                self.current_return = 0.0
        return True


def train(timesteps=40000, shortage_cost=None, holding_cost=None, wastage_cost=None,
          transport_cost_per_km_unit=None, reward_scale=None, save_name="ppo_bloodbank"):
    """Train a CPU-friendly PPO policy with optional reward overrides for sweep runs."""
    set_reward_weights(
        shortage_cost=shortage_cost,
        holding_cost=holding_cost,
        wastage_cost=wastage_cost,
        transport_cost_per_km_unit=transport_cost_per_km_unit,
        reward_scale=reward_scale,
    )
    env = BloodBankEnv(seed=SEED)
    callback = HistoryCallback()
    model = PPO("MlpPolicy", env, seed=SEED, verbose=1, n_steps=1024, batch_size=128,
                learning_rate=2.5e-4, gamma=0.99, gae_lambda=0.95)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(MODEL_DIR / save_name)
    history = callback.history[::max(1, len(callback.history) // 150)]
    (DATA_DIR / "training_history.json").write_text(json.dumps({"history": history, "timesteps": timesteps}, indent=2))
    return model


if __name__ == "__main__":
    train()