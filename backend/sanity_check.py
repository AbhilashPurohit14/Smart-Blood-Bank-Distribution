from .baseline import baseline_action
from .env import BloodBankEnv
env = BloodBankEnv(); obs, _ = env.reset(seed=42)
done = False; reward = 0
while not done:
    obs, r, done, _, _ = env.step(baseline_action(env)); reward += r
print({'days': env.day, 'total_reward': round(reward, 2), 'last_day': env.daily_metrics[-1]})
