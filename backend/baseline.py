"""Classical order-up-to policy: a transparent non-RL comparison policy."""
import numpy as np
from .config import NUM_HOSPITALS

def baseline_action(env):
    """Replenish low bank stock and ship each hospital toward a modest target level."""
    totals = env.inventory.sum(axis=2)
    action = np.zeros(env.action_size, dtype=np.float32)
    action[:env.groups] = np.clip((90 - totals[0]) / 80, 0, 1)
    shipments = np.zeros((NUM_HOSPITALS, env.groups))
    for h in range(NUM_HOSPITALS):
        target = 22
        shipments[h] = np.clip((target - totals[h + 1]) / 35, 0, 1)
    action[env.groups:] = shipments.flatten()
    return action
