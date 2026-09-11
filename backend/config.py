from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
NUM_HOSPITALS = 4
HORIZON_DAYS = 30
AGE_BUCKETS = 5
BLOOD_GROUPS = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
BLOOD_FREQUENCIES = [0.37, 0.01, 0.23, 0.01, 0.30, 0.01, 0.06, 0.01]
DISTANCES_KM = [7, 12, 18, 24]

# Cost weights used directly by the environment reward (higher reward = lower cost).
WASTAGE_COST = 18.0
# A missed blood request is a clinical service failure, so it must dominate the
# avoidable operational costs of carrying, expiring, or moving a unit. This is
# larger than wastage cost, while holding cost still prevents overstocking.
SHORTAGE_COST = 220.0
HOLDING_COST = 0.35
TRANSPORT_COST_PER_KM_UNIT = 0.08
# Keep PPO's value targets in a stable range; this does not alter cost trade-offs.
REWARD_SCALE = 100.0
MAX_PROCUREMENT_PER_GROUP = 30
MAX_SHIPMENT_PER_ROUTE_GROUP = 20


def set_reward_weights(*, shortage_cost=None, holding_cost=None, wastage_cost=None,
                      transport_cost_per_km_unit=None, reward_scale=None):
    """Synchronize reward coefficients across the config and runtime modules."""
    overrides = {
        "SHORTAGE_COST": shortage_cost,
        "HOLDING_COST": holding_cost,
        "WASTAGE_COST": wastage_cost,
        "TRANSPORT_COST_PER_KM_UNIT": transport_cost_per_km_unit,
        "REWARD_SCALE": reward_scale,
    }
    for key, value in overrides.items():
        if value is not None:
            globals()[key] = value
    return {
        key: globals()[key]
        for key, value in overrides.items()
        if value is not None
    }