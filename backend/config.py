from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
NUM_HOSPITALS = 4
HORIZON_DAYS = 30  # ASSUMPTION: 30 keeps local demos and API simulations responsive.
BLOOD_GROUPS = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
# Approximate Indian ABO/Rh frequencies, normalized for synthetic demand only.
# Source: Agrawal et al., Asian J Transfus Sci. 2014; Rh-negative groups are rare.
BLOOD_FREQUENCIES = [0.37, 0.01, 0.23, 0.01, 0.30, 0.01, 0.06, 0.01]
DISTANCES_KM = [7, 12, 18, 24]

# Cost weights used directly by the environment reward (higher reward = lower cost).
WASTAGE_COST = 18.0
SHORTAGE_COST = 55.0
HOLDING_COST = 0.35
TRANSPORT_COST_PER_KM_UNIT = 0.08
MAX_PROCUREMENT_PER_GROUP = 80
MAX_SHIPMENT_PER_ROUTE_GROUP = 35
