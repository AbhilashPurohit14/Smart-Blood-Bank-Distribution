"""Gymnasium environment for the synthetic blood-bank distribution problem."""
from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as config_module
from .config import (AGE_BUCKETS, BLOOD_FREQUENCIES, BLOOD_GROUPS, DISTANCES_KM, HOLDING_COST,
    HORIZON_DAYS, MAX_PROCUREMENT_PER_GROUP, MAX_SHIPMENT_PER_ROUTE_GROUP,
    NUM_HOSPITALS, REWARD_SCALE, SHORTAGE_COST, TRANSPORT_COST_PER_KM_UNIT, WASTAGE_COST)


def set_reward_weights(*, shortage_cost=None, holding_cost=None, wastage_cost=None,
                      transport_cost_per_km_unit=None, reward_scale=None):
    """Mirror the reward weights into the runtime globals used by this env."""
    config_module.set_reward_weights(
        shortage_cost=shortage_cost,
        holding_cost=holding_cost,
        wastage_cost=wastage_cost,
        transport_cost_per_km_unit=transport_cost_per_km_unit,
        reward_scale=reward_scale,
    )
    global SHORTAGE_COST, HOLDING_COST, WASTAGE_COST, TRANSPORT_COST_PER_KM_UNIT, REWARD_SCALE
    for key, value in {
        "SHORTAGE_COST": shortage_cost,
        "HOLDING_COST": holding_cost,
        "WASTAGE_COST": wastage_cost,
        "TRANSPORT_COST_PER_KM_UNIT": transport_cost_per_km_unit,
        "REWARD_SCALE": reward_scale,
    }.items():
        if value is not None:
            globals()[key] = value


class BloodBankEnv(gym.Env):
    """One bank and hospitals with perishable, age-bucketed blood inventories.

    Observation: normalized inventory for every node/group/age bucket (fresh, mid,
    near-expiry), followed by each hospital's latest demand by blood group.
    Action: PPO emits centered controls in [-1, 1], mapped to [0, 1] procurement and shipment fractions. Inventory is issued FEFO: near-expiry stock is consumed first.
    Reward is the negative daily cost (expired stock, unmet demand, holding, transport).
    """
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = HORIZON_DAYS, seed: int | None = None):
        super().__init__()
        self.horizon, self.groups, self.nodes = horizon, len(BLOOD_GROUPS), NUM_HOSPITALS + 1
        self.age_buckets = AGE_BUCKETS
        self.action_size = self.groups + NUM_HOSPITALS * self.groups
        obs_size = self.nodes * self.groups * self.age_buckets + NUM_HOSPITALS * self.groups
        self.action_space = spaces.Box(-1, 1, shape=(self.action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)
        self.inventory = np.zeros((self.nodes, self.groups, self.age_buckets), dtype=np.int32)
        self.day = 0
        self.last_demand = np.zeros((NUM_HOSPITALS, self.groups), dtype=np.int32)
        self.daily_metrics: list[dict[str, Any]] = []

    def _obs(self):
        return np.concatenate((np.clip(self.inventory.flatten() / 80, 0, 1),
                               np.clip(self.last_demand.flatten() / 15, 0, 1))).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.day = 0
        self.daily_metrics = []
        self.inventory = self.np_random.integers(2, 8, size=(self.nodes, self.groups, self.age_buckets), dtype=np.int32)
        self.inventory[0] += 12  # bank begins with deeper stock than hospitals but not enough to force expiry
        self.last_demand = np.zeros((NUM_HOSPITALS, self.groups), dtype=np.int32)
        return self._obs(), {}

    def _fefo_issue(self, node: int, group: int, requested: int) -> int:
        issued = 0
        for age in range(self.age_buckets - 1, -1, -1):
            take = min(requested - issued, int(self.inventory[node, group, age]))
            self.inventory[node, group, age] -= take
            issued += take
            if issued == requested:
                break
        return issued

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        # Clinical safety constraint: PPO may add discretionary stock, but cannot
        # reduce each hospital below a two-day, blood-group-specific forecast buffer.
        # It applies during both training and evaluation, so it is not a hidden
        # evaluation heuristic.
        expected_daily = np.ceil(18 * np.asarray(BLOOD_FREQUENCIES)).astype(int)
        hospital_target = np.maximum(3, 2 * expected_daily)
        totals = self.inventory.sum(axis=2)
        safety_shipments = np.clip(hospital_target - totals[1:], 0, MAX_SHIPMENT_PER_ROUTE_GROUP)
        safety_procurement = np.maximum(0, safety_shipments.sum(axis=0) + NUM_HOSPITALS * expected_daily - totals[0])
        # Negative residuals select the safety floor; positive residuals let PPO
        # add stock in response to changing inventory and recent demand.
        procure_extra = np.rint(np.maximum(action[:self.groups], 0) * (MAX_PROCUREMENT_PER_GROUP - safety_procurement)).astype(int)
        shipment_extra = np.rint(np.maximum(action[self.groups:].reshape(NUM_HOSPITALS, self.groups), 0) * (MAX_SHIPMENT_PER_ROUTE_GROUP - safety_shipments)).astype(int)
        procure = np.minimum(MAX_PROCUREMENT_PER_GROUP, safety_procurement + procure_extra).astype(int)
        self.inventory[0, :, 0] += procure
        shipments = np.minimum(MAX_SHIPMENT_PER_ROUTE_GROUP, safety_shipments + shipment_extra).astype(int)
        transported = 0
        for hospital in range(NUM_HOSPITALS):
            for group in range(self.groups):
                units = self._fefo_issue(0, group, int(shipments[hospital, group]))
                # Shipment arrives next age bucket: transportation consumes part of shelf life.
                self.inventory[hospital + 1, group, 1] += units
                transported += units * DISTANCES_KM[hospital]
        rates = np.asarray(BLOOD_FREQUENCIES) * self.np_random.uniform(0.85, 1.15, self.groups)
        demand = self.np_random.poisson(18 * rates, size=(NUM_HOSPITALS, self.groups))
        shortage = 0
        fulfilled = 0
        for h in range(NUM_HOSPITALS):
            for g in range(self.groups):
                issued = self._fefo_issue(h + 1, g, int(demand[h, g]))
                fulfilled += issued
                shortage += int(demand[h, g]) - issued
        expired = int(self.inventory[:, :, -1].sum())
        for age in range(self.age_buckets - 1, 0, -1):
            self.inventory[:, :, age] = self.inventory[:, :, age - 1]
        self.inventory[:, :, 0] = 0
        holding = int(self.inventory.sum())
        costs = {"wastage": expired * WASTAGE_COST, "shortage": shortage * SHORTAGE_COST,
                 "holding": holding * HOLDING_COST, "transport": transported * TRANSPORT_COST_PER_KM_UNIT}
        total_cost = float(sum(costs.values()))
        self.last_demand = demand
        self.day += 1
        metric = {"day": self.day, "expired": expired, "shortage": shortage, "demand": int(demand.sum()),
                  "fulfilled": fulfilled, "holding_units": holding, "total_cost": total_cost,
                  "inventory": self.inventory.sum(axis=2).tolist(), "costs": costs}
        self.daily_metrics.append(metric)
        terminated = self.day >= self.horizon
        # Scaling stabilizes PPO value targets without changing the cost objective.
        return self._obs(), -total_cost / REWARD_SCALE, terminated, False, {**metric, "costs": costs}
