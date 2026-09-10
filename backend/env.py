"""Gymnasium environment for the synthetic blood-bank distribution problem."""
from __future__ import annotations

from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import (BLOOD_FREQUENCIES, BLOOD_GROUPS, DISTANCES_KM, HOLDING_COST,
    HORIZON_DAYS, MAX_PROCUREMENT_PER_GROUP, MAX_SHIPMENT_PER_ROUTE_GROUP,
    NUM_HOSPITALS, SHORTAGE_COST, TRANSPORT_COST_PER_KM_UNIT, WASTAGE_COST)


class BloodBankEnv(gym.Env):
    """One bank and hospitals with perishable, age-bucketed blood inventories.

    Observation: normalized inventory for every node/group/age bucket (fresh, mid,
    near-expiry), followed by each hospital's latest demand by blood group.
    Action: first 8 continuous values procure stock at the bank; remaining values are
    shipment quantities, indexed hospital then blood group. Values are scaled to their
    configured maxima. Inventory is issued FEFO: near-expiry stock is consumed first.
    Reward is the negative daily cost (expired stock, unmet demand, holding, transport).
    """
    metadata = {"render_modes": []}

    def __init__(self, horizon: int = HORIZON_DAYS, seed: int | None = None):
        super().__init__()
        self.horizon, self.groups, self.nodes = horizon, len(BLOOD_GROUPS), NUM_HOSPITALS + 1
        self.action_size = self.groups + NUM_HOSPITALS * self.groups
        obs_size = self.nodes * self.groups * 3 + NUM_HOSPITALS * self.groups
        self.action_space = spaces.Box(0, 1, shape=(self.action_size,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, shape=(obs_size,), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)
        self.inventory = np.zeros((self.nodes, self.groups, 3), dtype=np.int32)
        self.day = 0
        self.last_demand = np.zeros((NUM_HOSPITALS, self.groups), dtype=np.int32)
        self.daily_metrics: list[dict[str, Any]] = []

    def _obs(self):
        return np.concatenate((np.clip(self.inventory.flatten() / 120, 0, 1),
                               np.clip(self.last_demand.flatten() / 15, 0, 1))).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.day = 0
        self.daily_metrics = []
        self.inventory = self.np_random.integers(4, 15, size=(self.nodes, self.groups, 3), dtype=np.int32)
        self.inventory[0] += 18  # bank begins with deeper stock than hospitals
        self.last_demand = np.zeros((NUM_HOSPITALS, self.groups), dtype=np.int32)
        return self._obs(), {}

    def _fefo_issue(self, node: int, group: int, requested: int) -> int:
        issued = 0
        for age in (2, 1, 0):
            take = min(requested - issued, int(self.inventory[node, group, age]))
            self.inventory[node, group, age] -= take
            issued += take
            if issued == requested:
                break
        return issued

    def step(self, action):
        action = np.clip(np.asarray(action), 0, 1)
        procure = np.rint(action[:self.groups] * MAX_PROCUREMENT_PER_GROUP).astype(int)
        self.inventory[0, :, 0] += procure
        shipments = np.rint(action[self.groups:].reshape(NUM_HOSPITALS, self.groups) * MAX_SHIPMENT_PER_ROUTE_GROUP).astype(int)
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
        expired = int(self.inventory[:, :, 2].sum())
        self.inventory[:, :, 2] = self.inventory[:, :, 1]
        self.inventory[:, :, 1] = self.inventory[:, :, 0]
        self.inventory[:, :, 0] = 0
        holding = int(self.inventory.sum())
        costs = {"wastage": expired * WASTAGE_COST, "shortage": shortage * SHORTAGE_COST,
                 "holding": holding * HOLDING_COST, "transport": transported * TRANSPORT_COST_PER_KM_UNIT}
        total_cost = float(sum(costs.values()))
        self.last_demand = demand
        self.day += 1
        metric = {"day": self.day, "expired": expired, "shortage": shortage, "demand": int(demand.sum()),
                  "fulfilled": fulfilled, "holding_units": holding, "total_cost": total_cost,
                  "inventory": self.inventory.sum(axis=2).tolist()}
        self.daily_metrics.append(metric)
        terminated = self.day >= self.horizon
        return self._obs(), -total_cost, terminated, False, {**metric, "costs": costs}
