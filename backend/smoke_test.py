"""Guardrails against both starvation and hoarding failure modes."""
from __future__ import annotations
from stable_baselines3 import PPO
from .config import MODEL_DIR, SEED
from .evaluate import evaluate, rollout

MAX_CONSECUTIVE_ZERO_DAYS = 3
MAX_CONSECUTIVE_HOARDING_DAYS = 3
SHORTAGE_REGRESSION_TOLERANCE_POINTS = 3.0
HOARDING_RATIO_THRESHOLD = 0.75


def max_zero_streak(timeseries: list[dict]) -> int:
    longest = current = 0
    for day in timeseries:
        empty_node = any(sum(node) == 0 for node in day["inventory"])
        current = current + 1 if empty_node else 0
        longest = max(longest, current)
    return longest


def max_bank_stock_streak(timeseries: list[dict], threshold: float = HOARDING_RATIO_THRESHOLD) -> int:
    if not timeseries:
        return 0
    bank_totals = [sum(day["inventory"][0]) for day in timeseries]
    peak = max(bank_totals, default=0)
    if peak == 0:
        return 0
    longest = current = 0
    for total in bank_totals:
        near_peak = total >= threshold * peak
        current = current + 1 if near_peak else 0
        longest = max(longest, current)
    return longest


def main() -> int:
    comparison = evaluate()
    episode = rollout("RL Policy", SEED, PPO.load(MODEL_DIR / "ppo_bloodbank"))
    rl, baseline = comparison["metrics"]["RL Policy"], comparison["metrics"]["Baseline"]
    failures = []
    zero_streak = max_zero_streak(episode["timeseries"])
    hoard_streak = max_bank_stock_streak(episode["timeseries"])
    if zero_streak > MAX_CONSECUTIVE_ZERO_DAYS:
        failures.append(f"COLLAPSED INVENTORY: empty for {zero_streak} consecutive days (limit {MAX_CONSECUTIVE_ZERO_DAYS}).")
    if hoard_streak > MAX_CONSECUTIVE_HOARDING_DAYS:
        failures.append(f"HOARDING REGRESSION: bank stock stayed near its peak for {hoard_streak} consecutive days (limit {MAX_CONSECUTIVE_HOARDING_DAYS}).")
    if rl["shortage_pct"] > baseline["shortage_pct"] + SHORTAGE_REGRESSION_TOLERANCE_POINTS:
        failures.append(f"SHORTAGE REGRESSION: RL {rl['shortage_pct']:.2f}% vs baseline {baseline['shortage_pct']:.2f}%.")
    if rl["wastage_pct"] >= baseline["wastage_pct"]:
        failures.append(f"WASTAGE REGRESSION: RL {rl['wastage_pct']:.2f}% vs baseline {baseline['wastage_pct']:.2f}%.")
    if rl["average_total_cost"] >= baseline["average_total_cost"]:
        failures.append(f"COST REGRESSION: RL {rl['average_total_cost']:.2f} vs baseline {baseline['average_total_cost']:.2f}.")
    print("Smoke test metrics:", comparison["metrics"])
    print(f"Longest zero-inventory streak: {zero_streak} day(s)")
    print(f"Longest near-peak bank stock streak: {hoard_streak} day(s)")
    if failures:
        print("\n".join(f"WARNING: {failure}" for failure in failures))
        return 1
    print("PASS: no starvation or hoarding pattern and RL meets all guardrails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())