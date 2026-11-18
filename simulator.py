"""
Simulator for a weekly budgeted recommendation problem.

Each week the recommender selects a set of items whose total price fits the
budget, then recommends one item from that set to every user. Each user accepts
the recommendation with probability P[user, item]. The reward is the number of
accepted recommendations summed over all weeks.
"""

import itertools
import time

import numpy as np


def generate_problem(n_users, n_items, budget, n_weeks, seed=0):
    """Create a random problem instance with reproducible acceptance probabilities."""
    rng = np.random.default_rng(seed)
    P = rng.beta(a=1.0, b=4.0, size=(n_users, n_items))
    prices = rng.integers(low=5, high=31, size=n_items)
    return {"P": P, "item_prices": prices, "budget": int(budget), "n_weeks": int(n_weeks)}


def oracle_reward_per_week(P, prices, budget):
    """
    Best expected reward per week for a fixed budget-feasible item set,
    found by enumeration. Only practical for a small number of items.
    """
    n_items = P.shape[1]
    best = 0.0
    for size in range(1, n_items + 1):
        for subset in itertools.combinations(range(n_items), size):
            subset = list(subset)
            if prices[subset].sum() > budget:
                continue
            best = max(best, P[:, subset].max(axis=1).sum())
    return best


class Simulation:
    """Runs one simulated game for a given recommender class."""

    def __init__(self, P, prices, budget, n_weeks, time_limit=120.0):
        self.P = np.asarray(P, dtype=float).copy()
        self.item_prices = np.asarray(prices).copy()
        self.budget = budget
        self.n_weeks = n_weeks
        self.time_limit = time_limit

    def _is_valid(self, recommendation):
        if not isinstance(recommendation, np.ndarray):
            return False
        if not np.issubdtype(recommendation.dtype, np.integer):
            return False
        if recommendation.shape != (self.P.shape[0],):
            return False
        if ((recommendation < 0) | (recommendation >= self.P.shape[1])).any():
            return False
        items = np.unique(recommendation)
        return self.item_prices[items].sum() <= self.budget

    def run(self, recommender_cls, seed=None):
        """
        Returns (reward, elapsed_seconds). Stops early if the time limit is
        exceeded; the reward collected so far is returned in that case.
        """
        if seed is not None:
            np.random.seed(seed)

        n_users, n_items = self.P.shape
        start = time.perf_counter()
        recommender = recommender_cls(
            n_weeks=self.n_weeks,
            n_users=n_users,
            prices=self.item_prices.copy(),
            budget=self.budget,
        )
        reward = 0
        for _ in range(self.n_weeks):
            if time.perf_counter() - start > self.time_limit:
                break
            recommendation = recommender.recommend()
            if not self._is_valid(recommendation):
                raise ValueError("Recommender returned an invalid recommendation")
            accepted = np.random.binomial(1, self.P[np.arange(n_users), recommendation])
            recommender.update(accepted)
            reward += int(accepted.sum())
        return reward, time.perf_counter() - start


if __name__ == "__main__":
    from recommender import Recommender

    problem = generate_problem(n_users=7, n_items=10, budget=40, n_weeks=500, seed=1)
    sim = Simulation(problem["P"], problem["item_prices"], problem["budget"], problem["n_weeks"])
    reward, elapsed = sim.run(Recommender, seed=0)
    oracle = oracle_reward_per_week(problem["P"], problem["item_prices"], problem["budget"]) * problem["n_weeks"]
    print(f"reward={reward} oracle={oracle:.1f} ratio={reward / oracle:.3f} time={elapsed:.1f}s")
