import numpy as np
from simulator import generate_problem, Simulation, oracle_reward_per_week
from recommender import Recommender

CONFIGS = {
    "small (7 users, 10 items)": dict(n_users=7, n_items=10, budget=40, n_weeks=500),
    "medium (20 users, 15 items)": dict(n_users=20, n_items=15, budget=60, n_weeks=500),
}
N_SEEDS = 30

for name, cfg in CONFIGS.items():
    ratios, rewards, times = [], [], []
    for seed in range(N_SEEDS):
        p = generate_problem(seed=seed, **cfg)
        sim = Simulation(p["P"], p["item_prices"], p["budget"], p["n_weeks"], time_limit=120.0)
        reward, elapsed = sim.run(Recommender, seed=seed)
        oracle = oracle_reward_per_week(p["P"], p["item_prices"], p["budget"]) * p["n_weeks"]
        ratios.append(reward / oracle)
        rewards.append(reward)
        times.append(elapsed)
    r = np.array(ratios)
    print(f"{name}: ratio mean {r.mean():.3f} std {r.std(ddof=1):.3f} min {r.min():.3f} max {r.max():.3f} "
          f"| reward mean {np.mean(rewards):.1f} | max time {max(times):.1f}s", flush=True)
