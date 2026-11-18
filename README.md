# Budgeted Weekly Recommender (Thompson sampling with strategy selection)

A recommender for a weekly setting with a budget. Each week the system must choose a set of items whose total price fits within the budget, and then recommend one item from that set to every user. Each user accepts the recommendation with an unknown probability, and the goal is to maximize the number of accepted recommendations over all weeks.

The repository contains the recommender, a simulator for this setting, and an experiment script.

## Approach

- **Estimating acceptance probabilities:** for every (user, item) pair the recommender keeps a Beta posterior over the acceptance probability, updated from the observed accept/reject results.
- **Exploration at the start:** the first weeks recommend each feasible item to all users once, so every item has some data.
- **Five candidate strategies** for choosing the weekly item set. Each one is run 25 times per week with posterior samples:
  - **A.** Greedy by marginal improvement per unit price.
  - **B.** Greedy by sum of sampled acceptance probabilities per unit price.
  - **C.** Items ranked by mean sampled probability relative to the average, normalized by price.
  - **D.** Items ranked by a UCB-style optimistic estimate, normalized by price.
  - **E.** Per-user budgeted greedy: each user gets an item within an equal share of the budget.
- **Choosing a strategy each week:** every candidate is scored by the expected accepted recommendations under the current estimates, multiplied by a strategy weight. The weights decay each week (factor 0.9) and the winning strategy's weight is increased.
- **Final recommendation:** within the chosen item set, each user receives the item with the highest estimated acceptance probability.

## Repository contents

| File | Description |
| --- | --- |
| `recommender.py` | The `Recommender` class (`recommend()` and `update(results)`). |
| `simulator.py` | Simulator with random problem generation, budget validation, a time limit, and an enumeration-based oracle (best fixed item set per week, for reference). |
| `run_experiments.py` | Runs 30 seeds on two synthetic configurations and reports the ratio of reward to the oracle. |

## Results

The experiments use synthetic data only: acceptance probabilities are drawn from Beta(1, 4) and prices from integers 5–30. Each configuration is run with 30 seeds and 500 weeks. The ratio is the total accepted recommendations divided by the oracle total (the best fixed item set per week, evaluated on the same probabilities).

| Configuration | Users | Items | Budget | Mean ratio | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Small | 7 | 10 | 40 | 0.914 | 0.032 | 0.810 | 0.968 |
| Medium | 20 | 15 | 60 | 0.882 | 0.025 | 0.837 | 0.951 |

The oracle knows the true probabilities and is not achievable by a learner, so these ratios measure how close the learner gets to the best fixed choice, not a benchmark against other methods.

## Reproducing

Requirements: Python 3 with `numpy`.

```bash
python3 simulator.py        # single smoke test
python3 run_experiments.py  # 30-seed experiments (about 4-5 minutes, mostly the oracle enumeration)
```

Results are reproducible because every run sets its seed explicitly.
