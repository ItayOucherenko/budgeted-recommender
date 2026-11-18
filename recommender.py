import numpy as np

class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.prices = np.array(prices)
        self.budget = budget
        self.n_items = len(self.prices)
        self.feasible = [i for i in range(self.n_items) if self.prices[i] <= self.budget]
        self.successes = np.zeros((n_users, self.n_items), dtype=float)
        self.attempts = np.zeros((n_users, self.n_items), dtype=float)
        self.last_recommend = None
        self.initial_index = 0
        self.strategy_weights = np.ones(5)  # A, B, C, D, E
        self.t = 0

    def recommend(self) -> np.array:
        if len(self.feasible) == 0:
            rec = np.zeros(self.n_users, dtype=int)
            self.last_recommend = rec
            return rec

        if self.initial_index < len(self.feasible):
            i = self.feasible[self.initial_index]
            rec = np.full(self.n_users, i, dtype=int)
            self.last_recommend = rec
            self.initial_index += 1
            return rec

        denom = self.attempts + 1.0
        prob_est = (self.successes + 1.0) / denom

        best_score, best_rec, best_strategy = -1, None, -1

        for s_id in range(5):  # A to E
            for _ in range(25):
                p_sample = np.random.beta(self.successes + 1.0, (self.attempts - self.successes) + 1.0)

                if s_id == 0:  # Strategy A: improvement-based greedy
                    best_current = np.zeros(self.n_users)
                    S, remaining_budget = [], self.budget
                    items = set(self.feasible)
                    while True:
                        best_item, best_score_i = None, 0.0
                        for i in items:
                            if self.prices[i] > remaining_budget:
                                continue
                            diff = p_sample[:, i] - best_current
                            diff = np.where(diff < 0, 0, diff)
                            improvement = diff.sum()
                            score = improvement / self.prices[i]
                            if score > best_score_i:
                                best_score_i = score
                                best_item = i
                        if best_item is None:
                            break
                        S.append(best_item)
                        best_current = np.maximum(best_current, p_sample[:, best_item])
                        remaining_budget -= self.prices[best_item]
                        items.remove(best_item)
                    if not S:
                        S = [min(self.feasible, key=lambda x: self.prices[x])]

                elif s_id == 1:  # Strategy B: sum-based greedy
                    S, remaining_budget = [], self.budget
                    items = set(self.feasible)
                    while True:
                        best_item, best_score_i = None, 0.0
                        for i in items:
                            if self.prices[i] > remaining_budget:
                                continue
                            score = p_sample[:, i].sum() / self.prices[i]
                            if score > best_score_i:
                                best_score_i = score
                                best_item = i
                        if best_item is None:
                            break
                        S.append(best_item)
                        remaining_budget -= self.prices[best_item]
                        items.remove(best_item)
                    if not S:
                        S = [min(self.feasible, key=lambda x: self.prices[x])]

                elif s_id == 2:  # Strategy C: normalized mean priority
                    mean_p = p_sample.mean(axis=0)
                    priority = (mean_p - mean_p.mean()) / self.prices
                    sorted_items = np.argsort(-priority)
                    S, total = [], 0
                    for i in sorted_items:
                        if i not in self.feasible or total + self.prices[i] > self.budget:
                            continue
                        total += self.prices[i]
                        S.append(i)
                    if not S:
                        S = [min(self.feasible, key=lambda x: self.prices[x])]

                elif s_id == 3:  # Strategy D: UCB-style selection
                    ucb = prob_est + np.sqrt(2 * np.log(self.t + 2) / (self.attempts + 1))
                    score_ucb = ucb.mean(axis=0) / self.prices
                    sorted_items = np.argsort(-score_ucb)
                    S, total = [], 0
                    for i in sorted_items:
                        if i not in self.feasible or total + self.prices[i] > self.budget:
                            continue
                        total += self.prices[i]
                        S.append(i)
                    if not S:
                        S = [min(self.feasible, key=lambda x: self.prices[x])]

                else:  # Strategy E: per-user budgeted greedy
                    user_budget = self.budget // self.n_users
                    S = []
                    for u in range(self.n_users):
                        est_p = prob_est[u]
                        scores = est_p / self.prices
                        sorted_items = np.argsort(-scores)
                        for i in sorted_items:
                            if self.prices[i] <= user_budget:
                                S.append(i)
                                break
                        else:
                            cheap = [i for i in self.feasible if self.prices[i] <= user_budget]
                            S.append(min(cheap, key=lambda x: self.prices[x]) if cheap else min(self.feasible, key=lambda x: self.prices[x]))

                if len(S) == 1:
                    rec = np.full(self.n_users, S[0], dtype=int)
                else:
                    sub_probs = prob_est[:, S]
                    rec = np.array([S[idx] for idx in np.argmax(sub_probs, axis=1)])

                sub_probs = prob_est[np.arange(self.n_users), rec]
                score = sub_probs.sum()

                weighted_score = score * self.strategy_weights[s_id]
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_rec = rec
                    best_strategy = s_id

        self.strategy_weights = 0.9 * self.strategy_weights  # decay previous weights
        self.strategy_weights[best_strategy] += 0.1 * (best_score / self.n_users)  # reward-based update
        self.last_recommend = best_rec
        return best_rec

    def update(self, results: np.array):
        self.t += 1
        rec = self.last_recommend
        for n in range(self.n_users):
            i = rec[n]
            self.attempts[n, i] += 1
            self.successes[n, i] += results[n]