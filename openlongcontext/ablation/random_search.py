"""
Random Search for Ablation Studies
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import random
from typing import Any, Callable, Dict, List


class RandomSearch:
    def __init__(self, objective_fn: Callable[[Dict[str, Any]], float], param_space: Dict[str, List[Any]], n_trials: int = 50):
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.history = []

    def sample(self) -> Dict[str, Any]:
        return {k: random.choice(v) for k, v in self.param_space.items()}

    def run(self):
        best_params = None
        best_score = float('inf')
        for _ in range(self.n_trials):
            params = self.sample()
            score = self.objective_fn(params)
            self.history.append((params, score))
            if score < best_score:
                best_score = score
                best_params = params
        return best_params, best_score

# Example usage:
# def my_objective(params):
#     ... # Compute loss/metric
#     return loss
# param_space = {"lr": [1e-5, 1e-4, 1e-3], "dropout": [0.0, 0.1, 0.2]}
# search = RandomSearch(my_objective, param_space)
# best_params, best_score = search.run()
