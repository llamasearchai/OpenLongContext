"""
Grid Search for Ablation Studies
Author: Nik Jois <nikjois@llamasearch.ai>
"""

import itertools
from typing import Any, Callable, Dict, List


class GridSearch:
    def __init__(self, objective_fn: Callable[[Dict[str, Any]], float], param_grid: Dict[str, List[Any]]):
        self.objective_fn = objective_fn
        self.param_grid = param_grid
        self.history = []

    def run(self):
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        best_params = None
        best_score = float('inf')
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
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
# param_grid = {"lr": [1e-5, 1e-4, 1e-3], "dropout": [0.0, 0.1, 0.2]}
# search = GridSearch(my_objective, param_grid)
# best_params, best_score = search.run()
