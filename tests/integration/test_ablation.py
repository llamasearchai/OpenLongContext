from openlongcontext.ablation import (
    BayesianOptimizer,
    ExperimentRegistry,
    GridSearch,
    HyperparameterSweep,
    RandomSearch,
)


def test_ablation_modules_importable():
    assert callable(BayesianOptimizer)
    assert callable(ExperimentRegistry)
    assert callable(GridSearch)
    assert callable(HyperparameterSweep)
    assert callable(RandomSearch)
