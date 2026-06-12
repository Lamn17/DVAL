from types import SimpleNamespace

import numpy as np
import torch

from src.strategies.uncertainty.caus import CAUSStrategy


class DummyModel:
    def __init__(self, quality):
        self.model = SimpleNamespace(
            classwise_quality=torch.tensor(quality, dtype=torch.float32),
            nc=len(quality),
        )


def test_binary_entropy_is_maximal_at_half_confidence():
    uncertainties = CAUSStrategy._detection_uncertainties(
        np.array([0.1, 0.5, 0.9], dtype=float)
    )

    assert np.isclose(uncertainties[0], uncertainties[2])
    assert np.isclose(uncertainties[1], 1.0)
    assert uncertainties[1] > uncertainties[0]


def test_harder_class_receives_larger_normalized_weight():
    strategy = CAUSStrategy(DummyModel([0.2, 0.8]), class_temperature=1.0)

    weights = strategy._get_classwise_weights()

    assert weights[0] > weights[1]
    assert np.isclose(np.mean(list(weights.values())), 1.0)


def test_top_m_prevents_many_easy_boxes_from_dominating():
    strategy = CAUSStrategy(DummyModel([0.5]), class_temperature=0.0, top_m=2)
    result = SimpleNamespace(
        probs=np.array([0.5, 0.5, 0.99, 0.99], dtype=float),
        classes=np.zeros(4, dtype=int),
    )

    score = strategy._compute_caus_score(result)

    assert np.isclose(score, 1.0)


def test_pool_quality_estimation_uses_mean_confidence_per_class():
    strategy = CAUSStrategy(DummyModel([0.0, 0.0]))
    results = [
        SimpleNamespace(
            probs=np.array([0.2, 0.6, 0.8], dtype=float),
            classes=np.array([0, 0, 1], dtype=int),
        )
    ]

    quality = strategy._estimate_classwise_quality(results)

    assert np.allclose(quality, [0.4, 0.8])
