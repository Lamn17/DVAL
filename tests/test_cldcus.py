from types import SimpleNamespace

import numpy as np
import torch

from src.strategies.uncertainty.cldcus import CLDCUSStrategy
from src.strategies.uncertainty.dcus_patching import update_classwise_quality_ema


class DummyModel:
    def __init__(self):
        self.model = SimpleNamespace()


def make_strategy(**kwargs):
    strategy = CLDCUSStrategy(DummyModel(), **kwargs)
    strategy.classwise_cls_quality = np.array([0.2, 0.8], dtype=float)
    strategy.classwise_loc_quality = np.array([0.4, 0.6], dtype=float)
    return strategy


def test_binary_entropy_is_symmetric_and_maximal_at_half():
    uncertainty = CLDCUSStrategy._binary_entropy(np.array([0.1, 0.5, 0.9]))

    assert np.isclose(uncertainty[0], uncertainty[2])
    assert np.isclose(uncertainty[1], 1.0)


def test_class_difficulty_combines_classification_and_localization():
    strategy = make_strategy(lambda_cls=0.6, lambda_loc=0.4, rho=0.5)

    weights = strategy._get_classwise_weights()

    assert weights[0] > weights[1]
    assert np.isclose(np.mean(list(weights.values())), 1.0)


def test_top_m_mean_ignores_many_easy_detections():
    strategy = make_strategy(rho=0.0, top_m=2)
    result = SimpleNamespace(
        probs=np.array([0.5, 0.5, 0.99, 0.99], dtype=float),
        classes=np.array([0, 1, 0, 1], dtype=int),
    )

    assert np.isclose(strategy._compute_cldcus_score(result), 1.0)


def test_empty_detection_image_scores_zero():
    strategy = make_strategy()
    result = SimpleNamespace(probs=np.array([]), classes=np.array([]))

    assert strategy._compute_cldcus_score(result) == 0.0


def test_masked_ema_preserves_unobserved_classes():
    tracker = SimpleNamespace(
        class_quality=torch.tensor([0.5, 0.5]),
        class_cls_quality=torch.tensor([0.5, 0.5]),
        class_loc_quality=torch.tensor([0.5, 0.5]),
    )

    update_classwise_quality_ema(
        tracker,
        class_quality_sum=torch.tensor([0.8, 0.0]),
        class_cls_sum=torch.tensor([0.6, 0.0]),
        class_loc_sum=torch.tensor([1.0, 0.0]),
        class_quality_count=torch.tensor([1.0, 0.0]),
        base_momentum=0.5,
    )

    assert torch.allclose(tracker.class_quality, torch.tensor([0.65, 0.5]))
    assert torch.allclose(tracker.class_cls_quality, torch.tensor([0.55, 0.5]))
    assert torch.allclose(tracker.class_loc_quality, torch.tensor([0.75, 0.5]))
