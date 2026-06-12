import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from src.strategies.base import BaseStrategy


class CAUSStrategy(BaseStrategy):
    """Class-Adaptive Uncertainty Sampling for object detection."""

    def __init__(
        self,
        model,
        class_temperature: float = 1.0,
        top_m: int = 5,
        experiment_dir: Optional[str] = None,
        round: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        if class_temperature < 0:
            raise ValueError("class_temperature must be non-negative")
        if top_m <= 0:
            raise ValueError("top_m must be positive")

        self.class_temperature = float(class_temperature)
        self.top_m = int(top_m)
        self.experiment_dir = experiment_dir
        self.round = round

    def query(
        self,
        unlabeled_indices: np.ndarray,
        image_paths: List[str],
        n_samples: int,
        **kwargs,
    ) -> np.ndarray:
        self._validate_inputs(unlabeled_indices, image_paths, n_samples)

        local_kwargs = dict(kwargs)
        num_inf = int(local_kwargs.pop("num_inference", -1))
        candidate_indices = unlabeled_indices[:num_inf] if num_inf > 0 else unlabeled_indices
        candidate_paths = self._get_image_paths_for_indices(candidate_indices, image_paths)

        print(f"Computing CAUS uncertainty scores for {len(candidate_paths)} images...")
        start_time = time.time()
        results = self.model.inference(
            candidate_paths,
            return_boxes=True,
            return_probs=True,
            return_classes=True,
            num_inference=-1,
            inference_device=self.strategy_params.get(
                "inference_device", self.strategy_params.get("device", "auto")
            ),
            inference_batch_size=self.strategy_params.get("inference_batch_size", 1),
            **local_kwargs,
        )

        classwise_quality = self._estimate_classwise_quality(results)
        logger.info("CAUS estimated classwise quality from the current unlabeled pool")
        self._set_classwise_quality(classwise_quality)

        processed_indices = candidate_indices[: len(results)]
        scores = np.array([self._compute_caus_score(result) for result in results], dtype=float)
        n_selected = min(n_samples, len(processed_indices))
        top_local = np.argsort(scores, kind="stable")[-n_selected:]
        selected_indices = processed_indices[top_local]

        elapsed = time.time() - start_time
        self._write_time_log(elapsed, len(processed_indices))

        if scores.size:
            selected_scores = np.sort(scores[top_local])
            print(f"Top 5 CAUS uncertainty scores: {selected_scores[-5:]}")
            print(
                f"CAUS score statistics - Mean: {scores.mean():.4f}, "
                f"Std: {scores.std():.4f}, Min: {scores.min():.4f}, "
                f"Max: {scores.max():.4f}"
            )

        return selected_indices

    def _set_classwise_quality(self, quality: np.ndarray) -> None:
        model_to_set = self._model_to_read()
        setattr(model_to_set, "classwise_quality", torch.as_tensor(quality, dtype=torch.float32))

    def _estimate_classwise_quality(self, results) -> np.ndarray:
        num_classes = self._num_classes(results)
        quality_sum = np.zeros(num_classes, dtype=float)
        quality_count = np.zeros(num_classes, dtype=float)

        for result in results:
            if result.probs is None or result.classes is None:
                continue
            classes = np.asarray(result.classes, dtype=int).reshape(-1)
            probs = np.asarray(result.probs, dtype=float)
            if probs.ndim == 1:
                confidences = probs
            elif probs.ndim == 2 and len(probs) == len(classes):
                confidences = probs.max(axis=1)
            else:
                continue
            if len(confidences) != len(classes):
                continue

            valid = (classes >= 0) & (classes < num_classes)
            np.add.at(quality_sum, classes[valid], confidences[valid])
            np.add.at(quality_count, classes[valid], 1.0)

        observed = quality_count > 0
        fallback = (
            float(quality_sum[observed].sum() / quality_count[observed].sum())
            if np.any(observed)
            else 0.5
        )
        quality = np.full(num_classes, fallback, dtype=np.float32)
        quality[observed] = (quality_sum[observed] / quality_count[observed]).astype(np.float32)
        return np.clip(quality, 0.0, 1.0)

    def _get_classwise_weights(self) -> Dict[int, float]:
        quality = getattr(self._model_to_read(), "classwise_quality", None)
        if quality is None:
            raise ValueError("Model does not have classwise_quality")

        if isinstance(quality, torch.Tensor):
            quality = quality.detach().cpu().numpy()
        quality = np.clip(np.asarray(quality, dtype=float), 0.0, 1.0)

        raw_weights = np.exp(self.class_temperature * (1.0 - quality))
        normalized = raw_weights / max(float(raw_weights.mean()), 1e-12)
        return {class_id: float(weight) for class_id, weight in enumerate(normalized)}

    def _compute_caus_score(self, result) -> float:
        if result.probs is None or len(result.probs) == 0:
            return 0.0

        uncertainties = self._detection_uncertainties(np.asarray(result.probs, dtype=float))
        if uncertainties.size == 0:
            return 0.0

        class_weights = self._get_classwise_weights()
        classes = (
            np.asarray(result.classes, dtype=int).reshape(-1)
            if result.classes is not None
            else np.zeros(len(uncertainties), dtype=int)
        )
        if len(classes) != len(uncertainties):
            raise ValueError(
                f"CAUS received {len(classes)} classes for {len(uncertainties)} detections"
            )

        weights = np.array([class_weights.get(int(class_id), 1.0) for class_id in classes])
        utilities = uncertainties * weights
        top_count = min(self.top_m, len(utilities))
        top_utilities = np.partition(utilities, len(utilities) - top_count)[-top_count:]
        return float(top_utilities.mean())

    @staticmethod
    def _detection_uncertainties(probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        if probs.ndim == 1:
            binary_entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))
            return binary_entropy / math.log(2.0)
        if probs.ndim == 2:
            categorical_entropy = -np.sum(probs * np.log(probs), axis=1)
            return categorical_entropy / max(math.log(probs.shape[1]), 1e-8)
        raise ValueError(f"Unsupported probability shape for CAUS: {probs.shape}")

    def _model_to_read(self):
        model = self.model.model
        return model.module if hasattr(model, "module") else model

    def _num_classes(self, results) -> int:
        model = self._model_to_read()
        num_classes = getattr(model, "nc", None)
        if num_classes is not None:
            return max(1, int(num_classes))

        names = getattr(model, "names", None)
        if isinstance(names, (list, tuple, dict)):
            return max(1, len(names))

        max_class = -1
        for result in results:
            if result.classes is not None and len(result.classes):
                max_class = max(max_class, int(np.max(result.classes)))
        return max(1, max_class + 1)

    def _write_time_log(self, elapsed: float, num_images: int) -> None:
        if not self.experiment_dir:
            return
        log_name = os.environ.get("TIME_LOGFILE", "time_log.txt")
        log_path = Path(self.experiment_dir) / log_name
        if not log_path.exists():
            log_path.write_text("Round,TotalTime,NumImages,TimePerImage\n")
        time_per_image = elapsed / num_images if num_images else 0.0
        with log_path.open("a") as log_file:
            log_file.write(
                f"{self.round},{elapsed:.4f},{num_images},{time_per_image:.4f}\n"
            )

    def get_strategy_name(self) -> str:
        return "caus"
