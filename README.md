# DVAL Active Learning for Traffic-Sign Detection

This repository contains an active-learning pipeline for object detection with YOLO and MaPLe-based visual re-ranking. The current public setup is focused on **DVAL** experiments for two traffic-sign datasets:

- `GTSDB`
- `VTSDB100`

DVAL is configured as a two-stage selection pipeline in this codebase:

1. `ddus` builds a detector-side uncertainty shortlist from unlabeled images.
2. `maple` re-ranks that shortlist with crop-level MaPLe scores.

## Requirements

- Python 3.9+
- CUDA-capable GPU is recommended
- `uv` package manager
- YOLO weights such as `yolo11n.pt`
- Prepared dataset folders for `GTSDB` and/or `VTSDB100`

Install dependencies:

```bash
uv sync
```

If your environment uses process titles, create the training environment file:

```bash
cp .env.example .env.training
```

At minimum, make sure `PROCTITLE_STARTSTR` is set before running experiments:

```bash
export PROCTITLE_STARTSTR=fdal
```

## Datasets

The supported config folders are intentionally limited to:

```text
configs/
+-- gtsdb/
+-- vtsdb100/
```

Expected dataset YAML files:

- `GTSDB/data_local.yaml`
- `VTSDB100/data.yaml`

Each dataset YAML should follow the YOLO format with train/val/test image paths and class names.

## Run DVAL

Run DVAL on VTSDB100:

```bash
uv run python scripts/run_experiment.py --config configs/vtsdb100/config_dval.yaml
```

Run DVAL on GTSDB:

```bash
uv run python scripts/run_experiment.py --config configs/gtsdb/config_dval.yaml
```

Useful overrides:

```bash
uv run python scripts/run_experiment.py \
  --config configs/vtsdb100/config_dval.yaml \
  --device 0 \
  --seed 1
```

## Available Configs

VTSDB100:

- `configs/vtsdb100/config_dval.yaml`
- `configs/vtsdb100/config_random.yaml`
- `configs/vtsdb100/config_entropy.yaml`
- `configs/vtsdb100/config_coreset.yaml`

GTSDB:

- `configs/gtsdb/config_dval.yaml`
- `configs/gtsdb/config_random.yaml`
- `configs/gtsdb/config_entropy.yaml`
- `configs/gtsdb/config_coreset.yaml`

## DVAL Configuration

The main DVAL config uses chained strategies:

```yaml
strategy: ["ddus", "maple"]
expand_ratios: [2.0]
```

This means each round selects a DDUS shortlist of `samples_per_round * 2`, then MaPLe re-ranks the shortlist down to the final `samples_per_round` images.

Important fields:

- `initial_labeled_count`: initial labeled training images.
- `samples_per_round`: final images selected per active-learning round.
- `max_rounds`: number of active-learning rounds.
- `strategy_args.ddus`: detector uncertainty and class/localization quality weights.
- `strategy_args.maple`: crop-level MaPLe scoring weights.
- `maple_training`: prompt-learning settings for the MaPLe checkpoint used each round.

## Project Layout

```text
.
+-- configs/
|   +-- gtsdb/
|   +-- vtsdb100/
+-- scripts/
|   +-- run_experiment.py
|   +-- train.py
|   +-- strategy.py
|   +-- setup_data.py
|   +-- simulate_labeling.py
|   +-- create_yolo_crops.py
|   +-- train_maple_round.py
+-- src/
|   +-- data/
|   +-- models/
|   +-- strategies/
+-- tests/
```

## Notes Before Pushing to GitHub

Avoid committing generated or heavy files such as:

- datasets: `GTSDB/`, `VTSDB100/`
- experiment outputs: `full_experiments/`, `runs/`, `wandb/`
- model weights: `*.pt`, `*.pth`
- local environments: `.venv/`, `.env`, `.env.training`

Keep only source code, configs, tests, and documentation in the repository.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
