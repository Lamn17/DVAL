import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.score_maple_crops import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    MaPLeCLIP,
    clean_classname,
    load_clip_to_cpu,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MaPLe prompt learner for one AL round.")
    parser.add_argument("--crop_root", required=True, help="ImageFolder crop dataset root")
    parser.add_argument("--round", type=int, required=True, help="AL round number")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--backbone", default="ViT-B/16")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--ctx_init", default="")
    parser.add_argument("--prompt_depth", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Validate every N epochs; 0 validates only the final epoch.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_loader(split_dir: Path, transform, batch_size: int, num_workers: int, shuffle: bool):
    if not split_dir.exists():
        return None
    dataset = ImageFolder(split_dir, transform=transform)
    if len(dataset) == 0:
        return None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, loader


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            total_loss += float(loss.item()) * int(labels.numel())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": 100.0 * correct / max(1, total),
        "num_samples": total,
    }


def write_metrics(metrics_path: Path, rows):
    if not rows:
        return
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def should_validate_epoch(epoch: int, epochs: int, val_interval: int) -> bool:
    epoch_num = epoch + 1
    if epoch_num == epochs:
        return True
    if val_interval <= 0:
        return False
    return epoch_num % val_interval == 0


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    crop_root = Path(args.crop_root)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    train_data = build_loader(
        crop_root / "train", transform, args.batch_size, args.num_workers, shuffle=True
    )
    if train_data is None:
        raise RuntimeError(f"No MaPLe training crops found at {crop_root / 'train'}")
    train_dataset, train_loader = train_data

    val_data = build_loader(
        crop_root / "test", transform, args.batch_size, args.num_workers, shuffle=False
    )
    if val_data is not None and val_data[0].classes != train_dataset.classes:
        print("Validation crop classes differ from train classes; disabling validation.")
        val_data = None

    classnames = [clean_classname(name) for name in train_dataset.classes]
    clip_model = load_clip_to_cpu(args.backbone)
    clip_model.float()
    model = MaPLeCLIP(
        classnames,
        clip_model,
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init,
        prompt_depth=args.prompt_depth,
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.prompt_learner.parameters():
        param.requires_grad = True

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output.with_suffix(".metrics.csv")
    best_score = -1.0
    rows = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            total_loss += float(loss.item()) * int(labels.numel())

        train_metrics = {
            "loss": total_loss / max(1, total),
            "accuracy": 100.0 * correct / max(1, total),
            "num_samples": total,
        }
        should_validate = (
            val_data is not None
            and should_validate_epoch(epoch, args.epochs, args.val_interval)
        )
        val_metrics = evaluate(model, val_data[1], device) if should_validate else None
        score = val_metrics["accuracy"] if val_metrics is not None else train_metrics["accuracy"]

        row = {
            "round": args.round,
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"] if val_metrics else "",
            "val_accuracy": val_metrics["accuracy"] if val_metrics else "",
        }
        rows.append(row)
        print(
            f"MaPLe round {args.round:03d} epoch {epoch + 1}/{args.epochs}: "
            f"train_acc={train_metrics['accuracy']:.2f}"
            + (f", val_acc={val_metrics['accuracy']:.2f}" if val_metrics else ""),
            flush=True,
        )
        write_metrics(metrics_path, rows)

        can_select_checkpoint = val_data is None or val_metrics is not None
        if can_select_checkpoint and score >= best_score:
            best_score = score
            checkpoint = {
                "round": args.round,
                "classnames": classnames,
                "model_state": {
                    "prompt_learner_state": model.prompt_learner.state_dict(),
                },
                "optimizer_state": optimizer.state_dict(),
                "metrics": row,
            }
            torch.save(checkpoint, output)

    print(f"Saved MaPLe checkpoint: {output}")
    print(f"Saved MaPLe metrics: {metrics_path}")


if __name__ == "__main__":
    main()
