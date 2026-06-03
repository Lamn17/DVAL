import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an ImageFolder crop dataset from YOLO detection labels."
    )
    parser.add_argument("--data_yaml", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--output", required=True, help="Output crop dataset directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="YOLO splits to crop, usually train val",
    )
    parser.add_argument(
        "--val_output_name",
        default="test",
        help="Output split name for YOLO val split, because NOVA-MIX expects test",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Square crop padding ratio relative to the larger bbox side",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=2,
        help="Skip boxes smaller than this many pixels before padding",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory before creating crops",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "class"


def load_data_yaml(path: Path) -> Tuple[Path, Dict[int, str], Dict]:
    with path.open("r") as f:
        data = yaml.safe_load(f)

    root_value = data.get("path") or data.get("root")
    if root_value is None:
        root = path.parent
    else:
        root = Path(root_value)
        if not root.is_absolute():
            root = (path.parent / root).resolve()

    names = data.get("names", {})
    if isinstance(names, list):
        class_names = {idx: str(name) for idx, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = {int(idx): str(name) for idx, name in names.items()}
    else:
        raise ValueError("data.yaml must contain names as a list or dict")

    return root, class_names, data


def split_output_name(split: str, val_output_name: str) -> str:
    return val_output_name if split == "val" else split


def resolve_split_dir(root: Path, data: Dict, split: str) -> Path:
    split_value = data.get(split)
    if split_value is None:
        return root / "images" / split
    if isinstance(split_value, list):
        if len(split_value) != 1:
            raise ValueError(f"Split {split!r} has multiple paths; this script expects one")
        split_value = split_value[0]
    split_path = Path(str(split_value))
    if split_path.is_absolute():
        return split_path
    return root / split_path


def image_to_label_path(image_path: Path, root: Path) -> Path:
    rel = image_path.relative_to(root / "images")
    return root / "labels" / rel.with_suffix(".txt")


def yolo_box_to_square_crop(
    xc: float,
    yc: float,
    bw: float,
    bh: float,
    img_w: int,
    img_h: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = max(box_w, box_h) * (1.0 + padding)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = int(round(cx + side / 2.0))
    crop_y2 = int(round(cy + side / 2.0))

    crop_x1 = max(0, min(crop_x1, img_w - 1))
    crop_y1 = max(0, min(crop_y1, img_h - 1))
    crop_x2 = max(crop_x1 + 1, min(crop_x2, img_w))
    crop_y2 = max(crop_y1 + 1, min(crop_y2, img_h))
    return crop_x1, crop_y1, crop_x2, crop_y2


def parse_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(value) for value in parts[1:])
        boxes.append((cls, xc, yc, bw, bh))
    return boxes


def create_crops(args) -> None:
    data_yaml = Path(args.data_yaml)
    root, class_names, data = load_data_yaml(data_yaml)
    output = Path(args.output)

    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    class_dirs = {
        class_id: f"{class_id:02d}_{sanitize_name(name)}"
        for class_id, name in class_names.items()
    }

    manifest_path = output / "manifest.csv"
    summary = {}
    manifest_rows = []

    for split in args.splits:
        split_dir = resolve_split_dir(root, data, split)
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        out_split = split_output_name(split, args.val_output_name)
        for dirname in class_dirs.values():
            (output / out_split / dirname).mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTS
        )
        split_counts = {class_id: 0 for class_id in class_names}
        skipped = 0

        for image_path in image_paths:
            label_path = image_to_label_path(image_path, root)
            boxes = parse_label_file(label_path)
            if not boxes:
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                for obj_idx, (class_id, xc, yc, bw, bh) in enumerate(boxes):
                    if class_id not in class_names:
                        raise ValueError(f"{label_path}: class {class_id} not in data.yaml names")

                    raw_w = bw * img_w
                    raw_h = bh * img_h
                    if raw_w < args.min_size or raw_h < args.min_size:
                        skipped += 1
                        continue

                    crop_box = yolo_box_to_square_crop(
                        xc, yc, bw, bh, img_w, img_h, args.padding
                    )
                    crop = img.crop(crop_box)
                    class_dir = output / out_split / class_dirs[class_id]
                    crop_name = f"{image_path.stem}_obj{obj_idx:02d}_cls{class_id}.jpg"
                    crop_path = class_dir / crop_name
                    crop.save(crop_path, quality=95)

                    split_counts[class_id] += 1
                    manifest_rows.append(
                        {
                            "split": out_split,
                            "source_split": split,
                            "source_image": str(image_path),
                            "source_label": str(label_path),
                            "crop_path": str(crop_path),
                            "class_id": class_id,
                            "class_name": class_names[class_id],
                            "object_index": obj_idx,
                            "yolo_x_center": xc,
                            "yolo_y_center": yc,
                            "yolo_width": bw,
                            "yolo_height": bh,
                            "crop_x1": crop_box[0],
                            "crop_y1": crop_box[1],
                            "crop_x2": crop_box[2],
                            "crop_y2": crop_box[3],
                        }
                    )

        summary[out_split] = {"counts": split_counts, "skipped": skipped}

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Created crop dataset: {output}")
    print(f"Manifest: {manifest_path}")
    for split, info in summary.items():
        total = sum(info["counts"].values())
        print(f"{split}: {total} crops, skipped={info['skipped']}")
        for class_id, count in info["counts"].items():
            print(f"  {class_id:02d} {class_names[class_id]}: {count}")


def main():
    create_crops(parse_args())


if __name__ == "__main__":
    main()
