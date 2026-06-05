import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from clip import clip
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score ImageFolder crops with a trained MaPLe checkpoint."
    )
    parser.add_argument("--crop_root", default="GTSDB_Crops", help="Crop dataset root")
    parser.add_argument("--split", default="test", help="ImageFolder split to score")
    parser.add_argument("--checkpoint", required=True, help="NOVA-MIX MaPLe checkpoint")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--backbone", default="ViT-B/16", help="CLIP backbone")
    parser.add_argument("--n_ctx", type=int, default=2)
    parser.add_argument("--ctx_init", default="")
    parser.add_argument("--prompt_depth", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--keep_checkpoint_tokens",
        action="store_true",
        help="Keep token_prefix/token_suffix from checkpoint instead of rebuilding class prompts",
    )
    return parser.parse_args()


def clean_classname(name: str) -> str:
    name = re.sub(r"^\d+[_-]+", "", name)
    return name.replace("_", " ")


def find_local_clip_model(url: str) -> Path | None:
    filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
    repo_root = Path(__file__).resolve().parents[1]
    search_dirs = [
        os.environ.get("CLIP_MODEL_DIR"),
        Path.cwd() / "models",
        repo_root / "models",
    ]

    for root in search_dirs:
        if not root:
            continue
        candidate = Path(root) / filename
        if candidate.exists():
            return candidate

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob(filename))
        if matches:
            return matches[0]
    return None


def load_clip_to_cpu(backbone_name: str):
    url = clip._MODELS[backbone_name]
    model_path = find_local_clip_model(url)
    if model_path is None:
        try:
            model_path = Path(clip._download(url, root=str(Path(__file__).resolve().parents[1] / "models")))
        except Exception as exc:
            filename = url.rsplit("/", 1)[-1].split("?", 1)[0]
            raise RuntimeError(
                f"CLIP backbone {backbone_name!r} was not found locally as {filename}. "
                "Add it to FDAL-main/models, set CLIP_MODEL_DIR, or attach a Kaggle "
                "dataset containing the file."
            ) from exc
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict())


class MaPLeTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        n_ctx = (
            compound_prompts_deeper_text[0].shape[0]
            if len(compound_prompts_deeper_text) > 0
            else 0
        )

        for i, resblock in enumerate(self.transformer.resblocks):
            if i > 0 and i <= len(compound_prompts_deeper_text):
                prefix = x[:1, :, :]
                suffix = x[1 + n_ctx :, :, :]
                textual_ctx = compound_prompts_deeper_text[i - 1]
                textual_ctx = textual_ctx.expand(x.shape[1], -1, -1)
                textual_ctx = textual_ctx.permute(1, 0, 2).to(x.dtype)
                x = torch.cat([prefix, textual_ctx, suffix], dim=0)
            x = resblock(x)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
        return x @ self.text_projection


class MultiModalPromptLearner(nn.Module):
    def __init__(self, classnames: List[str], clip_model, n_ctx: int, ctx_init: str, prompt_depth: int):
        super().__init__()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init and n_ctx <= 4:
            prompt = clip.tokenize(ctx_init.replace("_", " "))
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.proj = nn.Linear(ctx_dim, 768)
        self.ctx = nn.Parameter(ctx_vectors)
        self.compound_prompts_depth = prompt_depth
        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, 512)) for _ in range(prompt_depth - 1)]
        )
        for prompt_param in self.compound_prompts_text:
            nn.init.normal_(prompt_param, std=0.02)

        self.compound_prompt_projections = nn.ModuleList(
            [nn.Linear(ctx_dim, 768) for _ in range(prompt_depth - 1)]
        )

        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        visual_deep_prompts = [
            layer(self.compound_prompts_text[idx])
            for idx, layer in enumerate(self.compound_prompt_projections)
        ]
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


class VisionEncoder(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        self.conv1 = clip_visual.conv1
        self.class_embedding = clip_visual.class_embedding
        self.positional_embedding = clip_visual.positional_embedding
        self.ln_pre = clip_visual.ln_pre
        self.transformer = clip_visual.transformer
        self.ln_post = clip_visual.ln_post
        self.proj = clip_visual.proj

    def forward(self, x, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = self.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([x, visual_ctx], dim=1)
        n_ctx = shared_ctx.shape[0]

        x = self.ln_pre(x).permute(1, 0, 2)
        for i, resblock in enumerate(self.transformer.resblocks):
            if i > 0 and i <= len(compound_deeper_prompts):
                prefix = x[: x.shape[0] - n_ctx, :, :]
                visual_ctx_i = compound_deeper_prompts[i - 1]
                visual_ctx_i = visual_ctx_i.expand(x.shape[1], -1, -1)
                visual_ctx_i = visual_ctx_i.permute(1, 0, 2).to(x.dtype)
                x = torch.cat([prefix, visual_ctx_i], dim=0)
            x = resblock(x)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj if self.proj is not None else x


class MaPLeCLIP(nn.Module):
    def __init__(self, classnames: List[str], clip_model, n_ctx: int, ctx_init: str, prompt_depth: int):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(
            classnames, clip_model, n_ctx, ctx_init, prompt_depth
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = VisionEncoder(clip_model.visual)
        self.text_encoder = MaPLeTextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * image_features @ text_features.t()


def load_maple_model(args, classnames: List[str], device: torch.device):
    clip_model = load_clip_to_cpu(args.backbone)
    clip_model.float()
    model = MaPLeCLIP(classnames, clip_model, args.n_ctx, args.ctx_init, args.prompt_depth)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state"]["prompt_learner_state"]
    if not args.keep_checkpoint_tokens:
        state = dict(state)
        state.pop("token_prefix", None)
        state.pop("token_suffix", None)
    model.prompt_learner.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )
    split_dir = Path(args.crop_root) / args.split
    dataset = ImageFolder(split_dir, transform=transform)
    classnames = [clean_classname(name) for name in dataset.classes]

    model = load_maple_model(args, classnames, device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows = []
    correct = 0
    cursor = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = logits.softmax(dim=-1).cpu()
            labels = labels.cpu()
            preds = probs.argmax(dim=-1)
            entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)

            for local_idx in range(labels.shape[0]):
                sample_path, _ = dataset.samples[cursor + local_idx]
                true_id = int(labels[local_idx])
                pred_id = int(preds[local_idx])
                correct += int(true_id == pred_id)
                row = {
                    "path": sample_path,
                    "true_id": true_id,
                    "true_name": classnames[true_id],
                    "pred_id": pred_id,
                    "pred_name": classnames[pred_id],
                    "confidence": float(probs[local_idx, pred_id]),
                    "entropy": float(entropy[local_idx]),
                }
                for class_id, class_name in enumerate(classnames):
                    key = f"p_{class_id}_{class_name.replace(' ', '_')}"
                    row[key] = float(probs[local_idx, class_id])
                rows.append(row)
            cursor += labels.shape[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    accuracy = 100.0 * correct / max(1, len(dataset))
    print(f"Scored {len(dataset)} crops from {split_dir}")
    print(f"Classes: {classnames}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Output: {output}")
    print(f"Max entropy for {len(classnames)} classes: {math.log(len(classnames)):.4f}")


if __name__ == "__main__":
    main()
