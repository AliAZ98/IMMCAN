"""Command line training for the released experimental variants."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

from .data import make_loader
from .encoders import IdiomEncoder, JinaCLIPV2Encoder
from .losses import listnet_loss, smooth_cross_entropy
from .models import IMMCAN, TextOnlyRanker


TASKS = ("multimodal-regression", "multimodal-classification",
         "text-listnet", "text-regression")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an IMMCAN variant")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--train-tsv", nargs="+", required=True)
    parser.add_argument("--train-root", nargs="+", required=True)
    parser.add_argument("--dev-tsv", nargs="+", required=True)
    parser.add_argument("--dev-root", nargs="+", required=True)
    parser.add_argument("--idiom-model", required=True,
                        help="Local directory of the supplied fine-tuned XLM-R checkpoint")
    parser.add_argument("--vlm-model", default="jinaai/jina-clip-v2")
    parser.add_argument("--visual-text", choices=("sentence", "caption"),
                        default="sentence")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-gamma", type=float, default=0.98)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1376)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def make_model(args):
    encoder = IdiomEncoder(args.idiom_model, device=args.device, max_length=128)
    if args.task.startswith("multimodal"):
        vlm = JinaCLIPV2Encoder(args.vlm_model, device=args.device, max_length=128)
        task = args.task.removeprefix("multimodal-")
        model = IMMCAN(
            encoder, vlm, task=task, dropout=args.dropout,
            visual_text=args.visual_text,
        )
    else:
        model = TextOnlyRanker(encoder, dropout=args.dropout)
    return model.to(args.device)


def forward_loss_and_top1(model, batch, task, device):
    ranks = batch["ranks"].to(device)
    B, G = ranks.shape
    if G != 5:
        raise ValueError("Five candidates per idiom required")
    if task.startswith("multimodal"):
        compounds = [item for item in batch["compounds"] for _ in range(G)]
        sentences = [item for item in batch["sentences"] for _ in range(G)]
        captions = [item for group in batch["captions"] for item in group]
        output = model(compounds, sentences, batch["images"], captions=captions)
        if task == "multimodal-classification":
            logits = output
            labels = ranks.reshape(-1).long()  # FIX: 5 distinct ordinal labels
            loss = nn.functional.cross_entropy(logits, labels)
            loss = loss + 2 * smooth_cross_entropy(logits, labels, smoothing=0.2)
            probabilities = torch.softmax(logits, dim=-1)
            ordered_ranks = torch.arange(5, device=device, dtype=probabilities.dtype)
            best_scores = -(probabilities * ordered_ranks).sum(-1).view(B, G)
        else:
            relevance = batch["relevance"].to(device).reshape(-1)
            loss = nn.functional.smooth_l1_loss(output, relevance, beta=0.75)
            best_scores = output.view(B, G)
    else:
        output = model(batch["compounds"], batch["sentences"], batch["captions"])
        if task == "text-listnet":
            loss = listnet_loss(output, ranks)
            best_scores = output
        else:
            loss = nn.functional.smooth_l1_loss(output, ranks.float(), beta=3.0)
            best_scores = -output
    true_best = ranks.argmin(dim=-1)
    predicted_best = best_scores.argmax(dim=-1)
    top1_correct = (predicted_best == true_best).sum().item()
    return loss, top1_correct, B


def run_epoch(model, loader, task, device, optimizer=None):
    model.train(optimizer is not None)
    loss_total = 0.0
    correct = count = 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss, matched, batch_size = forward_loss_and_top1(model, batch, task, device)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        loss_total += loss.item() * batch_size
        correct += matched
        count += batch_size
    if not count:
        raise ValueError("Empty dataset")
    return {"loss": loss_total / count, "top1": correct / count}


def main(argv=None):
    args = parse_args(argv)
    if len(args.train_tsv) != len(args.train_root) or len(args.dev_tsv) != len(args.dev_root):
        raise ValueError("Each TSV must have its corresponding root directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    load_images = args.task.startswith("multimodal")
    train_loader = make_loader(args.train_tsv, args.train_root, load_images=load_images,
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    dev_loader = make_loader(args.dev_tsv, args.dev_root, load_images=load_images,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    model = make_model(args)
    optimizer = SGD((p for p in model.parameters() if p.requires_grad),
                    lr=args.learning_rate, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_top1 = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, args.task, args.device, optimizer)
        dev = run_epoch(model, dev_loader, args.task, args.device)
        print(f"Epoch {epoch}/{args.epochs}: train={train} dev={dev}", flush=True)
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "task": args.task,
            "config": vars(args),
            "metrics": dev,
        }
        if dev["loss"] < best_loss:
            best_loss = dev["loss"]
            torch.save(state, output_dir / "best_loss.pt")
        if dev["top1"] > best_top1:
            best_top1 = dev["top1"]
            torch.save(state, output_dir / "best_top1.pt")
        scheduler.step()
    print("Saved training checkpoints under:", output_dir)


if __name__ == "__main__":
    main()
