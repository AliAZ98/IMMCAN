"""AdMIRe candidate ranking dataset.

The TSV's expected_order is ground truth only. Candidate features are always
returned in their original image1..image5 order, not in ground-truth rank order.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset


NUM_CANDIDATES = 5


def parse_expected_order(value) -> list[str]:
    if isinstance(value, (tuple, list)):
        result = list(value)
    else:
        try:
            result = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            result = str(value).split()
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"Expected list of ranked filenames, received {value!r}")
    return [str(name).strip() for name in result]


class CandidateRankingDataset(Dataset):
    """One row = idiom/sentence with five candidate image-caption pairs."""

    def __init__(self, tsv_path: str | Path, image_root: str | Path, load_images=False):
        self.tsv_path = Path(tsv_path)
        self.image_root = Path(image_root)
        self.load_images = bool(load_images)
        if not self.tsv_path.is_file():
            raise FileNotFoundError(self.tsv_path)
        if self.load_images and not self.image_root.is_dir():
            raise FileNotFoundError(self.image_root)
        self.data = pd.read_csv(self.tsv_path, sep="\t")
        required = {"compound", "sentence", "sentence_type", "expected_order"}
        for i in range(1, NUM_CANDIDATES + 1):
            required.update({f"image{i}_name", f"image{i}_caption"})
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"{self.tsv_path}: missing columns {sorted(missing)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        compound = str(row["compound"]).strip().replace("'", "_")
        names = [str(row[f"image{i}_name"]).strip() for i in range(1, 6)]
        captions = [str(row[f"image{i}_caption"]) for i in range(1, 6)]
        correct_order = parse_expected_order(row["expected_order"])
        if len(correct_order) != 5 or len(set(correct_order)) != 5 or set(correct_order) != set(names):
            raise ValueError(f"{self.tsv_path}: row {index} has invalid expected_order")
        rank_of = {name: rank for rank, name in enumerate(correct_order)}
        ranks = torch.tensor([rank_of[name] for name in names], dtype=torch.long)
        relevance = 1.0 - ranks.float() / 4.0
        result = {
            "compound": compound,
            "sentence": str(row["sentence"]),
            "sentence_type": str(row["sentence_type"]),
            "names": names,
            "captions": captions,
            "ranks": ranks,
            "relevance": relevance,
        }
        if self.load_images:
            images = []
            for name in names:
                path = self.image_root / compound / name
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            result["images"] = images
        return result


def collate_candidates(examples):
    batch = {
        "compounds": [x["compound"] for x in examples],
        "sentences": [x["sentence"] for x in examples],
        "captions": [x["captions"] for x in examples],
        "names": [x["names"] for x in examples],
        "ranks": torch.stack([x["ranks"] for x in examples]),
        "relevance": torch.stack([x["relevance"] for x in examples]),
    }
    if "images" in examples[0]:
        batch["images"] = [im for example in examples for im in example["images"]]
    return batch


def make_loader(tsv_paths, image_roots, *, load_images, batch_size, shuffle, num_workers=0):
    """Fail loudly on unmatched TSV/root lists (zip would silently omit datasets)."""
    if not tsv_paths or len(tsv_paths) != len(image_roots):
        raise ValueError("Provide an equal, nonzero number of --*-tsv and --*-root arguments")
    datasets = [
        CandidateRankingDataset(tsv, root, load_images=load_images)
        for tsv, root in zip(tsv_paths, image_roots)
    ]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_candidates,
    )
