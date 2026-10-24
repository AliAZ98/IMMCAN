"""Fusion models and text-only baselines reconstructed from the Drive code."""
from __future__ import annotations

import torch
from torch import nn

from .attention import IdiomMMCrossAttention, TextImageCoAttention


def masked_mean(tokens, mask=None):
    if mask is None:
        return tokens.mean(dim=1)
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)


class ScoreHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, dropout=0.3, out_dim=1, deep=False):
        super().__init__()
        parts = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        if deep:
            parts += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        parts += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*parts)

    def forward(self, x):
        return self.net(x)


class IMMCAN(nn.Module):
    """Frozen XLM-R + Jina CLIP v2; trainable bidirectional fusion and output head.

    visual_text='sentence' exactly follows the supplied multimodal training code.
    visual_text='caption' is provided as a paper-alignment experiment, NOT
    claimed to be part of the originally reported results.
    """

    def __init__(self, idiom_encoder, vision_language_encoder, *, task="regression",
                 token_dim=1024, cross_dim=512, num_heads=8,
                 num_layers=1, dropout=0.3, visual_text="sentence"):
        super().__init__()
        if task not in {"regression", "classification"}:
            raise ValueError(task)
        if visual_text not in {"sentence", "caption"}:
            raise ValueError(visual_text)
        self.idiom_encoder = idiom_encoder
        self.vlm_encoder = vision_language_encoder
        self.task = task
        self.visual_text = visual_text
        self.coattention = TextImageCoAttention(
            token_dim=token_dim, cross_dim=cross_dim,
            num_heads=num_heads, num_layers=num_layers,
        )
        self.idiom_mm = IdiomMMCrossAttention(
            idiom_dim=idiom_encoder.config.hidden_size,
            mm_dim=2 * cross_dim, cross_dim=cross_dim,
            num_heads=num_heads, num_layers=1,
        )
        # The original regression code used a deep 2-hidden-layer MLP.
        self.head = ScoreHead(
            input_dim=cross_dim, dropout=dropout,
            out_dim=5 if task == "classification" else 1, deep=True,
        )

    def forward(self, compounds, sentences, images, captions=None):
        with torch.no_grad():
            idiom = self.idiom_encoder.get_token_embeddings(compounds, sentences)
            visual_texts = captions if self.visual_text == "caption" else sentences
            if visual_texts is None:
                raise ValueError("Candidate captions are required for visual_text='caption'")
            visual = self.vlm_encoder.get_token_embeddings(visual_texts, images)
        device = self.head.net[0].weight.device
        idiom_tokens = idiom["token_embeddings"].to(device).float()
        idiom_mask = idiom["attention_mask"].to(device)
        text_tokens = visual["text_tokens"].to(device).float()
        image_tokens = visual["image_tokens"].to(device).float()
        text_mask = visual["text_mask"].to(device)
        fused_text, fused_image = self.coattention(
            text_tokens, image_tokens, text_mask=text_mask,
        )
        multimodal = torch.cat(
            [masked_mean(fused_text, text_mask), masked_mean(fused_image)],
            dim=-1,
        )
        fused_idiom = self.idiom_mm(idiom_tokens, multimodal, idiom_mask)
        pooled = masked_mean(fused_idiom, idiom_mask)
        output = self.head(pooled)
        return output if self.task == "classification" else output.squeeze(-1)


class TextOnlyRanker(nn.Module):
    """Frozen XLM-R; trainable sentence-caption bidirectional co-attention.

    Returns one scalar per candidate [batch, 5]. For listnet, higher is better;
    for regression against ordinal ranks, lower is better.
    """

    def __init__(self, idiom_encoder, *, cross_dim=512,
                 num_heads=8, num_layers=8, dropout=0.3):
        super().__init__()
        self.idiom_encoder = idiom_encoder
        self.coattention = TextImageCoAttention(
            token_dim=idiom_encoder.config.hidden_size,
            cross_dim=cross_dim, num_heads=num_heads, num_layers=num_layers,
        )
        self.head = ScoreHead(input_dim=cross_dim, dropout=dropout)

    def forward(self, compounds, sentences, captions_grouped):
        batch_size = len(compounds)
        if any(len(c) != 5 for c in captions_grouped):
            raise ValueError("Exactly five captions per row are required")
        compounds_flat = [compound for compound in compounds for _ in range(5)]
        captions_flat = [caption for group in captions_grouped for caption in group]
        with torch.no_grad():
            source = self.idiom_encoder.get_token_embeddings(compounds, sentences)
            candidates = self.idiom_encoder.get_token_embeddings(compounds_flat, captions_flat)
        device = self.head.net[0].weight.device
        src = source["token_embeddings"].to(device).float()
        source_mask = source["attention_mask"].to(device)
        cap = candidates["token_embeddings"].to(device).float()
        cap_mask = candidates["attention_mask"].to(device)
        src = src.repeat_interleave(5, dim=0)
        source_mask = source_mask.repeat_interleave(5, dim=0)
        fused_sentence, _ = self.coattention(
            src, cap, text_mask=source_mask, image_mask=cap_mask,
        )
        scores = self.head(masked_mean(fused_sentence, source_mask)).squeeze(-1)
        return scores.view(batch_size, 5)
