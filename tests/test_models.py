from types import SimpleNamespace

import torch

from immcan.attention import TextImageCoAttention
from immcan.losses import listnet_loss
from immcan.models import IMMCAN, TextOnlyRanker


class DummyIdiom:
    config = SimpleNamespace(hidden_size=8)

    def get_token_embeddings(self, compounds, sentences):
        n = len(sentences)
        return {
            "token_embeddings": torch.randn(n, 4, 8),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }


class DummyVLM:
    def get_token_embeddings(self, texts, images):
        n = len(texts)
        return {
            "text_tokens": torch.randn(n, 4, 16),
            "image_tokens": torch.randn(n, 3, 16),
            "text_mask": torch.ones(n, 4, dtype=torch.long),
        }


def test_attention_shapes():
    attn = TextImageCoAttention(token_dim=16, cross_dim=8, num_heads=2)
    text, image = attn(torch.randn(3, 4, 16), torch.randn(3, 5, 16))
    assert text.shape == (3, 4, 8)
    assert image.shape == (3, 5, 8)


def test_multimodal_modes_backward():
    for task in ("regression", "classification"):
        model = IMMCAN(DummyIdiom(), DummyVLM(), task=task,
                       token_dim=16, cross_dim=8, num_heads=2)
        result = model(["compound"] * 5, ["sentence"] * 5, [None] * 5)
        assert result.shape == ((5,) if task == "regression" else (5, 5))
        result.sum().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_text_listnet_scores():
    model = TextOnlyRanker(DummyIdiom(), cross_dim=8, num_heads=2, num_layers=1)
    predictions = model(["compound"], ["sentence"], [["a", "b", "c", "d", "e"]])
    assert predictions.shape == (1, 5)
    ranks = torch.tensor([[3, 0, 1, 4, 2]])
    loss = listnet_loss(predictions, ranks)
    loss.backward()
    assert torch.isfinite(loss)
