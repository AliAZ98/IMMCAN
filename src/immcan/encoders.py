"""Frozen original pretrained encoders from the Drive experiments.

These wrappers intentionally are not nn.Module children: trainable-model
checkpoints store just fusion/heads, not gigabytes of frozen backbone weights.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
from transformers import AutoProcessor, AutoTokenizer


class IdiomEncoder:
    """Local fine-tuned XLM-R figurative/literal sentence encoder."""

    def __init__(self, model_dir, device="cpu", max_length=128):
        self.device = torch.device(device)
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.base_model = self.model.base_model
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def get_token_embeddings(self, compounds, sentences):
        inputs = self.tokenizer(
            list(compounds), list(sentences), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = {k: value.to(self.device) for k, value in inputs.items()}
        out = self.base_model(**inputs)
        return {"token_embeddings": out.last_hidden_state,
                "attention_mask": inputs["attention_mask"]}


class JinaCLIPV2Encoder:
    """Jina CLIP v2 patch/text embeddings using original model internals.

    Access to internal modules may need adjustments when upstream model code changes.
    The model requires trust_remote_code=True, so review the upstream model code.
    """

    def __init__(self, model_id="jinaai/jina-clip-v2", device="cpu", max_length=128):
        self.device = torch.device(device)
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def get_token_embeddings(self, texts, images):
        inputs = self.processor(
            text=list(texts), images=images, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        text_model = self.model.text_model.transformer.roberta
        text_tokens = text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"], return_dict=True,
        ).last_hidden_state

        patch_cache = {}
        def capture_patches(module, arguments, output):
            patch_cache["tokens"] = output

        vision = self.model.vision_model
        handle = vision.norm.register_forward_hook(capture_patches)
        try:
            try:
                vision(x=inputs["pixel_values"])
            except TypeError:
                vision(inputs["pixel_values"])
        finally:
            handle.remove()
        if "tokens" not in patch_cache:
            raise RuntimeError("Cannot extract Jina image patch tokens; verify model version")

        return {
            "text_tokens": text_tokens,
            "image_tokens": patch_cache["tokens"],
            "text_mask": inputs["attention_mask"],
        }
