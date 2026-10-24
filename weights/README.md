# Model weights (to be linked)

This directory intentionally contains **no model weights**. The repository
does not redistribute third-party pretrained checkpoints or evaluation data.

Expected resources:

1. Local XLM-R MAGPIE fine-tuned checkpoint directory:
   weights/xlmr-magpie-seqcls-manual/
   It should contain config.json, tokenizer files, and model.safetensors.
2. Jina CLIP v2, loaded by Transformers as jinaai/jina-clip-v2
   (download/cache upstream model; upstream license/terms apply).
3. IMMCAN fusion/checkpoint .pt/.pth files for each reported experiment.
   **Download URL: TODO — to be supplied by the authors.**
4. Text-only listwise regression/listnet checkpoint files.
   **Download URL: TODO — to be supplied by the authors.**

Checkpoint schema of this cleaned release:
epoch, model_state_dict, optimizer_state_dict, task, config, metrics.

Caution: the original Drive .pth checkpoints were generated under older
module/class names. They might not load into the new package without a
key-name conversion script. Keep originals intact until mapping is verified.
Do not publish personal paths or credentials in checkpoint metadata.
