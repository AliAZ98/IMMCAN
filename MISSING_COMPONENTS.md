# Pending items before claiming full paper reproducibility

This is a **cleaned initial code release**, based on the source scripts in the
authors' shared Drive folder, not yet an end-to-end reproduction of every
table in the MWE 2026 paper.

## Author inputs required

- [ ] Links and redistribution permissions for original IMMCAN trained .pth
      checkpoints (classification, regression, text baselines, ablations).
- [ ] The original XLM-R MAGPIE fine-tuning procedure, training split, and
      checkpoint provenance/URL (only the exported checkpoint is in Drive).
- [ ] Official AdMIRe dataset retrieval instructions, licensing, and exact
      AdMIRe 1.0 vs 2.0 train/dev/test/xeval versions.
- [ ] Remaining experiment/evaluation scripts used for reported top-1 and NDCG;
      in particular zero-shot evaluation and submission-format conversion.
- [ ] Exact model-selection protocol and numbers for published result tables,
      including language-specific results and random seeds.
- [ ] Back-translation and synonym replacement generation scripts, seeds,
      and augmentation provenance. Only their resulting TSVs were shared.
- [ ] Evidence/check of whether the published multimodal visual-text input
      was the idiom sentence or the candidate image caption. The available
      main_train.py passes sentence + image to Jina CLIP, while the README
      describes caption-image grounding. This release defaults to original
      script behavior; --visual-text caption is experimental until verified.
- [ ] Conversion/loading of legacy .pth state_dict key names to the new package.
- [ ] Pin exact dependency versions, pretrained revision hashes and compute
      details once the full experiment environment is provided.
- [ ] Validate numerical reproduction on the complete official datasets.

## Important fixes made in this refactor

1. Original classification: casting relevance targets
   [1.0, 0.75, 0.5, 0.25, 0.0] to long resulted in [1, 0, 0, 0, 0].
   Training now uses distinct ranks [0, 1, 2, 3, 4].
2. Original loaders constructed samples in ground-truth expected_order, so
   choosing always the first candidate gave perfect top-1 regardless of
   model quality. The new loader preserves original image column order and
   derives true ranks separately.
3. Original text regression script listed 3 augmentation TSVs but only 2
   root directories. The new loader raises an error on such mismatches.
4. Text-only mode no longer decodes unnecessary image files.
5. Frozen backbone weights are not saved alongside small fusion checkpoints.

These fixes mean new training metrics and checkpoint parameters **need not
match old runs**. Do not present new numbers as previously published results
without rerunning/validating the final benchmark.
