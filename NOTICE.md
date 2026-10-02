# mindor-laya-trainer

This package provides a programmatic Python API for fine-tuning
[Laya](https://github.com/NandhaKishorM/laya) typed-decision models.

The training recipe (RLCD + strictly-proper-scoring-rule rewards + GRPO-style
policy gradient, mixed with soft-target cross-entropy; temperature calibration
with LBFGS) is adapted from the upstream reference script
`notebooks/laya_finetune_typed_decisions_mps.py`.  Shared primitives
(`build_model`, `build_sequence`, `proper_reward`, `render_options`, `QTYPES`)
are imported directly from the installed `laya` package at runtime; no
upstream source files are vendored.

## Upstream

- Repository: https://github.com/NandhaKishorM/laya
- License: Apache-2.0

## This package

- License: Apache-2.0 (see `LICENSE`)
- Code original to this package: Copyright 2026 Hanyeol Cho
