from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LayaTrainerConfig:
    """Configuration for fine-tuning a Laya typed-decision checkpoint.

    Mirrors the knobs exposed by the upstream reference script
    (`notebooks/laya_finetune_typed_decisions_mps.py`), lifted into a dataclass
    so model-compose can supply the values directly from YAML.
    """

    # Checkpoint location — a snapshot of `convaiinnovations/laya` (or a
    # preset subfolder) that contains `rl_agent_config.json`, `encoder/`,
    # `tokenizer/`, and `model.safetensors`.
    model_dir: str
    output_dir: str

    # Dataset column layout.  Rows carry {state, schema, answers}:
    #   * state   — string
    #   * schema  — dict keyed by question id; each entry has
    #               {type, instructions, criteria|options, ...}
    #   * answers — dict keyed by question id; each entry is either a native
    #               value (choice: option id, noul: bool, score: int|str) or a
    #               full probability dict {option_id: probability}.
    state_column: str = "state"
    schema_column: str = "schema"
    answers_column: str = "answers"

    # Training schedule
    num_train_epochs: int = 4
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    encoder_learning_rate: float = 2.5e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    min_learning_rate: float = 1e-6

    # RLCD rollout hyperparameters.  Upstream fades sigma from 0.4 → 0.1 across
    # epochs; the trainer applies the same linear schedule unless overridden.
    rollout_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    reward_spherical_weight: float = 0.75
    reward_rps_weight: float = 1.0

    # Head / encoder freezing.  `freeze_encoder=True` zeros out the encoder's
    # learning rate (parameters still participate in forward passes but their
    # gradients never update them), matching the "heads-only" mode exposed by
    # the inference-side `freeze_encoder` toggle.
    freeze_encoder: bool = False

    # Checkpointing / memory
    gradient_checkpointing: bool = True
    head_checkpointing: bool = True

    # Temperature calibration
    calibration_max: int = 400

    # Reproducibility
    seed: int = 20260922

    # Device
    device: str = "auto"

    # Logging cadence (steps)
    logging_steps: int = 100
