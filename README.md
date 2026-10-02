# mindor-laya-trainer

Programmatic fine-tuning for [Laya](https://github.com/NandhaKishorM/laya)
typed-decision models.

Laya's upstream distribution ships training only as a notebook /
single-process reference script.  This package exposes the same recipe
(RLCD + strictly-proper-scoring-rule rewards + GRPO-style policy gradient,
mixed with soft-target cross-entropy; LBFGS temperature calibration) as a
regular Python function, so it can be driven from a config / orchestrator
instead of invoked from a shell.

## Install

```bash
pip install git+https://github.com/hanyeol/mindor-laya-trainer.git
```

## Usage

```python
from datasets import load_dataset
from laya_trainer import LayaTrainerConfig, train

train_ds = load_dataset("json", data_files="train.jsonl", split="train")

config = LayaTrainerConfig(
    model_dir="./laya_base",                         # snapshot of convaiinnovations/laya (any preset)
    output_dir="./runs/laya-ft",
    num_train_epochs=4,
    micro_batch_size=2,
    gradient_accumulation_steps=16,
    freeze_encoder=False,                            # True = heads-only training
)

result = train(train_ds, config)
print(result)
# {
#   "output_dir":    "./runs/laya-ft",
#   "train_loss":    0.37,
#   "temperatures":  [1.1, 0.9, 1.3],
#   "items":         12480,
#   "skipped_items": 3,
# }
```

## Dataset format

Each row has three fields (same shape as typed-decision inference input, plus
a ground-truth `answers` map):

```json
{
  "state": "User said: book me a flight to Tokyo tomorrow.",
  "schema": {
    "intent":   { "type": "choice", "instructions": "...", "criteria": {"buy": "...", "ask": "..."} },
    "urgent":   { "type": "noul",   "instructions": "...", "criteria": {} },
    "priority": { "type": "score",  "instructions": "...", "criteria": ["low", "medium", "high"] }
  },
  "answers": {
    "intent":   "buy",
    "urgent":   true,
    "priority": "high"
  }
}
```

`answers` values may also be a full probability dict (e.g. `{"true": 0.9,
"false": 0.1}`) — soft targets propagate through the CE term exactly as the
upstream notebook consumes them.

## Output

`train()` writes a checkpoint directory shaped exactly like a `laya` snapshot:

```
runs/laya-ft/
├── model.safetensors
├── rl_agent_config.json        # includes fitted temperatures
├── encoder/                    # encoder config
├── tokenizer/
└── checkpoint_latest/          # intermediate epoch checkpoint
```

The directory is a drop-in replacement for a `laya` snapshot path — load with
`laya.Agent(model_id_or_path="./runs/laya-ft")`.

## Loss

Adapted directly from `notebooks/laya_finetune_typed_decisions_mps.py` in the
upstream repository.  Per-step loss is `loss_rl + loss_ce`:

- `loss_rl`: GRPO-style policy gradient over `rollout_samples` noisy copies of
  the logits, with advantage normalized across rollouts.  The reward is a mix
  of the spherical and ranked-probability-score strictly-proper rules
  (`reward_spherical_weight`, `reward_rps_weight`).
- `loss_ce`: soft-target cross-entropy against the per-option probability
  distribution derived from `answers`.

Sigma is linearly faded from `sigma_start` → `sigma_end` across epochs.

## License

Apache-2.0.  See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
