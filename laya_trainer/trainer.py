"""Fine-tune a Laya typed-decision checkpoint.

Loss recipe is adapted from `notebooks/laya_finetune_typed_decisions_mps.py`
in the upstream Laya repository (RLCD policy gradient on sampled logit noise
+ soft-target cross-entropy; LBFGS temperature calibration at the end).
Primitives (`build_model`, `build_sequence`, `proper_reward`, `render_options`,
`QTYPES`) are imported from the installed `laya` package at runtime.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import os
import random
from pathlib import Path

from .config import LayaTrainerConfig
from .data import iter_training_items

__all__ = [ "train" ]


def _choose_device(requested: str) -> Any:
    import torch

    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")

    return torch.device(requested)


def _load_laya_primitives():
    """Pull the shared training primitives out of the installed laya package.

    Isolated into a helper so the import failure gives a readable error when
    laya is missing rather than crashing deep inside the trainer.
    """
    try:
        from laya.agent import _fix_tokenizer_config
        from laya.common import QTYPES, build_model, build_sequence, proper_reward, render_options
    except ImportError as exc:
        raise ImportError(
            "mindor-laya-trainer requires the `laya` package (>=0.3.20). "
            "Install it with `pip install laya` and retry."
        ) from exc

    return {
        "fix_tokenizer_config": _fix_tokenizer_config,
        "qtypes":               QTYPES,
        "build_model":          build_model,
        "build_sequence":       build_sequence,
        "proper_reward":        proper_reward,
        "render_options":       render_options,
    }


def _collate(items: List[Dict[str, Any]], pad_id: int):
    import torch

    batch_size = len(items)
    seq_len = max(len(item["ids"]) for item in items)
    marker_count = max(len(item["markers"]) for item in items)

    input_ids   = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention   = torch.zeros((batch_size, seq_len), dtype=torch.long)
    marker_pos  = torch.zeros((batch_size, marker_count), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, marker_count), dtype=torch.bool)
    target      = torch.zeros((batch_size, marker_count), dtype=torch.float32)

    for row_index, item in enumerate(items):
        length = len(item["ids"])
        input_ids[row_index, :length]  = torch.tensor(item["ids"], dtype=torch.long)
        attention[row_index, :length]  = 1

        marker_len = len(item["markers"])
        marker_pos[row_index, :marker_len]  = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[row_index, :marker_len] = True

        target[row_index, :len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)

    qtypes = torch.tensor([item["qtype"] for item in items], dtype=torch.long)
    return input_ids, attention, marker_pos, marker_mask, target, qtypes


def _fit_temperature(samples: List[Tuple[Any, List[float]]]) -> float:
    import torch

    if len(samples) < 10:
        return 1.0

    marker_max = max(len(logits) for logits, _ in samples)
    logits_tensor  = torch.full((len(samples), marker_max), -1e4)
    targets_tensor = torch.zeros((len(samples), marker_max))

    for row_index, (values, target) in enumerate(samples):
        logits_tensor[row_index, :len(values)]  = torch.as_tensor(values)
        targets_tensor[row_index, :len(target)] = torch.as_tensor(target, dtype=torch.float32)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets_tensor * torch.log_softmax(logits_tensor / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), 0.1, 10.0).item())


def _save_checkpoint(
    model: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    output_dir: str,
    epoch: int,
    final: bool,
) -> None:
    from safetensors.torch import save_file

    destination = Path(output_dir) if final else Path(output_dir) / "checkpoint_latest"
    destination.mkdir(parents=True, exist_ok=True)

    weights = {
        name: value.detach().half().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(weights, str(destination / "model.safetensors"))
    model.encoder.config.save_pretrained(destination / "encoder")
    tokenizer.save_pretrained(destination / "tokenizer")

    with open(destination / "checkpoint_meta.json", "w") as handle:
        json.dump({"epoch": epoch, "final": final}, handle, indent=2)
    with open(destination / "rl_agent_config.json", "w") as handle:
        json.dump(cfg, handle, indent=2)


def train(
    train_dataset: Any,
    config: LayaTrainerConfig,
    eval_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fine-tune a Laya checkpoint on the supplied dataset.

    Returns {output_dir, train_loss, temperatures, items, skipped_items}.

    `eval_dataset` is accepted for API symmetry with the other mindor trainers
    but Laya's upstream recipe performs calibration against a held-out slice of
    the training set, not a separate eval split.  When `eval_dataset` is
    provided its items are appended to the calibration pool.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    primitives = _load_laya_primitives()
    device = _choose_device(config.device)

    model_dir = Path(config.model_dir)
    primitives["fix_tokenizer_config"](str(model_dir))

    with open(model_dir / "rl_agent_config.json") as handle:
        cfg: Dict[str, Any] = json.load(handle)

    cfg.update({"max_tokens_per_batch": 2048, "max_len": cfg.get("max_len", 1024), "head_max_len": cfg.get("head_max_len", 256)})

    if config.gradient_checkpointing:
        cfg["gradient_checkpointing"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = primitives["build_model"](cfg, encoder_dir=model_dir / "encoder")
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    model.float()

    if config.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if config.head_checkpointing:
        model.head_checkpointing = True

    model.to(device).train()

    train_items, train_skipped = iter_training_items(
        dataset=train_dataset,
        tokenizer=tokenizer,
        cfg=cfg,
        state_column=config.state_column,
        schema_column=config.schema_column,
        answers_column=config.answers_column,
        build_sequence=primitives["build_sequence"],
        render_options=primitives["render_options"],
        qtypes_map=primitives["qtypes"],
    )
    if not train_items:
        raise ValueError("empty training set after preprocessing — every row was dropped by build_sequence")

    # Carve a calibration slice out of the training items, as upstream does.
    rng = random.Random(config.seed)
    order = list(range(len(train_items)))
    rng.shuffle(order)
    calibration_count = min(config.calibration_max, len(train_items) // 10)
    calibration_items = [train_items[i] for i in sorted(order[:calibration_count])]
    train_items       = [train_items[i] for i in sorted(order[calibration_count:])]

    if eval_dataset is not None:
        extra_items, _ = iter_training_items(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            cfg=cfg,
            state_column=config.state_column,
            schema_column=config.schema_column,
            answers_column=config.answers_column,
            build_sequence=primitives["build_sequence"],
            render_options=primitives["render_options"],
            qtypes_map=primitives["qtypes"],
        )
        calibration_items.extend(extra_items)

    # Build parameter groups.  Encoder-vs-head split lets us (a) pin different
    # learning rates per group, and (b) honour freeze_encoder by zeroing the
    # encoder's LR instead of disabling requires_grad on half the model — the
    # latter breaks gradient checkpointing on some torch builds.
    encoder_params = [parameter for name, parameter in model.named_parameters() if "encoder." in name]
    head_params    = [parameter for name, parameter in model.named_parameters() if "encoder." not in name]

    encoder_lr = 0.0 if config.freeze_encoder else config.encoder_learning_rate
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params,    "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(len(train_items) / config.micro_batch_size / config.gradient_accumulation_steps))
    total_updates = steps_per_epoch * config.num_train_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=config.min_learning_rate,
    )

    sigma_fade = (config.sigma_end - config.sigma_start) / max(1, config.num_train_epochs - 1)

    final_avg_loss = 0.0

    for epoch_index in range(config.num_train_epochs):
        rng_epoch = random.Random(config.seed + epoch_index)
        rng_epoch.shuffle(train_items)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        batch_count = 0
        sigma = config.sigma_start + sigma_fade * epoch_index

        for batch_start in range(0, len(train_items), config.micro_batch_size):
            batch = train_items[batch_start:batch_start + config.micro_batch_size]
            ids, attention, positions, mask, target, qtypes = _collate(batch, tokenizer.pad_token_id)
            ids       = ids.to(device)
            attention = attention.to(device)
            positions = positions.to(device)
            mask      = mask.to(device)
            target    = target.to(device)
            qtypes    = qtypes.to(device)

            logits, activation = model(ids, attention, positions, mask, qtypes)
            logits = logits.float()
            marker_counts = mask.sum(-1, keepdim=True).float()

            noise = torch.randn((config.rollout_samples,) + logits.shape, device=device) * sigma * mask
            # Center the noise per record so the rollout stays mean-preserving.
            noise = (noise - noise.sum(-1, keepdim=True) / marker_counts) * mask
            noisy_logits = logits.detach().unsqueeze(0) + noise
            probabilities = torch.softmax(noisy_logits.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                reward = primitives["proper_reward"](
                    probabilities,
                    target.unsqueeze(0),
                    qtypes,
                    mask,
                    w_sph=config.reward_spherical_weight,
                    w_rps=config.reward_rps_weight,
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            logp = -(((noisy_logits - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()

            # `0.0 * activation.sum()` keeps the activation tensor in the autograd
            # graph when gradient checkpointing re-materializes it, matching the
            # upstream script.
            loss = (loss_rl + loss_ce + 0.0 * activation.sum()) / config.gradient_accumulation_steps
            loss.backward()

            batch_count += 1
            is_last_microbatch = batch_start + config.micro_batch_size >= len(train_items)
            if batch_count % config.gradient_accumulation_steps == 0 or is_last_microbatch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * config.gradient_accumulation_steps
            total_loss += step_loss
            if config.logging_steps and batch_count % config.logging_steps == 0:
                print(f"epoch {epoch_index + 1}/{config.num_train_epochs} step {batch_count} loss={step_loss:.4f}", flush=True)

        avg_loss = total_loss / max(1, batch_count)
        final_avg_loss = avg_loss
        print(f"epoch {epoch_index + 1}/{config.num_train_epochs} complete; avg_loss={avg_loss:.4f}", flush=True)
        _save_checkpoint(model, tokenizer, cfg, config.output_dir, epoch_index + 1, final=False)

    # Temperature calibration — one per question type.
    model.eval()
    sample_bucket: List[List[Tuple[Any, List[float]]]] = [[] for _ in range(len(primitives["qtypes"]))]
    with torch.no_grad():
        for batch_start in range(0, len(calibration_items), config.micro_batch_size):
            batch = calibration_items[batch_start:batch_start + config.micro_batch_size]
            ids, attention, positions, mask, _target, qtypes = _collate(batch, tokenizer.pad_token_id)
            logits, _ = model(
                ids.to(device), attention.to(device), positions.to(device), mask.to(device), qtypes.to(device)
            )
            for row_index, item in enumerate(batch):
                sample_bucket[item["qtype"]].append(
                    (logits[row_index, :len(item["markers"])].cpu(), item["target"])
                )

    temperatures = [_fit_temperature(group) if group else 1.2 for group in sample_bucket]

    _save_checkpoint(model, tokenizer, cfg, config.output_dir, config.num_train_epochs, final=True)
    cfg.update({
        "fine_tuned":  True,
        "model_name":  "laya-typed-decisions",
        "temperature": temperatures,
    })
    cfg.pop("temperature_by_options", None)

    with open(Path(config.output_dir) / "rl_agent_config.json", "w") as handle:
        json.dump(cfg, handle, indent=2)
    latest_dir = Path(config.output_dir) / "checkpoint_latest"
    if latest_dir.exists():
        with open(latest_dir / "rl_agent_config.json", "w") as handle:
            json.dump(cfg, handle, indent=2)

    return {
        "output_dir":    str(Path(config.output_dir).resolve()),
        "train_loss":    final_avg_loss,
        "temperatures":  temperatures,
        "items":         len(train_items),
        "skipped_items": train_skipped,
    }
