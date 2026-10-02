from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "resolve_option_keys",
    "coerce_answer_to_probabilities",
    "build_training_item",
    "iter_training_items",
]


_NOUL_FALSE_INDEX = 0
_NOUL_TRUE_INDEX = 1


def resolve_option_keys(question_type: str, criteria: Any) -> List[str]:
    """Return the ordered option keys for one question — the order that
    laya.common.render_options iterates over, which also fixes the logit
    index per option.

      * choice: dict insertion order of `criteria`.
      * noul:   fixed ("false", "true") so index 0 = false, 1 = true, matching
                the convention in `build_training_item` upstream.
      * score:  numeric string keys "0".."n-1" for n levels; `criteria` can be
                a list of level descriptions or a dict (fallback: 4 levels).
    """
    if question_type == "noul":
        return ["false", "true"]

    if question_type == "score":
        if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            levels = len(criteria)
        elif isinstance(criteria, Mapping):
            levels = len(criteria)
        else:
            levels = 4
        return [str(index) for index in range(levels)]

    # choice (and anything else) — fall back to criteria keys.
    if isinstance(criteria, Mapping):
        return [str(option_id) for option_id in criteria.keys()]
    if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
        return [str(option_id) for option_id in criteria]
    raise ValueError(f"cannot resolve option keys for criteria of type {type(criteria).__name__}")


def coerce_answer_to_probabilities(
    question_type: str,
    answer: Any,
    option_keys: List[str],
) -> List[float]:
    """Map a model-compose-shaped answer to a probability vector aligned to
    `option_keys`.

    Accepts three answer shapes:
      1. A full probability dict ({option_id: probability}) — upstream shape.
      2. A native value (string for choice/score, bool for noul).  Expanded to
         a 1-hot distribution so the resulting CE loss still has signal.
      3. For `score`, an integer / numeric string pointing at the level index.
    """
    if isinstance(answer, Mapping):
        distribution = [float(answer.get(key, 0.0)) for key in option_keys]
        total = sum(distribution)
        if total <= 0:
            raise ValueError(f"answer probability dict {answer!r} sums to 0 over options {option_keys!r}")
        return [value / total for value in distribution]

    if question_type == "noul":
        probabilities = [0.0, 0.0]
        probabilities[_NOUL_TRUE_INDEX if bool(answer) else _NOUL_FALSE_INDEX] = 1.0
        return probabilities

    if question_type == "score":
        # Accept either the level index or its string label.
        if isinstance(answer, bool):
            raise ValueError(f"score answer must be an index or option id, got bool {answer!r}")
        if isinstance(answer, int):
            gold = str(answer)
        else:
            gold = str(answer)
    else:
        gold = str(answer)

    try:
        index = option_keys.index(gold)
    except ValueError as exc:
        raise ValueError(
            f"answer {answer!r} not in option keys {option_keys!r} for question type {question_type!r}"
        ) from exc

    probabilities = [0.0] * len(option_keys)
    probabilities[index] = 1.0
    return probabilities


def build_training_item(
    tokenizer: Any,
    cfg: Dict[str, Any],
    state: str,
    question: Dict[str, Any],
    answer: Any,
    build_sequence: Any,
    render_options: Any,
    qtypes_map: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """One encoded training item, matching the shape upstream's `collate`
    consumes: {ids, markers, qtype, target, label}.

    Returns None when `build_sequence` can't produce a usable sequence — for
    example when the state + schema overflow the encoder budget and the markers
    disagree with the option count.
    """
    question_type = question.get("type") or question.get("t")
    if question_type is None:
        raise ValueError(f"question is missing 'type': {question!r}")

    criteria = question.get("criteria") or question.get("crit") or question.get("options")
    if criteria is None:
        if question_type != "noul":
            raise ValueError(f"question is missing 'criteria'/'options': {question!r}")
        criteria = {}

    option_keys = resolve_option_keys(question_type, criteria)
    target = coerce_answer_to_probabilities(question_type, answer, option_keys)

    label = max(range(len(target)), key=target.__getitem__)
    rendered_options = render_options({"t": question_type, "crit": criteria})

    if len(rendered_options) != len(target):
        # Protect the caller: a mismatch here would otherwise crash the forward
        # pass with an unhelpful tensor-shape error.
        raise ValueError(
            f"rendered {len(rendered_options)} options from criteria but computed {len(target)} targets "
            f"— options and schema disagree for question {question!r}"
        )

    instructions = question.get("instructions") or question.get("ins") or ""
    sequence, markers = build_sequence(
        tokenizer,
        state,
        {"t": question_type, "ins": instructions, "crit": criteria},
        cfg["max_len"],
        cfg["head_max_len"],
    )

    if len(markers) != len(rendered_options):
        return None

    return {
        "ids":     sequence,
        "markers": markers,
        "qtype":   qtypes_map[question_type],
        "target":  target,
        "label":   label,
    }


def iter_training_items(
    dataset: Any,
    tokenizer: Any,
    cfg: Dict[str, Any],
    state_column: str,
    schema_column: str,
    answers_column: str,
    build_sequence: Any,
    render_options: Any,
    qtypes_map: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], int]:
    """Flatten a {state, schema, answers} dataset into per-question training
    items.  Returns (items, skipped) where `skipped` counts items dropped by
    build_sequence (overflow of max_len / head_max_len).
    """
    items: List[Dict[str, Any]] = []
    skipped = 0

    for row in dataset:
        state = row[state_column]
        schema = row[schema_column]
        answers = row[answers_column]

        for question_id, question in schema.items():
            if question_id not in answers:
                continue

            item = build_training_item(
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                question=question,
                answer=answers[question_id],
                build_sequence=build_sequence,
                render_options=render_options,
                qtypes_map=qtypes_map,
            )

            if item is None:
                skipped += 1
            else:
                items.append(item)

    return items, skipped
