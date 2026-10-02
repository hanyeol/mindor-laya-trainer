"""Unit tests for answer-shape coercion — no laya / torch dependency."""
import pytest

from laya_trainer.data import (
    coerce_answer_to_probabilities,
    resolve_option_keys,
)


def test_resolve_noul_fixed_order():
    assert resolve_option_keys("noul", {}) == ["false", "true"]


def test_resolve_choice_preserves_dict_order():
    assert resolve_option_keys("choice", {"a": "x", "b": "y"}) == ["a", "b"]


def test_resolve_choice_from_list():
    assert resolve_option_keys("choice", ["a", "b"]) == ["a", "b"]


def test_resolve_score_from_list_length():
    assert resolve_option_keys("score", ["low", "medium", "high"]) == ["0", "1", "2"]


def test_coerce_native_choice_becomes_onehot():
    assert coerce_answer_to_probabilities("choice", "b", ["a", "b", "c"]) == [0.0, 1.0, 0.0]


def test_coerce_native_noul_true_at_index_1():
    assert coerce_answer_to_probabilities("noul", True, ["false", "true"]) == [0.0, 1.0]
    assert coerce_answer_to_probabilities("noul", False, ["false", "true"]) == [1.0, 0.0]


def test_coerce_native_score_integer_index():
    assert coerce_answer_to_probabilities("score", 2, ["0", "1", "2", "3"]) == [0.0, 0.0, 1.0, 0.0]


def test_coerce_probability_dict_normalizes():
    probabilities = coerce_answer_to_probabilities(
        "noul", {"true": 0.9, "false": 0.1}, ["false", "true"],
    )
    assert probabilities == [0.1, 0.9]


def test_coerce_probability_dict_renormalizes():
    probabilities = coerce_answer_to_probabilities(
        "choice", {"a": 2.0, "b": 2.0}, ["a", "b"],
    )
    assert probabilities == pytest.approx([0.5, 0.5])


def test_coerce_rejects_unknown_option():
    with pytest.raises(ValueError, match="not in option keys"):
        coerce_answer_to_probabilities("choice", "xyz", ["a", "b"])


def test_coerce_rejects_zero_probability_dict():
    with pytest.raises(ValueError, match="sums to 0"):
        coerce_answer_to_probabilities("choice", {"a": 0.0, "b": 0.0}, ["a", "b"])
