# test_metrics.py — metrics must match values computed by hand, not by the same library.
# Author: Colile Sibanda
#
# Correctness-critical. Written before evaluation/metrics.py.
# A wrong averaging mode or a transposed confusion axis produces a plausible
# table that says the opposite of the truth, and nothing downstream would flag it.

from __future__ import annotations

import numpy as np
import pytest

from docclf.evaluation.confusion import confusion_frame, most_confused_pairs
from docclf.evaluation.metrics import compute_metrics
from docclf.evaluation.predict import Predictions

CLASSES = ("alpha", "beta", "gamma")
TOLERANCE = 1e-9


def predictions(y_true: list[int], y_pred: list[int]) -> Predictions:
    """Purpose: wrap two label lists as a Predictions object, no model involved."""
    logits = np.zeros((len(y_true), len(CLASSES)), dtype=float)
    for row, label in enumerate(y_pred):
        logits[row, label] = 1.0
    return Predictions(
        y_true=np.array(y_true),
        y_pred=np.array(y_pred),
        logits=logits,
        paths=[f"/page/{index}.tif" for index in range(len(y_true))],
        class_names=CLASSES,
    )


@pytest.fixture
def imbalanced() -> Predictions:
    """Eight alphas all correct; one beta and one gamma both called alpha.

    Hand-computed: accuracy 0.8, macro-F1 0.296296..., so the two disagree
    exactly as an imbalanced case should.
    """
    y_true = [0] * 8 + [1, 2]
    y_pred = [0] * 10
    return predictions(y_true, y_pred)


@pytest.fixture
def balanced() -> Predictions:
    """Two of each class, one alpha misread as beta. Hand-computed below."""
    return predictions([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2])


def test_accuracy_is_hand_computed(imbalanced: Predictions) -> None:
    """Eight of ten correct is 0.8, whatever the class balance."""
    assert compute_metrics(imbalanced).accuracy == pytest.approx(0.8, abs=TOLERANCE)


def test_accuracy_and_macro_f1_disagree_on_the_imbalanced_case(
    imbalanced: Predictions,
) -> None:
    """The case the report depends on: a high accuracy hiding two dead classes.

    Per class by hand — alpha: precision 8/10, recall 8/8, F1 16/18 = 0.888...;
    beta and gamma: no prediction and no recall, F1 0. Macro-F1 = 0.888.../3.
    """
    metrics = compute_metrics(imbalanced)
    assert metrics.macro_f1 == pytest.approx(0.8888888888888888 / 3, abs=TOLERANCE)
    assert metrics.macro_f1 < metrics.accuracy - 0.5
    assert metrics.macro_precision == pytest.approx(0.8 / 3, abs=TOLERANCE)
    assert metrics.macro_recall == pytest.approx(1.0 / 3, abs=TOLERANCE)


def test_weighted_averages_are_hand_computed(imbalanced: Predictions) -> None:
    """Weighted by support: only alpha has non-zero F1, and it holds 8 of 10 rows."""
    metrics = compute_metrics(imbalanced)
    assert metrics.weighted_f1 == pytest.approx(0.8 * 0.8888888888888888, abs=TOLERANCE)
    assert metrics.weighted_precision == pytest.approx(0.8 * 0.8, abs=TOLERANCE)
    assert metrics.weighted_recall == pytest.approx(0.8, abs=TOLERANCE)


def test_macro_and_weighted_are_not_the_same_number(imbalanced: Predictions) -> None:
    """A guard against the classic slip of computing one and labelling it the other."""
    metrics = compute_metrics(imbalanced)
    assert metrics.macro_f1 != pytest.approx(metrics.weighted_f1, abs=1e-6)


def test_balanced_case_per_class_table(balanced: Predictions) -> None:
    """Hand-computed: alpha F1 2/3, beta F1 0.8, gamma F1 1.0; accuracy 5/6."""
    metrics = compute_metrics(balanced)
    table = metrics.per_class.set_index("class_name")
    assert metrics.accuracy == pytest.approx(5 / 6, abs=TOLERANCE)
    assert table.loc["alpha", "precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert table.loc["alpha", "recall"] == pytest.approx(0.5, abs=TOLERANCE)
    assert table.loc["alpha", "f1"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert table.loc["beta", "precision"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert table.loc["beta", "recall"] == pytest.approx(1.0, abs=TOLERANCE)
    assert table.loc["gamma", "f1"] == pytest.approx(1.0, abs=TOLERANCE)
    assert metrics.macro_f1 == pytest.approx((2 / 3 + 0.8 + 1.0) / 3, abs=TOLERANCE)


def test_per_class_support_sums_to_the_sample_count(imbalanced: Predictions) -> None:
    """Support is the true-class count, so it must total the number of pages."""
    metrics = compute_metrics(imbalanced)
    assert int(metrics.per_class["support"].sum()) == len(imbalanced.y_true)
    assert metrics.n_samples == len(imbalanced.y_true)


def test_confusion_rows_are_true_and_columns_are_predicted(balanced: Predictions) -> None:
    """The orientation the whole error analysis rests on, asserted explicitly."""
    frame = confusion_frame(balanced)
    assert list(frame.index) == list(CLASSES)
    assert list(frame.columns) == list(CLASSES)
    assert frame.loc["alpha", "beta"] == 1  # one true alpha was predicted beta
    assert frame.loc["beta", "alpha"] == 0  # and no true beta was predicted alpha
    assert frame.loc["gamma", "gamma"] == 2
    assert frame.index.name == "true"
    assert frame.columns.name == "predicted"


def test_confusion_totals_equal_the_sample_count(imbalanced: Predictions) -> None:
    """Every page lands in exactly one cell."""
    assert int(confusion_frame(imbalanced).to_numpy().sum()) == len(imbalanced.y_true)


def test_confusion_diagonal_matches_accuracy(balanced: Predictions) -> None:
    """The diagonal is the correct predictions; accuracy must agree with it."""
    frame = confusion_frame(balanced)
    diagonal = int(np.trace(frame.to_numpy()))
    assert diagonal / len(balanced.y_true) == pytest.approx(
        compute_metrics(balanced).accuracy, abs=TOLERANCE
    )


def test_most_confused_pairs_are_ordered_and_off_diagonal(balanced: Predictions) -> None:
    """The pair table drives report section 7, so it must name the direction."""
    pairs = most_confused_pairs(balanced, top_n=3)
    assert not pairs.empty
    top = pairs.iloc[0]
    assert top["true"] == "alpha" and top["predicted"] == "beta"
    assert int(top["count"]) == 1
    assert (pairs["true"] != pairs["predicted"]).all()


def test_mismatched_lengths_are_rejected() -> None:
    """A truncated prediction array must fail loudly, not silently zip short."""
    with pytest.raises(ValueError, match="same length"):
        Predictions(
            y_true=np.array([0, 1, 2]),
            y_pred=np.array([0, 1]),
            logits=np.zeros((2, 3)),
            paths=["/a.tif", "/b.tif"],
            class_names=CLASSES,
        )


def test_label_outside_the_class_set_is_rejected() -> None:
    """A label of 9 in an eight-class problem is a bug, not a rounding issue."""
    with pytest.raises(ValueError, match="outside"):
        predictions([0, 1, 9], [0, 1, 2])
