# slides/

The oral presentation is delivered as a slide deck, separately from this
repository. Both group members present a section, and both answer questions on
any part of the work.

## What the presentation covers

The problem and why it matters, the dataset and class definitions, the three
model families and the transfer-learning strategy, the experimental setup, the
main results with the confusion matrices, the important errors and limitations,
and the recommended model.

## Where the material comes from

The figures are in [`outputs/report_figures/`](../outputs/report_figures/),
and each model's confusion matrix and error examples are in
`runs/<model>/figures/`. The dashboard in [`ui/`](../ui/) shows the same
results interactively and classifies an uploaded page live.

## The question that tests whether the result means anything

How do you know the test score is not inflated by leakage? The split is
grouped by source document, the overlap check runs before every evaluation,
and [`tests/test_leakage.py`](../tests/test_leakage.py) proves the check
rejects a split that shares a document.
