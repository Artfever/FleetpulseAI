# P16-S3 training replay labels

The RUL Prediction page uses the saved P06 development model. It does not load
a final refitted model. For an FD001/train engine, the page uses the saved
development split manifest to label the selected engine:

- A fitting engine is marked **In-sample illustration** because it was used
  during model fitting; its prediction is not held-out evidence.
- A validation engine is marked **Held-out development engine** because it was
  excluded from fitting. This is still development evidence, not official-test
  performance.
- An engine absent from either saved split group is marked as unknown; the page
  warns against treating its prediction as held-out evidence.
- The illustrative synthetic source is explicitly identified as synthetic,
  not an evaluation or maintenance result.

The classifier uses only dataset-qualified engine identities from the saved
split. It does not read validation target values. It rejects duplicate or
overlapping fitting/validation identities. If a final model trained on all
original training engines is added later, every FD001/train replay for that
model must be labeled in-sample; official test histories or a saved held-out
development bundle are needed for unseen examples.

In the current local split, engine 1 is in the fitting group, while engines 4,
5, and 12 (the packaged demo subset) are in the held-out validation group.
This describes membership for the saved P06 development model only; no final
refit or official-test inference was performed for this task.
