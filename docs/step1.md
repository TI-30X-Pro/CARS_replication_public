# Stage 1 — Select the adversary

## Components

`cars/selection/` contains:

- `features.py`: 13 features for every agent relative to the target;
- `model.py`: histogram-gradient-boosting training and a version-independent tree-array
  representation;
- `selector.py`: score-based adversary selection and the five-replan confirmation gate;
- `dataset.py`: labels, scene splitting, samples, and selection metrics.

Labels are stored in `configs/labels_nusc.json`. The released selector is retained in
`CARS_original/checkpoints/decision_tree_model.pkl` and is read without importing its historical
scikit-learn version.

## Train and evaluate

Build the dataset, fit a classifier with the installed scikit-learn, and evaluate both the fitted
and released classifiers with:

```bash
python -m cars.run.train_selector \
  --nusc "$NUSC" \
  --index "$WORK/nusc_index.npz" \
  --out "$WORK/selector"
```

The command writes `trees.npz`, `samples.npz`, and `metrics.json`.

## Replication result

The fitted classifier achieved adversary precision/recall/F1 of 0.75/0.79/0.77 and top-1/3/5
accuracy of 84.4/97.0/99.3 percent.

## Implementation notes

Implementation details that affect selection include using the target's last displacement for
its direction, computing offsets from the final history point, setting TTC to 999 when closing
speed is at most 0.1 m/s, and counting the confirmation threshold in re-plans rather than
individual simulation steps.

## Tests

Self-contained selector tests are in `tests/test_selection.py`.
