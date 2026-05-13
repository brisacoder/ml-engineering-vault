---
type: concept
title: "Sklearn Cross-Validation"
libs:
  - scikit-learn
tags:
  - concept/stats
  - task/model-selection
related:
  - "[[Sklearn Metrics]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Sklearn Decision Function and Thresholds]]"
  - "[[Sklearn GridSearchCV]]"
created: 2026-04-23
updated: 2026-05-13
---

# Sklearn Cross-Validation

> Scikit-learn provides two main cross-validation helpers: `cross_val_score` for quick single-metric evaluation, and `cross_validate` for anything more involved — multiple metrics, timing, train scores, fitted estimators, and fold indices. Under the hood, `cross_val_score` is just a thin wrapper that calls `cross_validate` and extracts one score array.

---

## cross_val_score — The Simple Case

> Returns a single array of scores (one per fold) for one metric. Use it when you just need a quick sanity check on model performance.

```python
"""
cross_val_score: one metric, one array.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

scores = cross_val_score(estimator=clf, X=X, y=y, cv=5, scoring="accuracy")

print(f"Scores per fold: {scores}")
print(f"Mean accuracy:   {scores.mean():.4f} ± {scores.std():.4f}")
# scores.shape == (5,) — that's all you get
```

### Signature Highlights

```python
cross_val_score(
    estimator,           # unfitted estimator
    X, y=None,
    scoring=None,        # single string like "accuracy", "f1", "roc_auc"
    cv=None,             # int, cross-validation generator, or iterable
    n_jobs=None,         # parallelism
    verbose=0,
    fit_params=None,
    error_score=np.nan,
)
# Returns: np.ndarray of shape (n_splits,)
```

Key limitation: `scoring` must be a **single** string or callable. Pass a list and it raises.

---

## cross_validate — The Full Toolbox

> Returns a dict with test scores, timing info, and optionally train scores, fitted estimators, and fold indices. This is what you reach for in any real evaluation workflow.

```python
"""
cross_validate: multiple metrics, timing, train scores,
fitted estimators, and fold indices — all in one call.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

X, y = load_digits(return_X_y=True)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

results = cross_validate(
    estimator=clf,
    X=X,
    y=y,
    cv=5,
    scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
    return_train_score=True,
    return_estimator=True,
    return_indices=True,
)

# ── What you get back ──────────────────────────────────────────────
print("Keys:", list(results.keys()))
# ['test_accuracy', 'test_f1_macro', 'test_roc_auc_ovr',
#  'train_accuracy', 'train_f1_macro', 'train_roc_auc_ovr',
#  'fit_time', 'score_time',
#  'estimator',
#  'indices']

# ── Multiple metrics at once ───────────────────────────────────────
for metric in ["accuracy", "f1_macro", "roc_auc_ovr"]:
    test = results[f"test_{metric}"]
    train = results[f"train_{metric}"]
    print(f"{metric:>12s}  train={train.mean():.4f}  test={test.mean():.4f}  "
          f"gap={train.mean() - test.mean():.4f}")

# ── Timing ─────────────────────────────────────────────────────────
print(f"\nMean fit time:   {np.mean(results['fit_time']):.3f}s")
print(f"Mean score time: {np.mean(results['score_time']):.3f}s")

# ── Fitted estimators (one per fold) ──────────────────────────────
best_fold = np.argmax(results["test_accuracy"])
best_model = results["estimator"][best_fold]
print(f"\nBest fold: {best_fold}, accuracy: {results['test_accuracy'][best_fold]:.4f}")
print(f"Best model feature importances shape: {best_model.feature_importances_.shape}")

# ── Fold indices ───────────────────────────────────────────────────
fold_0_train = results["indices"]["train"][0]
fold_0_test = results["indices"]["test"][0]
print(f"\nFold 0: {len(fold_0_train)} train samples, {len(fold_0_test)} test samples")
```

---

## Side-by-Side Comparison

> The capabilities that `cross_validate` adds beyond `cross_val_score`.

| Feature | `cross_val_score` | `cross_validate` |
|---|---|---|
| Return type | `np.ndarray` | `dict` |
| Multiple metrics | No (single string only) | Yes (list or dict of scorers) |
| `fit_time` / `score_time` | No | Always included |
| `return_train_score` | No | Yes (opt-in) |
| `return_estimator` | No | Yes (opt-in) |
| `return_indices` | No | Yes (opt-in) |
| Performance | Identical — calls `cross_validate` internally | — |

---

## Diagnosing Overfitting With Train vs Test Scores

> `return_train_score=True` is the fastest way to spot whether your model is memorizing the training data.

```python
"""
Comparing train vs test scores to diagnose bias/variance.
Large gap = overfitting. Both low = underfitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

X, y = load_digits(return_X_y=True)

models = {
    "DecisionTree (no limit)": DecisionTreeClassifier(random_state=42),
    "RandomForest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
}

for name, model in models.items():
    cv = cross_validate(
        model, X, y, cv=5,
        scoring="accuracy",
        return_train_score=True,
    )
    train_mean = cv["train_score"].mean()
    test_mean = cv["test_score"].mean()
    gap = train_mean - test_mean

    status = "⚠️ OVERFITTING" if gap > 0.05 else "✓ OK"
    print(f"{name:>30s}  train={train_mean:.4f}  test={test_mean:.4f}  "
          f"gap={gap:.4f}  {status}")
```

```
Expected output:

  DecisionTree (no limit)  train=1.0000  test=~0.85   gap=~0.15  ⚠️ OVERFITTING
  RandomForest (100 trees) train=1.0000  test=~0.97   gap=~0.03  ✓ OK
         LogisticRegression train=~0.99   test=~0.96   gap=~0.03  ✓ OK
```

---

## Using the scoring Dict for Named Metrics

> When you pass a dict instead of a list, you control the key names in the output — useful when you have custom scorers or want cleaner column names.

```python
from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_validate

X, y = load_digits(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Dict keys become the result keys (prefixed with test_ / train_)
scoring = {
    "acc": "accuracy",
    "f1_weighted": "f1_weighted",
    "f1_per_class": make_scorer(f1_score, average="macro"),
}

results = cross_validate(clf, X, y, cv=5, scoring=scoring)

# Access via your custom names
print(f"test_acc:          {results['test_acc'].mean():.4f}")
print(f"test_f1_weighted:  {results['test_f1_weighted'].mean():.4f}")
print(f"test_f1_per_class: {results['test_f1_per_class'].mean():.4f}")
```

---

## After Cross-Validation — Refit on Full Data

> `cross_validate` answers the question *"how well does this model generalise?"* — it does **not** produce a production model. Each fold estimator saw only ~80% of the data. The fold with the highest score didn't learn better — it got an easier test split. Cherry-picking it is overfitting to the evaluation procedure.

The correct two-step workflow:

1. **Evaluate** with `cross_validate` — check metrics, train/test gap, timing.
2. **Refit** the same estimator on the **entire** training set — this is what you deploy.

```python
"""
Correct workflow: evaluate with CV, then refit on full data.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

X, y = load_digits(return_X_y=True)

# Hold out a true test set that CV never touches
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

# ── Step 1: Evaluate via cross-validation ─────────────────────────
cv_results = cross_validate(
    clf, X_train, y_train,
    cv=5,
    scoring=["accuracy", "f1_macro"],
    return_train_score=True,
)

for metric in ["accuracy", "f1_macro"]:
    train = cv_results[f"train_{metric}"].mean()
    test = cv_results[f"test_{metric}"].mean()
    print(f"{metric:>10s}  train={train:.4f}  test={test:.4f}  gap={train - test:.4f}")

# ── Step 2: Satisfied? Refit on ALL training data ─────────────────
# clf is still unfitted — cross_validate clones internally
clf.fit(X_train, y_train)

# THIS is the model you use for predictions
y_pred = clf.predict(X_test)
print(f"\nHeld-out test accuracy: {np.mean(y_pred == y_test):.4f}")
```

### Why Not Pick the Best Fold Estimator?

```python
# ❌ WRONG: cherry-picking the "best" fold
best_fold = np.argmax(cv_results["test_accuracy"])
best_model = cv_results["estimator"][best_fold]  # trained on ~80% of X_train
y_pred = best_model.predict(X_test)               # using a partial model

# ✅ RIGHT: refit on full training set
clf.fit(X_train, y_train)                          # trained on 100% of X_train
y_pred = clf.predict(X_test)                       # using all available data
```

Three reasons the fold estimator is the wrong choice:

- **Less data** — it only saw 4/5 of `X_train`. More data → better model, always (assuming the model isn't already saturated).
- **Selection bias** — picking the highest-scoring fold is optimising on evaluation variance, not model quality. With 5 folds you're effectively selecting the max of 5 correlated random variables.
- **Non-reproducible** — the "best fold" changes with different random seeds or CV splits. Your production model shouldn't depend on which 20% happened to be held out.

### When return_estimator IS Useful

The fold estimators aren't useless — they're just not for prediction. Legitimate uses:

- **Inspecting per-fold variation**: compare `feature_importances_` across folds to check stability.
- **Ensemble trick** (rare): average predictions from all fold estimators as a poor-man's bagging — but if you want bagging, just use `BaggingClassifier`.
- **Debugging**: if one fold scores much worse, inspect that fold's estimator and its training indices to understand why.

### If You Want Hyperparameter Search + Automatic Refit

`cross_validate` evaluates a **single** configuration. If you want to search over hyperparameters and automatically refit the best configuration on full data, use [[Sklearn GridSearchCV]] or `RandomizedSearchCV` — they do CV + refit in one shot:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    refit=True,          # ← automatically refits best params on full X_train
)
search.fit(X_train, y_train)

# search.best_estimator_ is already fitted on ALL of X_train
y_pred = search.best_estimator_.predict(X_test)
print(f"Best params: {search.best_params_}")
print(f"Test accuracy: {np.mean(y_pred == y_test):.4f}")
```

---

## Common Misconceptions

- **"They train different models"** — No. `cross_val_score` calls `cross_validate` internally. Same folds, same fits, same results. The only difference is what gets returned.
- **"I need cross_val_score for speed"** — No performance difference. `cross_validate` does the same work; it just keeps more of the results instead of discarding them.
- **"return_estimator gives me a single best model"** — It gives you a **list** of `n_splits` fitted models, one per fold. Each was trained on a different subset. If you want a final model, refit on the full training set after CV.
