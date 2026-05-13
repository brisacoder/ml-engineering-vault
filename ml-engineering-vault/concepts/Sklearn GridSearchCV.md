---
type: concept
title: "Sklearn GridSearchCV"
libs:
  - scikit-learn
tags:
  - concept/optimization
  - task/model-selection
related:
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Metrics]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Sklearn Feature Scaling]]"
created: 2026-05-13
updated: 2026-05-13
---

# Sklearn GridSearchCV

> `GridSearchCV` and `RandomizedSearchCV` combine hyperparameter search with cross-validation and automatic refitting — the three things you'd otherwise wire together by hand. They answer: *"which hyperparameter configuration generalises best, and give me a production-ready estimator fitted on all the training data."*

---

## GridSearchCV — Exhaustive Search

### Intuition

Grid search evaluates **every** combination in the parameter grid via cross-validation, picks the combination with the best mean CV score, then refits that configuration on the full training set. The result (`best_estimator_`) is ready for prediction — no manual refit needed.

This is the key difference from plain [[Sklearn Cross-Validation|cross_validate]]: CV evaluates **one** configuration and returns fold estimators trained on partial data. `GridSearchCV` evaluates **many** configurations and — because `refit=True` is the default — automatically refits the winning configuration on **all** of `X_train`. That's why `best_estimator_` is production-ready out of the box.

### In Code

```python
"""
GridSearchCV: search hyperparameters, evaluate via CV, refit the best on full data.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
}
# 3 × 3 × 3 = 27 combinations × 5 folds = 135 fits

search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    refit=True,           # ← refit best params on ALL of X_train (default)
    n_jobs=-1,            # parallelise across cores
    return_train_score=True,
)
search.fit(X_train, y_train)

# ── Results ───────────────────────────────────────────────────────
print(f"Best params:     {search.best_params_}")
print(f"Best CV score:   {search.best_score_:.4f}")
print(f"Test accuracy:   {search.score(X_test, y_test):.4f}")

# best_estimator_ is fitted on 100% of X_train — ready for production
y_pred = search.best_estimator_.predict(X_test)
print(f"Predictions:     {y_pred[:10]}")
```

### Key Attributes After `.fit()`

| Attribute | Type | What it is |
|---|---|---|
| `best_params_` | `dict` | Hyperparameters of the best combination |
| `best_score_` | `float` | Mean CV score of the best combination |
| `best_estimator_` | estimator | Fitted on **all** of `X_train` with `best_params_` |
| `best_index_` | `int` | Index into `cv_results_` for the best combination |
| `cv_results_` | `dict` | Full table: params, mean/std scores, rank, timing per combination |

### Exploring cv_results_ as a DataFrame

```python
import pandas as pd

results_df = pd.DataFrame(search.cv_results_)

# Most useful columns
cols = [
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "mean_train_score",
    "mean_fit_time",
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_leaf",
]
print(results_df[cols].sort_values("rank_test_score").head(10).to_string(index=False))
```

This DataFrame is the full audit trail — you can sort, filter, and plot to understand the sensitivity of each hyperparameter.

---

## RandomizedSearchCV — When the Grid Is Too Large

### Intuition

With many hyperparameters or continuous ranges, exhaustive search is impractical. `RandomizedSearchCV` samples `n_iter` random combinations from distributions you specify, then follows the same CV + refit workflow. You trade completeness for speed.

### In Code

```python
"""
RandomizedSearchCV: sample from distributions instead of exhaustive grid.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

param_distributions = {
    "n_estimators": randint(50, 300),           # uniform integer [50, 300)
    "max_depth": [None, 5, 10, 20, 30],         # discrete choices
    "min_samples_leaf": randint(1, 10),          # uniform integer [1, 10)
    "max_features": uniform(0.1, 0.9),           # uniform float [0.1, 1.0)
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,            # number of random combinations to try
    cv=5,
    scoring="accuracy",
    refit=True,
    n_jobs=-1,
    random_state=42,
    return_train_score=True,
)
search.fit(X_train, y_train)

print(f"Best params:     {search.best_params_}")
print(f"Best CV score:   {search.best_score_:.4f}")
print(f"Test accuracy:   {search.score(X_test, y_test):.4f}")
```

### Grid vs Randomized — When to Use Which

| Criterion | `GridSearchCV` | `RandomizedSearchCV` |
|---|---|---|
| Parameter space | Small, discrete (< ~100 combos) | Large, continuous, or many params |
| Budget | You can afford exhaustive search | Fixed compute budget (`n_iter`) |
| Guarantees | Tests every combination | Probabilistic — may miss the optimum |
| Continuous params | Must discretise manually | Sample directly from distributions |

Rule of thumb from Bergstra & Bengio (2012): random search with 60 iterations has a 95% chance of finding a configuration within the top 5% of the search space — often matching or beating grid search at a fraction of the cost.

---

## refit With Multiple Metrics

### Intuition

When `scoring` is a list or dict, `refit` must know **which** metric to optimise. Pass the metric name as a string to `refit`, and that metric determines `best_params_` and the refit.

### In Code

```python
"""
Multiple metrics with refit on a specific one.
"""

from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10]}

search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
    refit="f1_macro",     # ← refit using the best f1_macro params
    n_jobs=-1,
)
search.fit(X_train, y_train)

# best_score_ and best_params_ are based on f1_macro
print(f"Best f1_macro (CV): {search.best_score_:.4f}")
print(f"Best params:        {search.best_params_}")

# All metrics are still available in cv_results_
import pandas as pd
df = pd.DataFrame(search.cv_results_)
print(df[["param_n_estimators", "param_max_depth",
          "mean_test_accuracy", "mean_test_f1_macro",
          "mean_test_roc_auc_ovr", "rank_test_f1_macro"]].to_string(index=False))
```

**Gotcha:** if you pass `scoring` as a list but leave `refit=True`, sklearn raises `ValueError` — it doesn't know which metric to use. You must pass `refit="metric_name"` or `refit=False`.

---

## Searching Over a Pipeline

### Intuition

In practice, you rarely search over just the estimator — you also want to tune preprocessing steps. Use `Pipeline` + double-underscore syntax to search across the full chain. This avoids [[Sklearn Cross-Validation#Common Misconceptions|data leakage]] from fitting scalers on test folds.

### In Code

```python
"""
GridSearchCV over a Pipeline: preprocessing + estimator together.
"""

from __future__ import annotations

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42)),
])

# Double-underscore notation: step_name__param_name
param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 10, 20],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best params:   {search.best_params_}")
print(f"Test accuracy: {search.score(X_test, y_test):.4f}")

# best_estimator_ is the full fitted Pipeline
print(f"Type: {type(search.best_estimator_)}")
# <class 'sklearn.pipeline.Pipeline'>
```

See [[Sklearn Pipelines and ColumnTransformer]] for building pipelines with heterogeneous column types.

---

## Common Misconceptions

- **"`best_estimator_` was trained on the best fold"** — No. It's trained on the **entire** `X_train` you passed to `.fit()`, using the hyperparameters that scored best during CV. This is the whole point of `refit=True`.
- **"I should also do `cross_validate` on `best_estimator_`"** — That's circular. `best_params_` was selected based on CV scores, so re-running CV with those exact params gives an optimistically biased estimate. Evaluate on a held-out test set that the search never saw.
- **"GridSearchCV is always better than RandomizedSearchCV"** — Only when the grid is small. Beyond ~100 combinations, random search is more efficient at exploring the space, especially when some hyperparameters matter much more than others (low effective dimensionality).
- **"I can skip `train_test_split` if I'm doing CV"** — No. CV selects hyperparameters — you need a **separate** test set to get an unbiased performance estimate after selection. Without it, your reported score is optimistic.

---

<!-- Add more hyperparameter search patterns below this line -->
