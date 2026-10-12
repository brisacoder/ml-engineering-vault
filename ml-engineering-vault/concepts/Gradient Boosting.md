---
type: concept
title: "Gradient Boosting"
libs:
  - scikit-learn
  - xgboost
  - numpy
tags:
  - concept/ensemble
  - concept/boosting
  - concept/supervised-learning
related:
  - "[[Gradient Boosting Step by Step]]"
  - "[[Boosting]]"
  - "[[AdaBoost]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn GridSearchCV]]"
  - "[[Ridge Regression]]"
created: 2026-05-23
updated: 2026-09-29
---

# Gradient Boosting

> Sequential boosting via **residual fitting**: each new tree predicts the negative gradient of the loss w.r.t. the current ensemble. Same rows, same features — the *target* changes every round.

---

## Core Mental Model

Build $F_M(x) = F_0(x) + \sum_{k=1}^{M} \nu \cdot f_k(x)$ where each $f_k$ regresses on the **negative gradient** of the loss at the current predictions:

$$r_i^{(k)} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{k-1}}$$

For squared error, $r_i = y_i - F_{k-1}(x_i)$ — literally the residuals. For other losses (log-loss, Huber, etc.) the residuals are loss-specific. $\nu$ is the learning rate (shrinkage); $F_0$ is typically the constant minimizer of the loss (mean for regression, log-odds for binary classification).

The mental shorthand: **gradient descent in function space**. Each tree is a step in the direction that reduces ensemble loss.

To see these symbols as actual printed trees and per-row arithmetic (tree 1, residuals, tree 2, ...), including why log-loss leaves are `sum(r)/sum(p(1-p))` and not mean residuals, see [[Gradient Boosting Step by Step]].

## Why the Sequential Dependency Is Unbreakable

Tree $k$ needs $F_{k-1}$ to compute its training target. You cannot compute round 5's residuals without first fitting trees 1–4 and evaluating them on every training row. **Subsampling does not change this** — XGBoost with `subsample=0.5` still samples rows *within* round $k$, after $F_{k-1}$ is fixed.

## Inference Is an Additive Sum, Not a Vote

Unlike [[AdaBoost]] where each stump is a standalone classifier that votes independently, gradient boosting trees predict **correction deltas** — tiny residual fragments that are meaningless in isolation. The inference formula is:

$$F(x) = F_0 + \nu \cdot f_1(x) + \nu \cdot f_2(x) + \ldots + \nu \cdot f_M(x)$$

Tree 5 might output $+0.03$ for a given input. That number has no interpretation without the cumulative sum of $F_0$ through $\nu \cdot f_4$. This is fundamentally different from AdaBoost, where stump 5 predicts "class 1" — a real class label with a real vote weight.

The practical consequence: in [[AdaBoost]] you can inspect what each stump predicts and it makes intuitive sense. In gradient boosting, inspecting individual trees tells you almost nothing — you need to track the cumulative prediction.

| | What each learner predicts | Learner output alone meaningful? |
|---|---|---|
| [[Random Forest]] | full target (class or value) | yes — it's a real prediction |
| [[AdaBoost]] | full class label | yes — weighted by $\alpha_k$ |
| **Gradient Boosting** | correction delta (residual) | no — meaningless without $F_0 + \sum$ previous |

### Runnable: Individual Trees Are Meaningless Residual Fragments (California Housing)

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    x_single = X[[100]]  # one house
    y_true = y[100]

    gb = GradientBoostingRegressor(
        n_estimators=10, max_depth=3, learning_rate=0.1, random_state=0,
    ).fit(X, y)

    # F_0 = training mean (the initial constant prediction).
    f0 = float(gb.init_.predict(x_single)[0])
    print(f"true value: {y_true:.4f}")
    print(f"F_0 (training mean): {f0:.4f}\n")

    # Show each tree's raw output and the running sum.
    cumulative = f0
    print("tree | raw delta  | cumulative prediction")
    print("-----|------------|---------------------")
    for i, tree in enumerate(gb.estimators_.ravel()):
        delta = float(gb.learning_rate * tree.predict(x_single)[0])
        cumulative += delta
        print(f"  {i:2d}  | {delta:+10.4f} | {cumulative:.4f}")

    print(f"\nfinal ensemble prediction: {gb.predict(x_single)[0]:.4f}")
    print(
        "\n^ Each delta is a tiny correction."
        "\n  Tree 5's output of e.g. +0.02 is a meaningless fragment"
        "\n  without the running sum from F_0 through tree 4."
        "\n  Contrast with AdaBoost, where each stump predicts a real class."
    )
```

## The Subsampling Knobs (XGBoost)

Three orthogonal axes, all default to no subsampling:

| Parameter | Granularity | Effect |
|---|---|---|
| `subsample` | per tree (rows) | each tree fits on a random row subset |
| `colsample_bytree` | per tree (cols) | each tree sees only a column subset |
| `colsample_bylevel` | per depth level (cols) | re-sample columns at each level within a tree |
| `colsample_bynode` | per split (cols) | re-sample columns at every single split |

The column-sampling levels **compose multiplicatively**: `colsample_bytree=0.8` and `colsample_bylevel=0.8` means each level sees $0.64$ of features.

`colsample_bynode` is the most aggressive — that's essentially what Random Forest does at every split, applied inside boosting.

## Why Feature Subsampling Helps (the part the textbook glossed over)

Without it, every tree's root split is on the same dominant feature, and trees become correlated. Correlated learners don't reduce variance when combined — see the variance formula in [[Boosting]]. Feature subsampling forces trees to discover patterns through *combinations* of features rather than leaning on the one or two columns that win every split-finding contest.

The other side: if you only have a handful of informative features and you set `colsample_bytree=0.1`, most trees will see zero useful features and contribute noise. Subsampling has a sweet spot that depends on feature redundancy.

## Install

```bash
uv add scikit-learn xgboost
```

## Runnable: Watching Residuals Shrink (California Housing)

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor


def gradient_boost_squared_error(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: int = 20,
    learning_rate: float = 0.1,
    max_depth: int = 3,
) -> tuple[list[DecisionTreeRegressor], list[float]]:
    """Return the trees and the train RMSE after each round."""
    f_current = np.full_like(y, fill_value=y.mean(), dtype=float)
    trees: list[DecisionTreeRegressor] = []
    rmse_history: list[float] = []

    for k in range(n_rounds):
        # For squared error, negative gradient = residual.
        residuals = y - f_current
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=k)
        tree.fit(X, residuals)
        f_current = f_current + learning_rate * tree.predict(X)
        trees.append(tree)
        rmse_history.append(float(np.sqrt(np.mean((y - f_current) ** 2))))

    return trees, rmse_history


if __name__ == "__main__":
    data = fetch_california_housing()
    X, y = data.data, data.target
    _, history = gradient_boost_squared_error(X, y, n_rounds=20)
    for k, rmse in enumerate(history):
        print(f"round {k:2d}: train RMSE = {rmse:.3f}")
```

Expected: RMSE shrinks rapidly at first, then plateaus. That plateau is the bias-variance regime where more rounds start overfitting — when you'd want to stop early or turn up regularization.

## Runnable: The Three Subsampling Knobs (California Housing)

Demonstrates how `subsample`, `colsample_bytree`, and `colsample_bynode` each affect generalization on a real problem with 8 features.

```python
from __future__ import annotations

import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    configs: list[dict[str, float]] = [
        # Baseline: no subsampling.
        {"subsample": 1.0, "colsample_bytree": 1.0, "colsample_bynode": 1.0},
        # Row subsampling only.
        {"subsample": 0.7, "colsample_bytree": 1.0, "colsample_bynode": 1.0},
        # Per-tree column subsampling.
        {"subsample": 1.0, "colsample_bytree": 0.5, "colsample_bynode": 1.0},
        # Per-split column subsampling (Random Forest flavor).
        {"subsample": 1.0, "colsample_bytree": 1.0, "colsample_bynode": 0.5},
        # All three combined.
        {"subsample": 0.7, "colsample_bytree": 0.7, "colsample_bynode": 0.7},
    ]

    for cfg in configs:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=0,
            **cfg,
        )
        model.fit(X_tr, y_tr)
        train_r2 = model.score(X_tr, y_tr)
        test_r2 = model.score(X_te, y_te)
        gap = train_r2 - test_r2
        print(
            f"{cfg}\n  train R²={train_r2:.3f}  test R²={test_r2:.3f}  gap={gap:.3f}\n"
        )
```

Expected: the baseline has the largest train-test gap (overfitting); each subsampling knob narrows it. With only 8 features, column subsampling is more aggressive here than on a wide dataset — watch how `colsample_bynode=0.5` behaves.

## Runnable: Gradient Boosting for Classification (Breast Cancer)

The same algorithm works for classification — the only difference is the loss function (log-loss instead of squared error) which changes what the "residuals" mean.

```python
from __future__ import annotations

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)

    # Classic GradientBoosting — supports subsample and max_features.
    gbc = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=0,
    )
    # Hist variant — faster, uses histogram-based splits (like LightGBM).
    hgbc = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, learning_rate=0.1, random_state=0,
    )

    for name, model in [("GradientBoosting", gbc), ("HistGradientBoosting", hgbc)]:
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        print(f"{name:25s}: {scores.mean():.3f} +/- {scores.std():.3f}")
```

## Library Defaults Worth Knowing

| Library | Row subsample | Col subsample |
|---|---|---|
| sklearn `GradientBoostingRegressor` | `subsample=1.0` | `max_features=None` (per-split, like Random Forest if set) |
| sklearn `HistGradientBoostingRegressor` | no row subsample param | no column subsample param |
| XGBoost | `subsample=1.0` | all `colsample_*` = 1.0 |
| LightGBM | `bagging_fraction=1.0`, `bagging_freq=0` | `feature_fraction=1.0` |
| CatBoost | `subsample=0.8` (with `bootstrap_type="Bernoulli"`) | `rsm=1.0` |

CatBoost is the outlier — it subsamples rows by default.

## Common Gotchas

- **Learning rate × n_estimators trade-off.** Lower learning rate needs more rounds. Always tune together, not independently.
- **`HistGradientBoostingRegressor` has no subsampling parameters.** It uses other regularizers (L2, max_leaf_nodes, early stopping). If you need column subsampling, use XGBoost or LightGBM.
- **Feature subsampling backfires with few informative features.** If only 3 of 100 columns carry signal, `colsample_bytree=0.1` gives most trees zero useful features.
- **Don't compose `colsample_*` blindly.** They multiply. `colsample_bytree=0.5 * colsample_bylevel=0.5 * colsample_bynode=0.5` means each split sees 12.5% of features.
- **Early stopping is the right regularizer to reach for first.** Use `early_stopping_rounds` with a validation set before tuning subsampling.

## Plays Well With

- [[Gradient Boosting Step by Step]] — the same algorithm traced tree by tree on Penguins, with a from-scratch implementation that matches sklearn
- [[Boosting]] — umbrella concept; explains the sequential-vs-parallel distinction
- [[AdaBoost]] — the row-reweighting cousin; equivalent to gradient boosting with exponential loss
- [[Sklearn GridSearchCV]] — tuning learning_rate × n_estimators together
- [[Sklearn Cross-Validation]] — proper evaluation of boosted models

## See Also

- Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*
- Friedman (1999), *Stochastic Gradient Boosting* (introduced row subsampling)
- Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*
- [XGBoost parameter docs](https://xgboost.readthedocs.io/en/stable/parameter.html)
