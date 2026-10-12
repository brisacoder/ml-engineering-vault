---
type: concept
title: "Boosting (Overview)"
libs:
  - scikit-learn
  - xgboost
  - numpy
tags:
  - concept/ensemble
  - concept/supervised-learning
related:
  - "[[AdaBoost]]"
  - "[[Gradient Boosting]]"
  - "[[Gradient Boosting Step by Step]]"
  - "[[Random Forest]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn GridSearchCV]]"
created: 2026-05-23
updated: 2026-09-29
---

# Boosting (Overview)

> Sequential ensembling where each learner is trained to fix what the previous ensemble got wrong. The data doesn't change between rounds — the *objective* does.

---

## Core Mental Model

Boosting builds an additive model $F_M(x) = \sum_{k=1}^{M} \alpha_k f_k(x)$ where each $f_k$ is fit **after** $f_{k-1}$, on a training signal derived from the current ensemble's mistakes.

Two boosting families differ in *how* the training signal changes between rounds:

- [[AdaBoost]] — reweights *rows*. Misclassified rows get higher sample weights; the next learner minimizes weighted error.
- [[Gradient Boosting]] — refits to *residuals* (negative gradients of the loss). The target $y$ each tree sees shrinks as the ensemble improves.

Both are sequential. Both use all rows and all features by default. Subsampling is an orthogonal regularization knob, not part of the core algorithm.

## Two Orthogonal Axes (the thing that confused me)

| | Full data | Subsampled |
|---|---|---|
| **Sequential** | classical AdaBoost, classical GBM | stochastic GBM, XGBoost defaults |
| **Parallel** | (empty — would give identical learners) | bagging, Random Forest |

Sequential vs. parallel is about **dependencies between learners**, not about subsampling. Subsampling does not make boosting parallel — XGBoost with `subsample=0.8` still trains tree $k+1$ only after tree $k$ is done, because the gradients depend on the current ensemble's prediction.

## Training vs. Inference: The Asymmetry Nobody Explains

The sequential dependency in boosting exists **only during training**. At inference time, the trained learners combine through a fixed rule that doesn't involve the sequential machinery at all. This asymmetry is non-obvious and critical to understanding what "ensemble" actually means for each method:

| Method | Training | Inference |
|---|---|---|
| [[Random Forest]] | parallel (independent trees) | parallel (independent votes, averaged) |
| [[AdaBoost]] | sequential (weights depend on previous errors) | parallel (independent weighted votes) |
| [[Gradient Boosting]] | sequential (residuals depend on previous sum) | additive sum (each tree contributes a correction delta) |

The key insight: **AdaBoost's stumps are standalone classifiers at inference.** Each one predicts a class label and votes with weight $\alpha_k$ — you could evaluate them in any order. The sequential training dependency vanishes once the $\alpha_k$ values are computed.

**Gradient boosting is different.** Each tree predicts a tiny residual correction, not a class or target value. Tree 5 by itself outputs something meaningless — it only makes sense as part of the running sum $F_0 + \nu f_1 + \ldots + \nu f_5$. The combination is additive, not a vote.

**Random Forest** is the cleanest: both training and inference are fully parallel and independent. No asymmetry to reason about.

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    x_single = X[[0]]  # one sample to trace through

    # --- Random Forest: each tree predicts a full class independently ---
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    print("=== Random Forest: independent votes ===")
    for i, tree in enumerate(rf.estimators_):
        print(f"  tree {i}: predicts class {tree.predict(x_single)[0]}")
    print(f"  ensemble vote: {rf.predict(x_single)[0]}\n")

    # --- AdaBoost: each stump predicts a class, weighted by alpha_k ---
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=5, algorithm="SAMME", random_state=0,
    ).fit(X, y)
    print("=== AdaBoost: independent weighted votes ===")
    for i, (stump, alpha) in enumerate(
        zip(ada.estimators_, ada.estimator_weights_)
    ):
        pred = stump.predict(x_single)[0]
        print(f"  stump {i}: predicts class {pred}, alpha={alpha:.3f}")
    print(f"  ensemble vote: {ada.predict(x_single)[0]}\n")

    # --- Gradient Boosting: each tree predicts a correction (meaningless alone) ---
    gb = GradientBoostingClassifier(
        n_estimators=5, max_depth=2, random_state=0,
    ).fit(X, y)
    print("=== Gradient Boosting: additive corrections ===")
    # staged_decision_function gives cumulative sum after each round.
    cumulative = list(gb.staged_decision_function(x_single))
    prev = 0.0
    for i, cum in enumerate(cumulative):
        delta = float(cum[0]) - prev
        prev = float(cum[0])
        print(f"  tree {i}: delta={delta:+.4f}, cumulative={float(cum[0]):+.4f}")
    print(f"  ensemble prediction: {gb.predict(x_single)[0]}")
    print(
        "\n  ^ Notice: each tree's delta is a tiny correction."
        "\n    Tree 3 alone predicting +0.02 is meaningless without trees 0-2."
    )
```

## Why Subsampling Helps an Ensemble at All

Variance of an averaged ensemble of $M$ learners with pairwise correlation $\rho$ and per-learner variance $\sigma^2$:

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{M} \sigma^2$$

As $M \to \infty$ the second term vanishes but the first doesn't. **The floor is set by learner correlation.** Subsampling rows (bagging) or columns (feature subsampling) lowers $\rho$ by forcing learners to be genuinely different.

- **Row subsampling** says: "don't memorize specific training points."
- **Feature subsampling** says: "don't lean too hard on any one feature."

In boosting, where each round naturally piles onto the same dominant features, **feature subsampling is the more interesting regularizer**.

## Runnable: Sequential Dependency on California Housing

Demonstrates that tree $k$'s training target depends on tree $k-1$'s output — you literally cannot parallelize without breaking the algorithm.

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor


def hand_rolled_gradient_boost(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: int = 5,
    learning_rate: float = 0.3,
    max_depth: int = 2,
) -> list[DecisionTreeRegressor]:
    """Squared-error gradient boosting from scratch. Returns the list of trees."""
    trees: list[DecisionTreeRegressor] = []
    # Initial prediction is the mean (the constant minimizer of squared error).
    f_current = np.full_like(y, fill_value=y.mean(), dtype=float)

    for k in range(n_rounds):
        # The "training signal" for tree k is the residual against the *current* ensemble.
        # This is the sequential dependency: residuals_k requires f_{k-1}.
        residuals = y - f_current
        print(
            f"round {k}: residual std={residuals.std():.3f}, "
            f"residual mean abs={np.abs(residuals).mean():.3f}"
        )

        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=k)
        tree.fit(X, residuals)
        trees.append(tree)

        # Update the ensemble's predictions.
        f_current = f_current + learning_rate * tree.predict(X)

    return trees


if __name__ == "__main__":
    data = fetch_california_housing()
    X, y = data.data, data.target
    _ = hand_rolled_gradient_boost(X, y, n_rounds=8)
```

Expected output: residual std shrinks monotonically. Round $k$ literally cannot be computed until round $k-1$'s tree has produced its predictions on every training row.

## Runnable: Learner Correlation Drops with Feature Subsampling

Shows that `colsample_bytree < 1.0` produces less correlated trees, which is the mechanism by which subsampling regularizes. Uses the Diabetes dataset (10 features, all numeric — a clean testbed for correlation analysis).

```python
from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_diabetes


def mean_pairwise_correlation(predictions: np.ndarray) -> float:
    """predictions: shape (n_trees, n_samples). Returns mean off-diagonal correlation."""
    corr = np.corrcoef(predictions)
    n = corr.shape[0]
    # Mean of off-diagonal entries.
    return float((corr.sum() - np.trace(corr)) / (n * (n - 1)))


def per_tree_predictions(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Get per-tree predictions by predicting with iteration_range=(k, k+1)."""
    n_trees: int = model.get_booster().num_boosted_rounds()
    preds = np.stack(
        [model.predict(X, iteration_range=(k, k + 1)) for k in range(n_trees)]
    )
    return preds  # shape: (n_trees, n_samples)


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)

    for colsample in (1.0, 0.5, 0.25):
        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            colsample_bytree=colsample,
            random_state=0,
        )
        model.fit(X, y)
        rho = mean_pairwise_correlation(per_tree_predictions(model, X))
        print(f"colsample_bytree={colsample}: mean pairwise tree corr = {rho:.3f}")
```

Expected: as `colsample_bytree` drops, mean pairwise correlation drops. That's the regularization mechanism made visible.

## Common Confusions

- **"Weak learner" ≠ bad learner.** It means *only slightly better than random*. Boosting theory shows you can combine many weak learners into an arbitrarily strong one — that's the whole point. Strong base learners (deep trees, SVMs with RBF) often break the boosting dynamic.
- **Subsampling does not make boosting parallel.** It introduces randomness *within* each sequential step.
- **Bottom-left of the 2×2 grid is empty.** Parallel + full data = identical learners = no ensemble benefit. Parallel ensembles *need* randomness; sequential ensembles get diversity for free from the changing objective.
- **AdaBoost ≡ gradient boosting with exponential loss.** The weight updates fall out as the gradient. Useful unification, rarely useful in practice.

## Plays Well With

- [[AdaBoost]] — the row-reweighting variant
- [[Gradient Boosting]] — the residual-fitting variant
- [[Random Forest]] — the parallel counterpart; useful contrast
- [[Ridge Regression]] — regularization concept contrast (L2 penalty vs. ensemble regularization)

## See Also

- Friedman (1999), *Stochastic Gradient Boosting*
- Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*
- Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*
