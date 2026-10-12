---
type: concept
title: "Random Forest"
libs:
  - scikit-learn
  - numpy
tags:
  - concept/ensemble
  - concept/supervised-learning
  - concept/bagging
related:
  - "[[Boosting]]"
  - "[[AdaBoost]]"
  - "[[Gradient Boosting]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn GridSearchCV]]"
  - "[[Sklearn Decision Tree Visualization]]"
created: 2026-05-23
updated: 2026-05-23
---

# Random Forest

> Parallel ensemble of decision trees where each tree is trained independently on a bootstrap row sample **and** — crucially — considers only a random subset of features at every node split. That per-node feature randomness is what separates Random Forest from plain bagging.

---

## The One Distinction That Matters: Bagging vs. Random Forest

Plain bagging and Random Forest both train independent trees on bootstrap samples. The difference is a single mechanism that happens *inside* each tree:

| | Row sampling | Feature selection at each node split |
|---|---|---|
| **Bagging** | bootstrap (with replacement) | all features considered |
| **Random Forest** | bootstrap (with replacement) | random `max_features` subset |

That's it. Random Forest = Bagging + per-node random feature subsampling.

Why does this matter so much? Because without feature subsampling, every tree's root split gravitates toward the same dominant feature. You end up with $M$ trees that are structurally similar — high pairwise correlation $\rho$. The ensemble variance floor is $\rho \sigma^2$ (see [[Boosting]] for the formula), so correlated trees give you diminishing returns no matter how many you add.

Per-node feature subsampling forces each tree to build its decision boundary through *different combinations* of features. Trees become genuinely diverse → lower $\rho$ → lower ensemble variance → better generalization.

## How a Single Tree Gets Built

For each of the $M$ trees:

1. **Row bootstrap**: draw $N$ samples with replacement from the $N$ training rows. ~63.2% unique rows, rest are duplicates. The ~36.8% left out are the "out-of-bag" (OOB) samples.
2. **Grow the tree**: at every node split:
   - Randomly draw `max_features` columns from the full $p$ features.
   - Evaluate the impurity criterion (Gini, entropy, MSE) across all thresholds on *only those selected features*.
   - Pick the best split. Move to the next node — **new random feature draw**.
3. Grow until stopping criteria (max_depth, min_samples_leaf, etc.).

Step 2 is where the magic happens. A new random feature subset at *every single node* means two trees can share the same root split feature by chance, but they'll almost certainly diverge deeper in the tree.

## Training AND Inference Are Both Parallel — The Clean Case

Random Forest is the only major ensemble method where **both training and inference are fully independent**. No asymmetry to reason about:

- **Training**: each tree is fit on its own bootstrap sample with its own random feature draws. Tree 47 knows nothing about tree 1. This is why `n_jobs=-1` gives near-linear speedup.
- **Inference**: each tree predicts the full target (a class label or regression value) independently. The ensemble combines them by majority vote (classification) or simple average (regression). No weights, no cumulative sums — every tree's vote counts equally.

This contrasts sharply with boosting methods:

| Method | Training | Inference | Each learner standalone? |
|---|---|---|---|
| **Random Forest** | parallel (independent) | parallel (equal votes, averaged) | yes — full prediction |
| [[AdaBoost]] | sequential (weight dependency) | parallel (weighted votes) | yes — but vote weighted by $\alpha_k$ |
| [[Gradient Boosting]] | sequential (residual dependency) | additive sum of corrections | no — each tree is a meaningless delta |

AdaBoost has an interesting asymmetry: sequential at training time, parallel at inference. Gradient boosting's trees are correction deltas that are meaningless without the running sum. Random Forest has no such subtlety — every tree is independent, always.

### Runnable: Every Tree Is a Standalone Classifier (Iris)

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    class_names = ["setosa", "versicolor", "virginica"]

    rf = RandomForestClassifier(n_estimators=7, random_state=0).fit(X, y)

    x_new = X[[75]]  # pick a versicolor sample

    # Every tree predicts independently — show them all.
    print("=== Each tree's independent prediction ===")
    votes: dict[int, int] = {0: 0, 1: 0, 2: 0}
    for i, tree in enumerate(rf.estimators_):
        pred = int(tree.predict(x_new)[0])
        votes[pred] += 1
        proba = tree.predict_proba(x_new)[0]
        print(
            f"  tree {i}: predicts {class_names[pred]:>10s}  "
            f"(proba: {proba})"
        )

    print(f"\nMajority vote tally:")
    for cls, count in votes.items():
        print(f"  {class_names[cls]:>10s}: {count}/{len(rf.estimators_)} votes")

    print(f"\nEnsemble prediction: {class_names[int(rf.predict(x_new)[0])]}")
    print(
        "\n^ Every tree is a real standalone classifier."
        "\n  No weights, no cumulative sums."
        "\n  Training was independent too — tree 0 never saw tree 6's output."
    )
```

## The `max_features` Default Tells You What Works

| Task | sklearn default | Typical range |
|---|---|---|
| Classification | `sqrt(n_features)` | `sqrt` to `log2` |
| Regression | `n_features` (all) | `n_features/3` to `n_features` |

Classification benefits more from aggressive subsampling because decision boundaries tend to be simpler — you'd rather have diverse trees than individually accurate ones. Regression defaults to all features because regression trees need more expressiveness to approximate continuous targets, but `n_features/3` is a well-known recommendation (Friedman's original suggestion).

## Install

```bash
uv add scikit-learn
```

## Runnable: Bagging vs. Random Forest — Seeing the Correlation Drop (Iris)

This is the experiment that makes the distinction concrete. We train bagged trees and Random Forest trees on Iris, then measure pairwise correlation between tree predictions.

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def tree_prediction_correlation(
    estimators: list,
    X: np.ndarray,
    n_classes: int,
) -> float:
    """Mean pairwise correlation of predicted class probabilities across trees."""
    # Flatten each tree's probability output into a 1-D vector per tree.
    prob_vectors = np.array([
        est.predict_proba(X).ravel() for est in estimators
    ])  # shape: (n_trees, n_samples * n_classes)
    corr = np.corrcoef(prob_vectors)
    n = corr.shape[0]
    return float((corr.sum() - np.trace(corr)) / (n * (n - 1)))


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    n_classes = len(np.unique(y))

    # Plain bagging: bootstrap rows, but each tree sees ALL features at every split.
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_features=1.0,  # all features available to each tree
        bootstrap=True,
        random_state=0,
    ).fit(X, y)

    # Random Forest: bootstrap rows AND random feature subset at each split.
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",  # sqrt(4) ≈ 2 features per split
        bootstrap=True,
        random_state=0,
    ).fit(X, y)

    rho_bag = tree_prediction_correlation(bag.estimators_, X, n_classes)
    rho_rf = tree_prediction_correlation(rf.estimators_, X, n_classes)

    print(f"Bagging       — mean tree correlation: {rho_bag:.3f}")
    print(f"Random Forest — mean tree correlation: {rho_rf:.3f}")
    print(f"Correlation drop: {rho_bag - rho_rf:.3f}")
    print(f"\nBagging test acc (OOB):  {bag.oob_score_:.3f}" if hasattr(bag, "oob_score_") else "")
    print(f"RF test acc (OOB):       {rf.oob_score:.3f}")
```

Expected: Random Forest trees have noticeably lower pairwise correlation than bagged trees, even on a 4-feature dataset like Iris. That correlation gap is the entire value proposition.

## Runnable: Visualizing What `max_features` Does to Split Diversity (Breast Cancer)

Shows how aggressive feature subsampling changes *which features* appear at root splits across the ensemble.

```python
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    for max_feat in (None, "sqrt", 2):
        rf = RandomForestClassifier(
            n_estimators=200, max_features=max_feat,
            random_state=0,
        ).fit(X, y)

        # What feature does each tree split on at the root?
        root_features: list[int] = [
            tree.tree_.feature[0] for tree in rf.estimators_
        ]
        counts = Counter(root_features)
        top3 = counts.most_common(3)

        label = str(max_feat) if max_feat is not None else "all (30)"
        print(f"max_features={label}:")
        for feat_idx, count in top3:
            print(f"  {feature_names[feat_idx]:>30s}: root split in {count}/200 trees")
        # How many distinct features ever appear as root?
        print(f"  distinct root features: {len(counts)}/30\n")
```

Expected: with `max_features=None` (all 30), one or two features dominate the root split across nearly all trees. With `sqrt` (~5), root splits are spread across many more features — that's the decorrelation at work.

## Runnable: OOB Score — Free Validation Without a Hold-Out Set

Each tree's ~36.8% out-of-bag rows give you a built-in validation estimate at no extra cost.

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)

    rf = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        oob_score=True, random_state=0,
    ).fit(X, y)

    # OOB score: each sample is predicted only by trees that did NOT see it.
    print(f"OOB accuracy:     {rf.oob_score_:.3f}")

    # Compare with 5-fold CV — should be close.
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
    print(f"5-fold CV:        {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"\nDifference:       {abs(rf.oob_score_ - cv_scores.mean()):.3f}")
```

Expected: OOB score and 5-fold CV accuracy are typically within 1–2% of each other. OOB is effectively leave-one-out at the tree level — a remarkably good estimate for free.

## Runnable: Random Forest vs. Gradient Boosting (California Housing)

The classic parallel-vs-sequential showdown on a regression problem.

```python
from __future__ import annotations

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    X, y = fetch_california_housing(return_X_y=True)

    rf = RandomForestRegressor(
        n_estimators=300, max_features="sqrt", random_state=0,
    )
    hgb = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05, random_state=0,
    )

    for name, model in [("Random Forest", rf), ("HistGradientBoosting", hgb)]:
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"{name:25s}: R² = {scores.mean():.3f} +/- {scores.std():.3f}")
```

Gradient boosting usually wins on accuracy, but Random Forest is simpler to tune and more robust out of the box — there's no learning rate or n_estimators interaction to worry about.

## Common Gotchas

- **`max_features` is not `max_depth`.** `max_features` controls how many features are *candidates* at each split — it doesn't limit the tree's depth or complexity. A Random Forest with `max_features="sqrt"` still grows deep trees by default.
- **Bootstrap ≠ subsampling.** Bootstrap samples *with replacement* to the same size $N$. Row subsampling in XGBoost samples *without replacement* to a fraction of $N$. Different mechanisms, different variance profiles.
- **More trees rarely hurts, just costs compute.** Unlike boosting where too many rounds can overfit, Random Forest generally plateaus — OOB error flattens, it doesn't go back up. The main cost is wall-clock time.
- **Feature importance from RF is biased toward high-cardinality features.** Continuous features and features with many unique values get more split opportunities. Use permutation importance (`sklearn.inspection.permutation_importance`) for unbiased estimates.
- **`n_jobs=-1` actually helps here.** Unlike boosting, trees are truly independent — parallel training gives near-linear speedup.

## Plays Well With

- [[Boosting]] — the sequential counterpart; useful conceptual contrast
- [[Gradient Boosting]] — typically higher accuracy, harder to tune
- [[Sklearn Cross-Validation]] — though OOB score often suffices
- [[Sklearn GridSearchCV]] — tuning max_features, n_estimators, max_depth
- [[Sklearn Decision Tree Visualization]] — inspect individual trees in the forest

## See Also

- Breiman (2001), *Random Forests*
- [sklearn RandomForestClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [sklearn ensemble user guide](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
