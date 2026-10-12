---
type: concept
title: "AdaBoost"
libs:
  - scikit-learn
tags:
  - concept/ensemble
  - concept/boosting
  - concept/classification
related:
  - "[[Boosting]]"
  - "[[Gradient Boosting]]"
  - "[[Sklearn Decision Tree Visualization]]"
  - "[[Sklearn Cross-Validation]]"
created: 2026-05-23
updated: 2026-05-23
---

# AdaBoost

> Sequential boosting via **sample reweighting**: misclassified rows get higher weights each round, so the next weak learner focuses on them. Each learner sees all rows, all features — only the weights change.

---

## Core Mental Model

Every learner is trained on the *same* $(X, y)$ but with a different **weight per row**. The split-finding algorithm inside the weak learner is minimizing *weighted* error, so what looks like an attractive split changes between rounds.

Algorithm sketch (SAMME, what sklearn implements since 1.6):

1. Initialize $w_i = 1/N$.
2. For $k = 1, \ldots, M$:
   - Fit weak learner $f_k$ minimizing weighted error $\epsilon_k = \sum_i w_i \mathbb{1}[y_i \ne f_k(x_i)]$.
   - Compute learner's voting weight: $\alpha_k = \eta \cdot \log\big((1 - \epsilon_k) / \epsilon_k\big) + \log(K - 1)$, where $\eta$ is the learning rate and $K$ is the number of classes.
   - Update sample weights: $w_i \leftarrow w_i \cdot \exp\big(\alpha_k \cdot \mathbb{1}[y_i \ne f_k(x_i)]\big)$, then renormalize.
3. Final prediction: $\arg\max_c \sum_k \alpha_k \mathbb{1}[f_k(x) = c]$.

## What "n_estimators" Actually Means

`n_estimators=M` = $M$ **sequential** rounds of training, producing $M$ weak learners that all participate in the final weighted vote. The learners are created in order (each depends on the previous round's errors), but at inference all $M$ vote simultaneously.

## Training Is Sequential, Inference Is a Parallel Vote

This is easy to miss: even though stump $k$ is trained using weights derived from stumps $1 \ldots k-1$, **at inference every stump is an independent classifier**. The final prediction is just:

$$\hat{y} = \arg\max_c \sum_{k=1}^{M} \alpha_k \cdot \mathbb{1}[f_k(x) = c]$$

Each stump predicts a class label. Each vote is weighted by $\alpha_k$. There is no sequential dependency — you could evaluate them in any order, in parallel, and sum the weighted votes. The sequential training dependency vanishes once the model is fitted.

This is fundamentally different from [[Gradient Boosting]], where each tree outputs a tiny residual correction that is meaningless without the cumulative sum from all previous trees. In AdaBoost, stump 3 predicts "class 1" with weight 0.74 — that's a real standalone prediction with a real weight. In gradient boosting, tree 3 predicts "+0.03" — that's just a delta.

### Runnable: Every Stump Votes Independently (Breast Cancer)

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    class_names = ["malignant", "benign"]

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10, algorithm="SAMME", random_state=0,
    ).fit(X, y)

    x_new = X[[42]]  # pick one sample

    # Each stump votes independently — show them all.
    print("=== Each stump's independent vote ===")
    vote_tally: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (stump, alpha) in enumerate(
        zip(ada.estimators_, ada.estimator_weights_)
    ):
        pred = int(stump.predict(x_new)[0])
        vote_tally[pred] += alpha
        print(
            f"  stump {i:2d}: predicts {class_names[pred]:>9s}, "
            f"alpha={alpha:.3f}"
        )

    print(f"\nWeighted vote tally:")
    for cls, weight in vote_tally.items():
        print(f"  {class_names[cls]:>9s}: {weight:.3f}")
    print(f"\nEnsemble prediction: {class_names[int(ada.predict(x_new)[0])]}")
    print(
        "\n^ Every stump is a standalone classifier."
        "\n  The sequential training dependency is gone at inference."
    )
```

## The Weak Learner Choice Matters a Lot

AdaBoost's math assumes each learner achieves weighted error in roughly $(0, 0.5)$:

- If $\epsilon_k = 0$ exactly → $\alpha_k \to \infty$ → ensemble collapses to that single learner.
- If $\epsilon_k \geq 0.5$ → SAMME refuses the learner and stops (or in older AdaBoost.M1, terminates).

So **strong base learners break AdaBoost**. An RBF-kernel SVM will often achieve near-zero weighted training error and blow up $\alpha$. The canonical choice is a `DecisionTreeClassifier(max_depth=1)` — a stump.

## Install

```bash
uv add scikit-learn
```

## Runnable: Watching Sample Weights Evolve (Breast Cancer)

The clearest way to internalize AdaBoost is to *see* the weights migrate toward hard examples across rounds. Uses the Breast Cancer Wisconsin dataset (569 samples, 30 features, binary classification).

```python
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def trace_sample_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 5,
    learning_rate: float = 1.0,
) -> np.ndarray:
    """Reimplement SAMME's weight updates so we can inspect them per round.

    Returns array of shape (n_estimators + 1, n_samples) with weights at each round
    (row 0 is uniform initial weights).
    """
    n: int = len(y)
    n_classes: int = len(np.unique(y))
    weights = np.full(n, fill_value=1.0 / n)
    history = [weights.copy()]

    for _ in range(n_estimators):
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X, y, sample_weight=weights)
        pred = stump.predict(X)
        miss = (pred != y).astype(float)

        # Weighted error and SAMME's alpha.
        eps = float(np.dot(weights, miss))
        # Clip to avoid log(0) when a stump perfectly fits weighted data.
        eps = np.clip(eps, 1e-10, 1 - 1e-10)
        alpha = learning_rate * (np.log((1 - eps) / eps) + np.log(n_classes - 1))

        # Update: multiplicative bump for misclassified rows, renormalize.
        weights = weights * np.exp(alpha * miss)
        weights /= weights.sum()
        history.append(weights.copy())

    return np.array(history)


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    history = trace_sample_weights(X, y, n_estimators=5)

    # Show how much weight mass concentrated on the hardest rows over rounds.
    for r, w in enumerate(history):
        top10_share = float(np.sort(w)[-10:].sum())
        print(f"round {r}: top-10 hardest rows hold {top10_share:.1%} of total weight")

    # Sanity check against sklearn.
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=5,
        learning_rate=1.0,
        algorithm="SAMME",
        random_state=0,
    )
    ada.fit(X, y)
    print(f"\nsklearn AdaBoost train accuracy: {ada.score(X, y):.3f}")
```

Expected: top-10 weight share grows monotonically. Round 0 holds $10/569 \approx 1.8\%$; by round 5 it's typically 10–25%.

## Runnable: Why You Don't Use a Strong Base Learner (Iris)

Iris is small enough (150 samples) that a strong learner trivially memorizes it, making the failure mode obvious.

```python
from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)

    # Stump: weak learner, AdaBoost works as designed.
    stump_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, algorithm="SAMME", random_state=0,
    ).fit(X, y)
    print(f"stump base, 50 rounds:    train acc = {stump_ada.score(X, y):.3f}, "
          f"actually trained = {len(stump_ada.estimators_)} rounds")

    # Deep tree: strong learner, often terminates early or overfits round 1.
    deep_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50, algorithm="SAMME", random_state=0,
    ).fit(X, y)
    print(f"deep tree, 50 rounds:     train acc = {deep_ada.score(X, y):.3f}, "
          f"actually trained = {len(deep_ada.estimators_)} rounds")

    # SVC: another strong learner, same problem.
    svc_ada = AdaBoostClassifier(
        estimator=SVC(C=1.0, kernel="rbf"),
        n_estimators=50, algorithm="SAMME", random_state=0,
    ).fit(X, y)
    print(f"SVC base, 50 rounds:      train acc = {svc_ada.score(X, y):.3f}, "
          f"actually trained = {len(svc_ada.estimators_)} rounds")
```

Watch `len(estimators_)` — strong learners often produce far fewer than 50 rounds because SAMME bails when weighted error crosses 0.5 or hits zero.

## Runnable: AdaBoost vs. Gradient Boosting Head-to-Head (Penguins)

Demonstrates the practical reality that gradient boosting typically outperforms AdaBoost on real-world tasks.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    # Penguins dataset — 3-class, mix of numeric and categorical.
    penguins = fetch_openml(data_id=42585, as_frame=True, parser="auto")
    df = penguins.frame.dropna()
    X = pd.get_dummies(df.drop(columns=["species"]), drop_first=True).astype(float)
    y = df["species"]

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, algorithm="SAMME", random_state=0,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=100, max_depth=3, random_state=0,
    )

    ada_scores = cross_val_score(ada, X, y, cv=5, scoring="accuracy")
    hgb_scores = cross_val_score(hgb, X, y, cv=5, scoring="accuracy")

    print(f"AdaBoost (stumps):           {ada_scores.mean():.3f} +/- {ada_scores.std():.3f}")
    print(f"HistGradientBoosting:        {hgb_scores.mean():.3f} +/- {hgb_scores.std():.3f}")
```

## Common Gotchas

- **Don't use strong base learners.** Stumps or `max_depth=2` trees. SVMs, deep trees, random forests as base estimators almost always misbehave.
- **`learning_rate` and `n_estimators` trade off.** Halving learning rate roughly doubles the number of rounds you need. `learning_rate=0.2, n_estimators=10` is severely under-trained relative to the defaults.
- **`sample_weight` support in base estimator.** sklearn falls back to *resampling rows* when the base estimator doesn't reliably support weights. This is a different algorithm and is noisier. Sticking to trees avoids this.
- **AdaBoost is largely superseded.** Gradient boosting (XGBoost, LightGBM, `HistGradientBoostingClassifier`) handles more loss functions, regularizes better, and is faster. Reach for AdaBoost only when you have a specific reason.
- **SAMME.R was removed in sklearn 1.6.** Use `algorithm="SAMME"`.

## Plays Well With

- [[Boosting]] — the umbrella concept
- [[Gradient Boosting]] — AdaBoost is equivalent to gradient boosting with exponential loss
- [[Sklearn Decision Tree Visualization]] — visualize the stumps

## See Also

- Freund & Schapire (1997), *A Decision-Theoretic Generalization of On-Line Learning*
- [sklearn AdaBoostClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
