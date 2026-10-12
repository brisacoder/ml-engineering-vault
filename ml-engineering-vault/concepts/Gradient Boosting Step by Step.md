---
type: concept
title: "Gradient Boosting Step by Step"
libs:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
tags:
  - concept/ensemble
  - concept/boosting
  - concept/optimization
  - concept/loss-function
related:
  - "[[Gradient Boosting]]"
  - "[[Boosting]]"
  - "[[AdaBoost]]"
  - "[[Random Forest]]"
  - "[[Sklearn Decision Tree Visualization]]"
  - "[[Sklearn Decision Function and Thresholds]]"
created: 2026-09-29
updated: 2026-09-29
---

# Gradient Boosting Step by Step

> The math of gradient boosting is three lines. The confusion comes from not seeing *which numbers* those lines produce inside *which tree*. This note fits trees one at a time on Penguins, prints each tree, and does the arithmetic for three specific birds after every round. Read it with [[Gradient Boosting]] (the conceptual overview) open next to it.

Every number below was produced by the code in this note. Dataset: seaborn's `penguins`, rows with any missing value dropped, giving 333 rows (146 Adelie, 68 Chinstrap, 119 Gentoo; 168 male, 165 female).

---

## The Algorithm as a Recipe

Friedman's gradient boosting, for any differentiable loss $L(y, F)$:

$$
\begin{aligned}
F_0(x) &= \arg\min_c \sum_i L(y_i, c) && \text{(1) one constant for everyone} \\
r_i^{(m)} &= -\frac{\partial L(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)} && \text{(2) the tree's target this round} \\
h_m &= \text{a regression tree fit to } (x_i, r_i^{(m)}) && \text{(3) splits chosen by squared error on } r \\
\gamma_{jm} &= \arg\min_\gamma \sum_{x_i \in R_{jm}} L\big(y_i, F_{m-1}(x_i) + \gamma\big) && \text{(4) one number per leaf } R_{jm} \\
F_m(x) &= F_{m-1}(x) + \nu \cdot \gamma_{j(x), m} && \text{(5) shrink and add}
\end{aligned}
$$

Where each symbol shows up in code and in a printed tree:

| Symbol | Meaning | Variable in the code below | Where you see it in the tree printout |
|---|---|---|---|
| $F_{m-1}(x_i)$ | current ensemble prediction for row $i$ (raw score: grams for regression, log-odds for classification) | `F` | nowhere; it lives outside the tree |
| $r_i^{(m)}$ | negative gradient = "pseudo-residual" | `r = y - F` (L2) or `r = y - p` (log-loss) | it is the target the splits are chosen on |
| $h_m$ | the tree structure (which feature, which threshold) | `tree` returned by `fit_tree` | the `feature <= threshold` lines |
| $R_{jm}$ | leaf $j$ of tree $m$: a set of rows | `step == v` mask | one `leaf value=...` line |
| $\gamma_{jm}$ | leaf output | `tree.value` / `leaf value` | the number printed on the leaf |
| $\nu$ | learning rate | `lr` | not in the tree; applied when adding |

Step (4) is where most of the confusion lives. For squared error it collapses to "mean of the residuals in the leaf", so people learn that as the rule. For log-loss it does **not** collapse to the mean, and the leaf values printed by sklearn look nothing like the residuals. Section 3 works through exactly that.

---

## 1. A Tree Builder You Can Read

sklearn's trees are Cython; you cannot step through them. This is a ~90 line NumPy CART that produces the **same splits and the same leaf values** as `GradientBoostingRegressor` / `GradientBoostingClassifier` (verified in Section 4), with a hook that also reproduces XGBoost's Newton variant (Section 5). Save it as `gb_trace.py` next to the other scripts.

```python
"""Gradient boosting from scratch, built to be *inspected*.

Every tree is a plain Node object you can print. Every round is a function
call whose intermediate arrays (residuals, leaf values, updated predictions)
are returned so you can look at the numbers, not just the final score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float64]


@dataclass(frozen=True)
class Node:
    """One node of a binary regression tree. Leaves have feature=None."""

    n: int
    value: float  # leaf output (only meaningful when feature is None)
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def best_split(
    X: Float, g: Float, h: Float, lam: float
) -> tuple[float, int, float]:
    """Exhaustive best split, scored as Newton gain.

    score(split) = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)

    With h == 1 and lam == 0 this is exactly the squared-error reduction
    n_L*n_R/n * (mean_L - mean_R)^2 that sklearn's CART maximises, so the same
    routine serves both the classic Friedman GBM and the XGBoost variant.
    Vectorised over all candidate thresholds of a feature via prefix sums.
    """
    n, p = X.shape
    G, H = g.sum(), h.sum()
    parent = G * G / (H + lam)
    best_gain, best_feat, best_thr = 0.0, -1, np.nan

    for j in range(p):
        order = np.argsort(X[:, j], kind="stable")
        xs, gs, hs = X[order, j], g[order], h[order]
        GL, HL = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]  # left = first k rows
        GR, HR = G - GL, H - HL
        gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent
        valid = xs[:-1] != xs[1:]  # cannot split between equal values
        gain = np.where(valid, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain, best_feat = float(gain[k]), j
            best_thr = float((xs[k] + xs[k + 1]) / 2)  # midpoint, like sklearn
    return best_gain, best_feat, best_thr


def fit_tree(
    X: Float,
    g: Float,
    h: Float,
    *,
    max_depth: int,
    lam: float = 0.0,
    split_h: Float | None = None,
) -> Node:
    """Grow a depth-limited tree.

    g       : per-row "target" for the tree (negative gradient, i.e. residual)
    h       : per-row hessian used for the LEAF value  sum(g)/(sum(h)+lam)
    split_h : hessian used for SPLIT scoring; defaults to h.
              sklearn's GBM scores splits with plain squared error (split_h=1)
              but fills leaves with the Newton step (h = p(1-p)). XGBoost uses
              h for both. Passing them separately makes that difference visible.
    """
    if split_h is None:
        split_h = h
    value = float(g.sum() / (h.sum() + lam))
    if max_depth == 0 or len(g) < 2:
        return Node(n=len(g), value=value)
    gain, j, thr = best_split(X, g, split_h, lam)
    if j < 0:
        return Node(n=len(g), value=value)
    mask = X[:, j] <= thr
    return Node(
        n=len(g),
        value=value,
        feature=j,
        threshold=thr,
        left=fit_tree(X[mask], g[mask], h[mask], max_depth=max_depth - 1, lam=lam, split_h=split_h[mask]),
        right=fit_tree(X[~mask], g[~mask], h[~mask], max_depth=max_depth - 1, lam=lam, split_h=split_h[~mask]),
    )


def predict_tree(node: Node, X: Float) -> Float:
    """Route every row to its leaf and return the leaf values (vectorised)."""
    out = np.full(len(X), node.value)
    if node.is_leaf:
        return out
    mask = X[:, node.feature] <= node.threshold
    out[mask] = predict_tree(node.left, X[mask])
    out[~mask] = predict_tree(node.right, X[~mask])
    return out


def format_tree(node: Node, names: list[str], depth: int = 0) -> str:
    """Text rendering in the style of sklearn.tree.export_text."""
    pad = "|   " * depth
    if node.is_leaf:
        return f"{pad}|--- leaf value={node.value:+.4f}  (n={node.n})\n"
    name = names[node.feature]
    return (
        f"{pad}|--- {name} <= {node.threshold:.2f}\n"
        + format_tree(node.left, names, depth + 1)
        + f"{pad}|--- {name} >  {node.threshold:.2f}\n"
        + format_tree(node.right, names, depth + 1)
    )


def sigmoid(z: Float) -> Float:
    return 1.0 / (1.0 + np.exp(-z))
```

Two design points worth noticing:

- **The leaf value is always `sum(g) / (sum(h) + lam)`.** With `h = 1` and `lam = 0` that is the mean of `g`, which is the squared-error leaf. With `h = p(1-p)` it is the log-loss Newton step. With `lam > 0` it is XGBoost's regularised leaf. One formula, three textbooks' worth of special cases.
- **Split scoring and leaf filling are separate arguments** (`split_h` vs `h`). sklearn's classic GBM scores splits with plain squared error on the residuals and then overwrites the leaves with the Newton step; XGBoost uses the hessian in both places. The difference is real and shows up in Section 5.

---

## 2. Regression: Squared Error, Tree by Tree

Predict `body_mass_g` from bill length, bill depth and flipper length. Depth-2 trees (four leaves) so the whole tree fits on the screen, learning rate 0.1. Three birds are tracked through every round: row 0 (Adelie male, 3750 g), row 146 (Chinstrap female, 3500 g), row 215 (Gentoo male, 5700 g).

```python
"""Trace squared-error gradient boosting round by round on Penguins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

from gb_trace import fit_tree, format_tree, predict_tree

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
TARGET = "body_mass_g"
WATCH = [0, 146, 215]  # an Adelie male, a Chinstrap female, a Gentoo male


def load() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
    return df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float), df


def main() -> None:
    X, y, df = load()
    n = len(y)
    lr, depth, rounds = 0.1, 2, 3
    print(df.loc[WATCH, ["species", "sex", *FEATURES, TARGET]].to_string())

    F = np.full(n, y.mean())  # F_0: the constant that minimises squared error
    print(f"\nF0 = mean(y) = {y.mean():.4f}")

    ones = np.ones(n)
    for m in range(1, rounds + 1):
        r = y - F  # negative gradient of 0.5*(y-F)^2 w.r.t. F
        tree = fit_tree(X, r, ones, max_depth=depth)
        step = predict_tree(tree, X)
        F_new = F + lr * step
        print(f"\n===== round {m} =====")
        print(f"residual stats: mean={r.mean():+.4f} std={r.std():.4f}")
        print(format_tree(tree, FEATURES), end="")
        tbl = pd.DataFrame(
            {
                "y": y[WATCH],
                f"F{m-1}": F[WATCH],
                f"r{m}=y-F{m-1}": r[WATCH],
                f"tree{m}(x)": step[WATCH],
                f"F{m}=F{m-1}+{lr}*tree": F_new[WATCH],
            },
            index=WATCH,
        )
        print(tbl.round(4).to_string())
        F = F_new
        print(f"train RMSE after round {m}: {np.sqrt(np.mean((y - F) ** 2)):.4f}")

    # keep going silently and report the curve
    for m in range(rounds + 1, 101):
        r = y - F
        tree = fit_tree(X, r, ones, max_depth=depth)
        F = F + lr * predict_tree(tree, X)
        if m in (5, 10, 20, 50, 100):
            print(f"train RMSE after round {m:3d}: {np.sqrt(np.mean((y - F) ** 2)):.4f}")


if __name__ == "__main__":
    main()
```

### Round 0: the constant

Step (1) with squared error: $\arg\min_c \sum (y_i - c)^2$ is the mean.

```
F0 = mean(y) = 4207.0571
```

Every penguin is predicted to weigh 4207 g. The Gentoo male (5700 g) is under-predicted by 1493 g, the Chinstrap female (3500 g) over-predicted by 707 g. Those gaps are the round-1 residuals.

### Round 1

Step (2): `r = y - F0`. Step (3): fit a tree to `r`. Here is tree 1:

```
|--- flipper_length_mm <= 206.50
|   |--- bill_depth_mm <= 18.05
|   |   |--- leaf value=-757.3444  (n=87)
|   |--- bill_depth_mm >  18.05
|   |   |--- leaf value=-322.7595  (n=121)
|--- flipper_length_mm >  206.50
|   |--- flipper_length_mm <= 214.50
|   |   |--- leaf value=+407.7389  (n=49)
|   |--- flipper_length_mm >  214.50
|   |   |--- leaf value=+1117.9429  (n=76)
```

How to read it against the math:

- **The root split** `flipper_length_mm <= 206.50` is the (feature, threshold) pair that maximises the drop in $\sum r_i^2$. Left side: 208 rows with mean residual $-504.53$. Right side: 125 rows with mean residual $+839.54$. Gain $= \frac{208 \cdot 125}{333}(839.54 + 504.53)^2 \approx 1.41 \times 10^8$. Every other candidate on every feature scored lower.
- **A leaf value is step (4)**, and for squared error $\arg\min_\gamma \sum (r_i - \gamma)^2$ is the mean of the residuals in that leaf. Check: the 87 birds with flipper $\le 206.5$ and bill depth $\le 18.05$ have mean residual exactly $-757.3444$.
- **The leaf is in grams**, the same unit as $y$ and $F$. That is the special property of squared error that makes the next section confusing.

Step (5) for the three tracked birds:

```
          y         F0    r1=y-F0   tree1(x)  F1=F0+0.1*tree
0    3750.0  4207.0571  -457.0571  -322.7595       4174.7811
146  3500.0  4207.0571  -707.0571  -757.3444       4131.3226
215  5700.0  4207.0571  1492.9429  1117.9429       4318.8514
```

Row 215 by hand: flipper 230 > 206.5, and 230 > 214.5, so it lands in the `+1117.9429` leaf. $F_1 = 4207.0571 + 0.1 \times 1117.9429 = 4318.8514$. Only 10% of the correction is applied; the bird is still 1381 g too light. That is the learning rate doing its job, and it is why round 2 sees nearly the same residual pattern.

Train RMSE: 804.0 (round 0) to 742.9.

### Round 2

Step (2) again, but now `r = y - F1`. The residuals changed, so the tree changes:

```
|--- flipper_length_mm <= 202.50
|   |--- bill_depth_mm <= 18.05
|   |   |--- leaf value=-695.2761  (n=86)
|   |--- bill_depth_mm >  18.05
|   |   |--- leaf value=-318.3652  (n=113)
|--- flipper_length_mm >  202.50
|   |--- flipper_length_mm <= 212.50
|   |   |--- leaf value=+259.6352  (n=47)
|   |--- flipper_length_mm >  212.50
|   |   |--- leaf value=+960.5305  (n=87)
```

Note the root threshold moved from 206.5 to 202.5, and the right-hand split from 214.5 to 212.5. Nothing about the *features* changed; the *target* changed, because the 76 heaviest birds already had their residual shrunk by 112 g, so a slightly different cut now separates the residuals best. The leaf magnitudes also shrank (1118 to 961) because 10% of the signal was already absorbed.

```
          y         F1    r2=y-F1  tree2(x)  F2=F1+0.1*tree
0    3750.0  4174.7811  -424.7811 -318.3652       4142.9446
146  3500.0  4131.3226  -631.3226 -695.2761       4061.7950
215  5700.0  4318.8514  1381.1486  960.5305       4414.9044
```

Train RMSE 742.9 to 688.5.

### Round 3

```
|--- flipper_length_mm <= 202.50
|   |--- bill_depth_mm <= 18.05
|   |   |--- leaf value=-625.7485  (n=86)
|   |--- bill_depth_mm >  18.05
|   |   |--- leaf value=-286.5287  (n=113)
|--- flipper_length_mm >  202.50
|   |--- flipper_length_mm <= 214.50
|   |   |--- leaf value=+293.5318  (n=58)
|   |--- flipper_length_mm >  214.50
|   |   |--- leaf value=+910.0956  (n=76)
```

```
          y         F2    r3=y-F2  tree3(x)  F3=F2+0.1*tree
0    3750.0  4142.9446  -392.9446 -286.5287       4114.2917
146  3500.0  4061.7950  -561.7950 -625.7485       3999.2202
215  5700.0  4414.9044  1285.0956  910.0956       4505.9140
```

Train RMSE 688.5 to 640.9. Continuing silently: 561.7 (round 5), 434.4 (10), 339.6 (20), 296.7 (50), 276.6 (100).

### What the final model is

After 100 rounds, the prediction for any bird is

$$F_{100}(x) = 4207.06 + 0.1 \cdot \big[\text{tree}_1(x) + \text{tree}_2(x) + \dots + \text{tree}_{100}(x)\big]$$

where each $\text{tree}_m(x)$ is "route $x$ to a leaf, read the leaf's number". There is no voting and no averaging; it is a running sum of corrections, each computed against the sum of everything before it. Tree 50 in isolation says something like $+31$ g, which means nothing unless you know what trees 0 through 49 already predicted for that bird.

---

## 3. Classification: Log-Loss, Tree by Tree

Now predict `sex == MALE` from the four body measurements. Same tree builder, same recipe, different loss. This is where the mapping from math to tree usually breaks, for one reason: **the tree's target is in probability units, the tree's leaves are in log-odds units, and the two are connected by step (4).**

```python
"""Trace log-loss gradient boosting round by round on Penguins (sex = MALE?)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

from gb_trace import fit_tree, format_tree, predict_tree, sigmoid

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
WATCH = [0, 146, 215]


def load() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
    return df[FEATURES].to_numpy(float), (df["sex"].str.upper() == "MALE").to_numpy(float), df


def log_loss(y: np.ndarray, F: np.ndarray) -> float:
    # numerically stable: log(1+exp(F)) - y*F
    return float(np.mean(np.logaddexp(0.0, F) - y * F))


def main() -> None:
    X, y, df = load()
    n = len(y)
    lr, depth, rounds = 0.1, 2, 2
    print(df.loc[WATCH, ["species", "sex", *FEATURES]].to_string())

    prior = y.mean()
    F = np.full(n, np.log(prior / (1 - prior)))  # F_0 = log-odds of the base rate
    print(f"\nprior P(male) = {prior:.6f}  ->  F0 = log(p/(1-p)) = {F[0]:+.6f}")
    print(f"log-loss at F0: {log_loss(y, F):.6f}")

    for m in range(1, rounds + 1):
        p = sigmoid(F)
        r = y - p           # negative gradient of log-loss w.r.t. F
        h = p * (1 - p)     # second derivative (hessian) of log-loss w.r.t. F
        # split on plain squared error of r (split_h = 1), leaf = sum(r)/sum(h)
        tree = fit_tree(X, r, h, max_depth=depth, split_h=np.ones(n))
        step = predict_tree(tree, X)
        F_new = F + lr * step
        print(f"\n===== round {m} =====")
        print(f"residual r=y-p takes values: {np.unique(r.round(6))[:6]} ...")
        print(format_tree(tree, FEATURES), end="")

        # show the Newton leaf arithmetic for every leaf
        for v in np.unique(step):
            in_leaf = step == v
            print(
                f"  leaf {v:+.4f}: sum(r)={r[in_leaf].sum():+.4f}  "
                f"sum(p(1-p))={h[in_leaf].sum():.4f}  ratio={r[in_leaf].sum()/h[in_leaf].sum():+.4f}  "
                f"mean(r)={r[in_leaf].mean():+.4f}  n={in_leaf.sum()}  frac_male={y[in_leaf].mean():.3f}"
            )
        tbl = pd.DataFrame(
            {
                "y": y[WATCH],
                f"F{m-1}": F[WATCH],
                f"p{m-1}=sig(F)": p[WATCH],
                f"r{m}=y-p": r[WATCH],
                f"tree{m}(x)": step[WATCH],
                f"F{m}": F_new[WATCH],
                f"p{m}": sigmoid(F_new)[WATCH],
            },
            index=WATCH,
        )
        print(tbl.round(4).to_string())
        F = F_new
        print(f"log-loss after round {m}: {log_loss(y, F):.6f}   "
              f"accuracy: {np.mean((F > 0) == (y == 1)):.4f}")

    for m in range(rounds + 1, 101):
        p = sigmoid(F)
        tree = fit_tree(X, y - p, p * (1 - p), max_depth=depth, split_h=np.ones(n))
        F = F + lr * predict_tree(tree, X)
        if m in (5, 10, 20, 50, 100):
            print(f"round {m:3d}: log-loss={log_loss(y, F):.6f} accuracy={np.mean((F > 0) == (y == 1)):.4f}")


if __name__ == "__main__":
    main()
```

### The loss and its two derivatives

$F$ is the raw score (log-odds), $p = \sigma(F) = 1/(1+e^{-F})$. Log-loss per row:

$$L(y, F) = -y \log p - (1-y)\log(1-p) = \log(1 + e^{F}) - yF$$

$$\frac{\partial L}{\partial F} = p - y \quad\Rightarrow\quad r = -(p - y) = y - p \qquad\qquad \frac{\partial^2 L}{\partial F^2} = p(1-p)$$

So the tree's target `r = y - p` is a number in $(-1, 1)$: how far the current probability is from the label. The hessian `h = p(1-p)` is the curvature; it is what converts a probability gap back into a log-odds step.

### Round 0: the constant

Step (1): $\arg\min_c \sum L(y_i, c)$ is the log-odds of the base rate.

```
prior P(male) = 0.504505  ->  F0 = log(p/(1-p)) = +0.018019
log-loss at F0: 0.693107
```

Note $F_0$ is in log-odds, not probability. `predict_proba` at this point is $\sigma(0.018) = 0.5045$ for every bird. (This is the same raw-score idea as in [[Sklearn Decision Function and Thresholds]]: `decision_function` returns $F$, `predict` thresholds it at 0.)

### Round 1

Every bird has the same $p_0 = 0.5045$, so the residual takes only two values: $+0.4955$ for males, $-0.5045$ for females.

```
residual r=y-p takes values: [-0.504505  0.495495]
|--- body_mass_g <= 3712.50
|   |--- bill_depth_mm <= 18.55
|   |   |--- leaf value=-1.9218  (n=83)
|   |--- bill_depth_mm >  18.55
|   |   |--- leaf value=+0.3820  (n=30)
|--- body_mass_g >  3712.50
|   |--- bill_depth_mm <= 14.85
|   |   |--- leaf value=-1.6610  (n=56)
|   |--- bill_depth_mm >  14.85
|   |   |--- leaf value=+1.4699  (n=164)
```

Stop here. The residuals are all within $\pm 0.51$, yet the leaves say $-1.92$ and $+1.47$. **A leaf is not the mean residual.** Here is what it is, leaf by leaf:

```
leaf -1.9218: sum(r)=-39.8739  sum(p(1-p))=20.7483  ratio=-1.9218  mean(r)=-0.4804  n=83   frac_male=0.024
leaf -1.6610: sum(r)=-23.2523  sum(p(1-p))=13.9989  ratio=-1.6610  mean(r)=-0.4152  n=56   frac_male=0.089
leaf +0.3820: sum(r)=+2.8649   sum(p(1-p))=7.4994   ratio=+0.3820  mean(r)=+0.0955  n=30   frac_male=0.600
leaf +1.4699: sum(r)=+60.2613  sum(p(1-p))=40.9967  ratio=+1.4699  mean(r)=+0.3674  n=164  frac_male=0.872
```

Take the big leaf (heavy birds with bill depth > 14.85; 164 rows, 143 male, 21 female):

- $\sum r = 143 \times 0.4955 + 21 \times (-0.5045) = 70.86 - 10.59 = 60.26$
- $\sum p(1-p) = 164 \times 0.5045 \times 0.4955 = 40.997$
- $\gamma = 60.26 / 40.997 = 1.4699$

That is step (4) solved by one Newton step. The exact minimiser $\arg\min_\gamma \sum_{i \in \text{leaf}} L(y_i, F_{m-1}(x_i) + \gamma)$ has no closed form for log-loss, so Friedman approximates it with a single Newton-Raphson step from $\gamma = 0$:

$$\gamma_{jm} \approx \frac{-\sum_{i \in R_{jm}} \partial L / \partial F}{\sum_{i \in R_{jm}} \partial^2 L / \partial F^2} = \frac{\sum_{i \in R_{jm}} (y_i - p_i)}{\sum_{i \in R_{jm}} p_i (1 - p_i)}$$

Units check: numerator is a probability gap, denominator is $dp/dF$, so the ratio is a step in $F$ (log-odds). Sanity check: this leaf is 87.2% male; the log-odds of 0.872 is $1.92$; one Newton step from $F_0 \approx 0$ gets $1.47$ of the way there, and later trees will add the rest.

If you used the mean residual ($+0.3674$) as the leaf value instead, you would be adding a probability to a log-odds. The model would still "work" (the sign is right) but it would need far more rounds, and it would not be gradient boosting as sklearn or XGBoost implement it.

Also notice: **the split structure was chosen on `r` alone with plain squared error.** The hessian only enters when filling the leaves. That is sklearn's `GradientBoostingClassifier` exactly (criterion `friedman_mse` on the residuals, then `_update_terminal_regions` overwrites each leaf with the ratio above).

Step (5) for the tracked birds:

```
       y     F0  p0=sig(F)  r1=y-p  tree1(x)      F1      p1
0    1.0  0.018     0.5045  0.4955    1.4699  0.1650  0.5412
146  0.0  0.018     0.5045 -0.5045   -1.9218 -0.1742  0.4566
215  1.0  0.018     0.5045  0.4955    1.4699  0.1650  0.5412
```

Row 146 by hand: body mass 3500 $\le$ 3712.5, bill depth 17.9 $\le$ 18.55, so leaf $-1.9218$. $F_1 = 0.0180 + 0.1 \times (-1.9218) = -0.1742$; $p_1 = \sigma(-0.1742) = 0.4566$. The probability moved from 50.5% to 45.7% male. Accuracy is already 88% (the sign of $F$ is right for most birds) but log-loss is only 0.693 to 0.635 because the probabilities are still timid. Log-loss, not accuracy, is what the trees are chasing.

### Round 2

Now $p$ differs by bird, so `r` takes several values. Tree 2:

```
|--- body_mass_g <= 3712.50
|   |--- bill_depth_mm <= 18.55
|   |   |--- leaf value=-1.7430  (n=83)
|   |--- bill_depth_mm >  18.55
|   |   |--- leaf value=+0.3441  (n=30)
|--- body_mass_g >  3712.50
|   |--- bill_depth_mm <= 14.85
|   |   |--- leaf value=-1.5033  (n=56)
|   |--- bill_depth_mm >  14.85
|   |   |--- leaf value=+1.3322  (n=164)
```

Leaf arithmetic:

```
leaf -1.7430: sum(r)=-35.8953  sum(p(1-p))=20.5934  ratio=-1.7430  n=83   frac_male=0.024
leaf -1.5033: sum(r)=-20.9306  sum(p(1-p))=13.9235  ratio=-1.5033  n=56   frac_male=0.089
leaf +0.3441: sum(r)=+2.5785   sum(p(1-p))=7.4941   ratio=+0.3441  n=30   frac_male=0.600
leaf +1.3322: sum(r)=+54.2499  sum(p(1-p))=40.7222  ratio=+1.3322  n=164  frac_male=0.872
```

The **structure is identical** to tree 1 (same features, same thresholds, same leaf counts). With $\nu = 0.1$, round 1 moved every $p$ by at most 0.05, so the ranking of residuals barely changed and the same cuts win again. The leaves shrank (1.4699 to 1.3322) because the remaining gap is smaller. This is the learning-rate/n_estimators trade-off made visible: a small $\nu$ makes consecutive trees near-duplicates, each taking a small bite.

```
       y      F1  p1=sig(F)  r2=y-p  tree2(x)      F2      p2
0    1.0  0.1650     0.5412  0.4588    1.3322  0.2982  0.5740
146  0.0 -0.1742     0.4566 -0.4566   -1.7430 -0.3485  0.4138
215  1.0  0.1650     0.5412  0.4588    1.3322  0.2982  0.5740
```

Continuing: log-loss 0.498 (round 5), 0.392 (10), 0.282 (20), 0.176 (50), 0.117 (100); accuracy 0.880 until round 10, then 0.919 (20), 0.958 (50), 0.976 (100). Rows 0 and 215 are indistinguishable to a depth-2 tree for the first rounds (both heavy-ish with deep bills); deeper trees or later rounds separate them.

### The final model

$$P(\text{male} \mid x) = \sigma\Big(0.0180 + 0.1 \sum_{m=1}^{100} \text{tree}_m(x)\Big)$$

`decision_function(x)` is the thing inside $\sigma$. `predict_proba` applies $\sigma$. `predict` checks the sign. Every tree's leaf is a log-odds increment computed by the Newton ratio, never a probability.

---

## 4. Checking Against scikit-learn

The trees above should be bit-for-bit the trees sklearn builds (same CART, same midpoint thresholds, same leaf rule). This script asserts it. The one place ties could diverge is when two (feature, threshold) pairs have identical gain, which does not happen on this data.

```python
"""Assert the from-scratch trace matches sklearn's GradientBoosting* exactly."""

from __future__ import annotations

import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import export_text

from gb_trace import fit_tree, predict_tree, sigmoid


def main() -> None:
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
    watch = [0, 146, 215]

    # ---------- regression ----------
    feats_r = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
    X = df[feats_r].to_numpy(float)
    y = df["body_mass_g"].to_numpy(float)
    gbr = GradientBoostingRegressor(n_estimators=3, learning_rate=0.1, max_depth=2, random_state=0)
    gbr.fit(X, y)

    t1 = gbr.estimators_[0, 0].tree_
    print("sklearn tree 1 root:", feats_r[t1.feature[0]], "<=", t1.threshold[0])  # flipper <= 206.5
    print(export_text(gbr.estimators_[0, 0], feature_names=feats_r, decimals=4))

    F = np.full(len(y), y.mean())
    for m, F_sk in enumerate(gbr.staged_predict(X), start=1):
        tree = fit_tree(X, y - F, np.ones(len(y)), max_depth=2)
        F = F + 0.1 * predict_tree(tree, X)
        assert np.allclose(F, F_sk), f"regression round {m} diverged"
        print(f"regression round {m}: from-scratch == sklearn  ", F[watch].round(4))

    # ---------- classification ----------
    feats_c = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = df[feats_c].to_numpy(float)
    y = (df["sex"].str.upper() == "MALE").to_numpy(float)
    gbc = GradientBoostingClassifier(n_estimators=2, learning_rate=0.1, max_depth=2, random_state=0)
    gbc.fit(X, y)

    # leaves in tree_.value are the Newton-updated log-odds steps, e.g. -1.9218 / +1.4699
    print(export_text(gbc.estimators_[0, 0], feature_names=feats_c, decimals=4))

    prior = y.mean()
    F = np.full(len(y), np.log(prior / (1 - prior)))
    for m, F_sk in enumerate(gbc.staged_decision_function(X), start=1):
        p = sigmoid(F)
        tree = fit_tree(X, y - p, p * (1 - p), max_depth=2, split_h=np.ones(len(y)))
        F = F + 0.1 * predict_tree(tree, X)
        assert np.allclose(F, F_sk.ravel()), f"classification round {m} diverged"
        print(f"classification round {m}: from-scratch == sklearn  F=", F[watch].round(4),
              " p=", sigmoid(F)[watch].round(4))


if __name__ == "__main__":
    main()
```

What to look at in sklearn's own objects once this passes:

- `gbr.init_.constant_` is $F_0$ (the mean). `gbc.init_` is a `DummyClassifier(strategy="prior")`; sklearn converts its probability to log-odds internally.
- `est.estimators_[m, 0]` is tree $m+1$, a plain `DecisionTreeRegressor`. Print it with `export_text` as in [[Sklearn Decision Tree Visualization]]. For the classifier, the `value` on each leaf is the Newton ratio, not a class probability.
- `staged_predict` / `staged_decision_function` yield $F_1, F_2, \dots$, which is how you reproduce the per-round tables without re-implementing anything.

---

## 5. XGBoost's Variant: Newton at the Split Too

XGBoost (Chen & Guestrin 2016) keeps steps (1), (2), (5) and changes (3) and (4) in two ways:

1. **Splits are scored with the hessian.** Instead of squared-error reduction on $r$, a split is scored by

$$\text{gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

   where $G = \sum g_i$ (XGBoost's $g_i = \partial L/\partial F = p_i - y_i$, the *positive* gradient) and $H = \sum h_i$ over the rows in a node. With $h_i = 1$ and $\lambda = 0$ this is exactly the squared-error gain, which is why `best_split` above serves both.

2. **Leaves are regularised:** $w_j = -\dfrac{G_j}{H_j + \lambda}$, with `reg_lambda` $= \lambda = 1$ by default. Same Newton ratio as sklearn, plus $\lambda$ in the denominator.

Because `fit_tree` takes the *negative* gradient as `g`, its `sum(g)/(sum(h)+lam)` equals XGBoost's $-G/(H+\lambda)$ directly.

```python
"""Same data, XGBoost-style tree: Newton gain at the split, lambda in the leaf."""

from __future__ import annotations

import numpy as np
import seaborn as sns

from gb_trace import fit_tree, format_tree, predict_tree, sigmoid

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def main() -> None:
    df = sns.load_dataset("penguins").dropna().reset_index(drop=True)
    X = df[FEATURES].to_numpy(float)
    y = (df["sex"].str.upper() == "MALE").to_numpy(float)
    n, lam, lr = len(y), 1.0, 0.3  # XGBoost defaults: reg_lambda=1, eta=0.3

    prior = y.mean()
    F = np.full(n, np.log(prior / (1 - prior)))
    for m in (1, 2):
        p = sigmoid(F)
        g, h = y - p, p * (1 - p)  # g here is the NEGATIVE gradient (y - p)
        # split_h defaults to h -> Newton gain at every split, lambda everywhere
        tree = fit_tree(X, g, h, max_depth=2, lam=lam)
        print(f"===== XGBoost-style round {m} (lambda={lam}) =====")
        print(format_tree(tree, FEATURES), end="")
        G, H = g.sum(), h.sum()
        mask = X[:, tree.feature] <= tree.threshold
        GL, HL, GR, HR = g[mask].sum(), h[mask].sum(), g[~mask].sum(), h[~mask].sum()
        gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam))
        print(f"root: G={G:+.4f} H={H:.4f} | GL={GL:+.4f} HL={HL:.4f} | GR={GR:+.4f} HR={HR:.4f}")
        print(f"root gain = 0.5*[GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam)] = {gain:.4f}")
        step = predict_tree(tree, X)
        for v in np.unique(step):
            in_leaf = step == v
            print(f"  leaf {v:+.4f} = sum(g)/(sum(h)+lam) = "
                  f"{g[in_leaf].sum():+.4f}/({h[in_leaf].sum():.4f}+{lam}) n={in_leaf.sum()}")
        F = F + lr * step


if __name__ == "__main__":
    main()
```

Round 1 output:

```
|--- body_mass_g <= 3712.50
|   |--- bill_depth_mm <= 18.55
|   |   |--- leaf value=-1.8334  (n=83)
|   |--- bill_depth_mm >  18.55
|   |   |--- leaf value=+0.3371  (n=30)
|--- body_mass_g >  3712.50
|   |--- bill_depth_mm <= 14.85
|   |   |--- leaf value=-1.5503  (n=56)
|   |--- bill_depth_mm >  14.85
|   |   |--- leaf value=+1.4349  (n=164)
root: G=+0.0000 H=83.2432 | GL=-37.0090 HL=28.2477 | GR=+37.0090 HR=54.9955
root gain = 35.6451
  leaf -1.8334 = -39.8739/(20.7483+1.0)  n=83
  leaf +1.4349 = +60.2613/(40.9967+1.0)  n=164
```

Same structure as sklearn's tree 1 (in round 1 every $h_i$ is the same constant, so Newton gain and squared-error gain rank splits identically), but every leaf is pulled toward zero by the $+1$ in the denominator: $60.2613/40.9967 = 1.4699$ became $60.2613/41.9967 = 1.4349$. That is what `reg_lambda` does, leaf by leaf.

Round 2 is where the two algorithms part ways:

```
|--- bill_length_mm <= 48.30
|   |--- bill_depth_mm <= 17.95
|   |   |--- leaf value=-1.1269  (n=146)
|   |--- bill_depth_mm >  17.95
|   |   |--- leaf value=+0.5165  (n=97)
|--- bill_length_mm >  48.30
|   |--- body_mass_g <= 5175.00
|   |   |--- leaf value=+0.7857  (n=49)
|   |--- body_mass_g >  5175.00
|   |   |--- leaf value=+1.5681  (n=41)
```

A different root feature entirely. Two things changed together: $\nu = 0.3$ moved the probabilities three times as far as in Section 3, so the residual pattern shifted more; and the hessians are no longer uniform, so rows whose $p$ is already near 0 or 1 (small $h$) count for less in the gain. Confident rows stop steering the tree. That is the practical content of "second-order boosting".

Comparing against real XGBoost on your machine:

```python
import xgboost as xgb

prior = y.mean()
booster = xgb.XGBClassifier(
    n_estimators=2, max_depth=2, learning_rate=0.3, reg_lambda=1.0,
    base_score=prior,        # pin it; xgboost>=2 otherwise estimates it from data
    min_child_weight=0.0,    # default 1.0 is a hessian-sum floor; the trace has none
    tree_method="exact",
)
booster.fit(X, y)
print(booster.get_booster().get_dump(with_stats=True)[0])   # tree 1 with gain/cover
raw = booster.predict(X, output_margin=True)                # F_2, log-odds
```

`get_dump(with_stats=True)` prints `gain=` and `cover=` per node; `cover` is $H$ for that node (83.24 at the root above) and `gain` should match the 35.6451 computed by hand. `output_margin=True` returns $F$ rather than $\sigma(F)$. Expect the leaf values to agree to ~1e-4 (XGBoost stores floats). I could not run xgboost in the sandbox where this note was produced, so this last block is the one thing here you verify yourself.

---

## 6. Quick Reference: Which Number Is Which

| You are looking at | Regression (L2) | Binary classification (log-loss) |
|---|---|---|
| $F_0$ | `y.mean()` | `log(prior / (1 - prior))` |
| tree target `r` | `y - F` (grams) | `y - sigmoid(F)` (probability gap, in $(-1,1)$) |
| split criterion | SSE reduction on `r` | sklearn: SSE reduction on `r`; XGBoost: Newton gain with `h = p(1-p)` |
| leaf value | `mean(r_leaf)` | `sum(r_leaf) / sum(p(1-p))_leaf` (XGBoost: `+ lambda` in denominator) |
| leaf units | grams | log-odds |
| update | `F += lr * leaf` | `F += lr * leaf` (still log-odds) |
| what you report | `F` | `sigmoid(F)` for probability, `F > 0` for the label |
| sklearn accessor for $F_m$ | `staged_predict` | `staged_decision_function` |

### Common Misconceptions

- **"The leaf is the average residual."** Only for squared error. For every other loss the leaf is a Newton step $\sum g / \sum h$ (sklearn, XGBoost, LightGBM, CatBoost all do this). The tree's *splits* may be chosen on residuals, but its *leaves* are not residual means.
- **"Each tree predicts the class."** No tree in a boosted classifier predicts a class. Each emits a log-odds increment; only $\sigma(F_M)$ is a probability. Compare [[AdaBoost]], where each stump really does emit a label.
- **"Round 2 sees the same target as round 1."** The target is `y - F` and `F` changed. The thresholds moving between rounds (206.5 to 202.5 in Section 2) is the visible evidence.
- **"A small learning rate just makes training slower."** It also makes consecutive trees near-identical (Section 3, rounds 1 and 2 had the same structure). That redundancy is a form of regularisation, which is why lowering $\nu$ and raising `n_estimators` together usually generalises better than the reverse. See the tuning notes in [[Gradient Boosting]].
- **"XGBoost is just faster sklearn GBM."** The leaf formula is the same family, but the split criterion differs (hessian-weighted) and $\lambda$ shrinks every leaf. Given identical hyperparameters the two libraries build different trees from round 2 onward.

## See Also

- Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*, Algorithm 1 is steps (1) to (5); Section 4.5 derives the log-loss Newton leaf.
- Chen & Guestrin (2016), *XGBoost*, equations (6) and (7) are the leaf weight and gain used in Section 5.
- sklearn source: `sklearn/ensemble/_gb.py`, `_update_terminal_regions` is where the leaf overwrite happens.
