---
type: concept
title: "Lasso Regression"
libs:
  - scikit-learn
  - numpy
tags:
  - concept/regularization
  - concept/optimization
related:
  - "[[Ridge Regression]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Elastic Net]]"
created: 2026-04-29
updated: 2026-04-29
---

# Lasso Regression

> Lasso — Least Absolute Shrinkage and Selection Operator — adds an L1 penalty (sum of absolute values of weights) to the loss function. Unlike [[Ridge Regression]] which only shrinks weights toward zero, Lasso can drive them to **exactly** zero, performing automatic feature selection. The reason is purely mechanical: the L1 penalty gradient is a constant force regardless of weight magnitude, so it can overpower the data gradient when a feature is weak.

---

## The Lasso Objective

### Intuition

Ridge penalizes the *square* of each weight, so the penalty force weakens as the weight approaches zero — equilibrium is reached at some small nonzero value. Lasso penalizes the *absolute value*, so the penalty force is constant regardless of weight magnitude. For a feature that contributes little to reducing MSE, that constant force is enough to push the weight all the way to zero and hold it there.

### Formal Definition

$$
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + 2\alpha \sum_{j=1}^{p} |\theta_j|
$$

Note the $2\alpha$ factor (not $\alpha/m$ as in Ridge). Géron explains this convention: the different norms require different scaling factors to ensure the optimal $\alpha$ is independent of training set size. See [scikit-learn issue #15657](https://github.com/scikit-learn/scikit-learn/issues/15657) for the full rationale.

The L1 norm is not differentiable at zero — the absolute value function has a kink there. This is not a minor technicality; it's the entire reason Lasso can produce exact zeros. We need the **subgradient** to handle this.

### In Code

```python
"""
Lasso regression: basic usage showing feature elimination.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 10 features, only 3 actually informative
X, y, true_coef = make_regression(
    n_samples=200, n_features=10, n_informative=3,
    noise=5.0, coef=True, random_state=42,
)
X = StandardScaler().fit_transform(X)

lasso = Lasso(alpha=0.5, fit_intercept=False)
lasso.fit(X, y)

print("True coefficients:")
print(np.round(true_coef, 2))
print("\nLasso coefficients:")
print(np.round(lasso.coef_, 2))
print(f"\nFeatures zeroed out: {np.sum(np.abs(lasso.coef_) < 1e-10)} / {X.shape[1]}")
print(f"Features kept:       {np.sum(np.abs(lasso.coef_) >= 1e-10)} / {X.shape[1]}")
```

---

## How Lasso Zeros Out Weights — The Subgradient View

### Intuition

The absolute value $|w_j|$ is not differentiable at $w_j = 0$. Its derivative is $+1$ when $w_j > 0$, $-1$ when $w_j < 0$, and **undefined** at exactly zero. The subgradient fills that gap: at zero, the subgradient is any value in $[-1, +1]$. This interval is what creates the "dead zone" where the weight can park at exactly zero.

### The Math, Step by Step

The Lasso loss:

$$
L = \text{MSE} + 2\alpha \sum_j |w_j|
$$

The subgradient with respect to a single weight $w_j$:

$$
\frac{\partial L}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha \cdot \partial|w_j|
$$

where the subgradient of the absolute value is:

$$
\partial|w_j| = \begin{cases} +1 & \text{if } w_j > 0 \\ -1 & \text{if } w_j < 0 \\ [-1, +1] & \text{if } w_j = 0 \end{cases}
$$

Now trace the gradient descent update for a weight that's approaching zero from the positive side. At each step:

$$
w_j \leftarrow w_j - \eta \left( \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha \cdot 1 \right)
$$

That $2\alpha$ term is a **constant** kick toward zero — it doesn't weaken as $w_j$ shrinks (unlike Ridge's $2\alpha w_j$ which vanishes). If the data gradient $\partial\text{MSE}/\partial w_j$ is smaller in magnitude than $2\alpha$, the penalty wins every step and the weight is driven to zero.

Once $w_j = 0$, the subgradient can take any value in $[-1, +1]$. The weight stays at zero as long as:

$$
\left| \frac{\partial \text{MSE}}{\partial w_j} \right| \leq 2\alpha
$$

This is the **stationarity condition**: the data gradient must exceed the penalty strength to pull the weight out of zero. Features whose signal is too weak to clear that threshold are permanently zeroed.

### Contrast With Ridge

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty gradient at $w_j > 0$ | $2\alpha w_j$ (proportional) | $2\alpha$ (constant) |
| Penalty gradient at $w_j = 0$ | $0$ (gone) | $[-2\alpha, +2\alpha]$ (still there) |
| Weight reaches zero? | No — force vanishes | Yes — constant force wins |

### In Code

```python
"""
Subgradient descent for Lasso — watch features get zeroed out.

Note: this is pedagogical. Sklearn uses coordinate descent (next section),
which is far more efficient. Subgradient descent for L1 is slow because
the subgradient doesn't smoothly guide you to the kink at zero.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y = make_regression(
    n_samples=200, n_features=10, n_informative=3,
    noise=5.0, random_state=42,
)
X = StandardScaler().fit_transform(X)

alpha: float = 0.5
lr: float = 0.001
n_steps: int = 5000
n_samples, n_features = X.shape

w = np.zeros(n_features)

for step in range(n_steps):
    residuals = X @ w - y
    grad_mse = (2.0 / n_samples) * (X.T @ residuals)

    # Subgradient of L1 penalty: sign(w), with 0 mapped to 0
    # (choosing 0 from the [-1, +1] subgradient at w=0)
    grad_l1 = 2 * alpha * np.sign(w)

    w = w - lr * (grad_mse + grad_l1)

    if step % 1000 == 0:
        loss = np.mean(residuals ** 2) + 2 * alpha * np.sum(np.abs(w))
        n_zero = np.sum(np.abs(w) < 1e-6)
        print(f"Step {step:5d}  loss={loss:10.2f}  "
              f"zeros={n_zero}/{n_features}  "
              f"w={np.round(w, 3)}")

print(f"\nFinal weights: {np.round(w, 4)}")
print(f"Exact zeros:   {np.sum(w == 0.0)}")
print(f"Near zeros:    {np.sum(np.abs(w) < 1e-4)}")
```

**Important caveat:** subgradient descent for L1 is a poor optimizer in practice. The subgradient at zero is discontinuous, so the algorithm tends to oscillate around zero without landing on it exactly. This is why sklearn doesn't use gradient descent for Lasso — it uses coordinate descent, which handles the kink analytically.

---

## Inside Sklearn — How Lasso Is Actually Solved

### Why Not Gradient Descent?

Lasso has no closed-form solution (unlike Ridge), and subgradient descent is slow and imprecise for L1 problems. Sklearn uses **coordinate descent**: optimize one weight at a time while holding all others fixed. For a single coordinate, the L1 problem has an exact analytical solution via the **soft-thresholding operator**. Cycle through all coordinates, repeat until convergence.

### The Soft-Thresholding Operator

When you fix all weights except $w_j$ and minimize the Lasso objective with respect to $w_j$ alone, the solution is:

$$
w_j \leftarrow S\left(\rho_j, \, 2\alpha\right) = \begin{cases} \rho_j - 2\alpha & \text{if } \rho_j > 2\alpha \\ 0 & \text{if } |\rho_j| \leq 2\alpha \\ \rho_j + 2\alpha & \text{if } \rho_j < -2\alpha \end{cases}
$$

where $\rho_j$ is the **partial residual** — the OLS solution for coordinate $j$ if there were no penalty:

$$
\rho_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij} \left( y_i - \sum_{k \neq j} x_{ik} w_k \right)
$$

This is elegant: $\rho_j$ tells you where the data wants to push $w_j$. If that push is weaker than $2\alpha$, the weight is zeroed. If stronger, it's shrunk by exactly $2\alpha$ toward zero.

### Coordinate Descent — `cd_fast.pyx`

The core Lasso solver lives in `sklearn/linear_model/_cd_fast.pyx` (Cython for speed). Here's what it does, reconstructed:

```python
"""
What sklearn's coordinate descent loop does for Lasso.
Reconstructed from sklearn.linear_model._cd_fast.pyx.

The real code is Cython operating on raw memory pointers.
This is the same algorithm in readable NumPy.
"""

from __future__ import annotations

import numpy as np


def soft_threshold(rho: float, lambda_: float) -> float:
    """The soft-thresholding (proximal) operator for L1.

    This is the analytical solution for a single-coordinate
    Lasso subproblem. The dead zone [-lambda_, +lambda_]
    is where weights get zeroed out.
    """
    if rho > lambda_:
        return rho - lambda_
    elif rho < -lambda_:
        return rho + lambda_
    else:
        return 0.0


def coordinate_descent_lasso(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """Lasso via coordinate descent — mirrors sklearn's cd_fast.

    Key insight: for each coordinate j, we compute the partial
    residual (what the OLS solution would be without the penalty),
    then apply soft-thresholding to get the L1-penalized solution.

    Cost per full cycle: O(n·p). Typically converges in ~100 cycles
    for well-scaled data, making total cost O(100·n·p) — much
    cheaper than the O(n·p²) of Ridge's Cholesky solver when p is large.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    residual = y.copy()  # y - X @ w, maintained incrementally

    # Precompute column norms (denominator for coordinate update)
    # sklearn precomputes these once: ||x_j||² / n
    col_norm_sq = np.sum(X ** 2, axis=0) / n_samples

    for iteration in range(max_iter):
        w_old = w.copy()

        for j in range(n_features):
            # Step 1: Temporarily "remove" feature j's contribution
            # to the residual. This is the key efficiency trick —
            # instead of recomputing X @ w from scratch, we
            # incrementally update the residual.
            residual += X[:, j] * w[j]

            # Step 2: Compute the partial residual rho_j
            # This is the OLS solution for coordinate j alone:
            # rho_j = X_j^T @ residual / (n * ||X_j||²)
            rho_j = X[:, j] @ residual / (n_samples * col_norm_sq[j])

            # Step 3: Apply soft-thresholding
            # The penalty threshold is alpha / ||x_j||² (after normalization)
            threshold = alpha / col_norm_sq[j]
            w[j] = soft_threshold(rho_j, threshold)

            # Step 4: Update residual with new w[j]
            residual -= X[:, j] * w[j]

        # Convergence check: max coordinate change
        max_change = np.max(np.abs(w - w_old))
        if max_change < tol:
            print(f"Converged at iteration {iteration}")
            break

    return w


# ── Verify against sklearn ──────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

X, y = make_regression(
    n_samples=200, n_features=10, n_informative=3,
    noise=5.0, random_state=42,
)
X = StandardScaler().fit_transform(X)

w_cd = coordinate_descent_lasso(X, y, alpha=0.5)
w_sklearn = Lasso(alpha=0.5, fit_intercept=False, max_iter=1000).fit(X, y).coef_

print(f"\nOur coordinate descent: {np.round(w_cd, 4)}")
print(f"Sklearn Lasso:          {np.round(w_sklearn, 4)}")
print(f"Max diff:               {np.max(np.abs(w_cd - w_sklearn)):.2e}")

# Show which features were eliminated
print(f"\nOur zeros:    {np.sum(np.abs(w_cd) < 1e-10)}/{X.shape[1]}")
print(f"Sklearn zeros: {np.sum(np.abs(w_sklearn) < 1e-10)}/{X.shape[1]}")
```

### Why Coordinate Descent Beats Gradient Descent for L1

Three reasons:

1. **Exact zeros.** Each coordinate update applies soft-thresholding analytically — the weight is set to *exactly* 0.0, not approximately. Subgradient descent oscillates around zero and never lands on it.

2. **Incremental residual updates.** The `residual += X[:, j] * w[j]` / `residual -= X[:, j] * w[j]` trick means each coordinate update costs $O(n)$, and a full cycle costs $O(n \cdot p)$. No matrix multiplications needed.

3. **Active set strategy.** Sklearn's implementation tracks which features are currently zero (the "active set"). After the first few passes, it only updates nonzero features, making subsequent iterations much cheaper. The Cython code in `cd_fast.pyx` implements this as an inner/outer loop: the outer loop cycles through all features, the inner loop only touches active features.

### The Cython Inner Loop

The actual `cd_fast.pyx` inner loop (simplified) looks roughly like:

```python
"""
Simplified view of the Cython inner loop in cd_fast.pyx.
The real code uses raw C pointers and BLAS calls for speed.
"""

# For each coordinate j:
#   1. w_j_old = w[j]
#   2. Compute partial residual dot product:
#      tmp = X[:, j].T @ residual / n_samples + w[j] * col_norm_sq[j]
#      (this combines steps 1-2 above into one expression)
#   3. Apply soft-thresholding:
#      w[j] = sign(tmp) * max(abs(tmp) - alpha, 0) / col_norm_sq[j]
#   4. Update residual:
#      residual += X[:, j] * (w_j_old - w[j])
#      (only if w[j] actually changed — saves a BLAS call)

# The sign(tmp) * max(abs(tmp) - alpha, 0) pattern IS soft-thresholding.
# It's the same as the three-case formula but written branch-free.
```

The branch-free form `sign(tmp) * max(abs(tmp) - alpha, 0)` is equivalent to the three-case soft-thresholding operator — it's how you write it in one line without conditionals.

---

## The Geometry — Why L1 Produces Corners

### Intuition

The constrained optimization view: Ridge constrains weights to a **hypersphere** ($\sum w_j^2 \leq t$), Lasso to a **hyperdiamond** ($\sum |w_j| \leq t$). The MSE contours are ellipses centered at the OLS solution. The regularized solution is where these ellipses first touch the constraint region.

A sphere has no corners — the ellipse tangentially touches the curved surface, generically at a point where no coordinate is zero. A diamond has corners on the axes — and the ellipse is far more likely to first touch at a corner (where one or more coordinates are exactly zero) than along a face.

In $p$ dimensions, the L1 diamond has $2p$ corners (each on a coordinate axis) and $2^p$ faces. As $p$ grows, the corners dominate the surface. This is why Lasso's sparsity becomes more pronounced in high dimensions.

### In Code

```python
"""
Visualize the L1 diamond vs L2 sphere constraint regions in 2D,
showing why L1 solutions land on axes (corners) while L2 doesn't.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# ── 2D problem so we can plot ───────────────────────────────────────
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] * 0.3 + np.random.randn(100) * 0.5  # correlated
y = 3.0 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(100) * 0.5
X = StandardScaler().fit_transform(X)

# OLS solution (the center of the MSE contours)
w_ols = np.linalg.lstsq(X, y, rcond=None)[0]

# Ridge and Lasso solutions
alpha = 1.5
w_ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X, y).coef_
w_lasso = Lasso(alpha=alpha, fit_intercept=False).fit(X, y).coef_

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, title, w_reg, constraint in [
    (axes[0], "Ridge (L2 sphere)", w_ridge, "sphere"),
    (axes[1], "Lasso (L1 diamond)", w_lasso, "diamond"),
]:
    # MSE contours
    w0_range = np.linspace(-2, 4, 200)
    w1_range = np.linspace(-2, 4, 200)
    W0, W1 = np.meshgrid(w0_range, w1_range)
    MSE = np.array([
        np.mean((y - X @ np.array([w0, w1])) ** 2)
        for w0, w1 in zip(W0.ravel(), W1.ravel())
    ]).reshape(W0.shape)
    ax.contour(W0, W1, MSE, levels=20, alpha=0.4, cmap="coolwarm")

    # Constraint region
    theta = np.linspace(0, 2 * np.pi, 200)
    t = np.sum(np.abs(w_reg))  # constraint radius matching the solution
    if constraint == "sphere":
        cx = t * np.cos(theta)
        cy = t * np.sin(theta)
    else:  # diamond
        cx = t * np.array([1, 0, -1, 0, 1])
        cy = t * np.array([0, 1, 0, -1, 0])
    ax.fill(cx, cy, alpha=0.15, color="green")
    ax.plot(cx, cy, "g-", linewidth=2, label="Constraint region")

    # Solutions
    ax.plot(*w_ols, "ro", markersize=10, label=f"OLS {np.round(w_ols, 2)}")
    ax.plot(*w_reg, "g^", markersize=12,
            label=f"Regularized {np.round(w_reg, 2)}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 4)

plt.tight_layout()
plt.savefig("lasso_vs_ridge_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
print("Lasso w₂ is exactly zero!" if abs(w_lasso[1]) < 1e-10 else
      f"Lasso w₂ = {w_lasso[1]:.6f}")
```

---

## Choosing Alpha With Cross-Validation

### Intuition

`LassoCV` uses coordinate descent along a **regularization path**: it starts from a large $\alpha$ (where all weights are zero) and decreases it, using the previous solution as a warm start for the next. This path-following is much faster than fitting independent models at each alpha because neighboring solutions are similar.

The largest useful alpha — `alpha_max` — is the smallest value that zeros out all weights:

$$
\alpha_{\max} = \frac{1}{2n} \|\mathbf{X}^T \mathbf{y}\|_\infty
$$

This comes directly from the stationarity condition: all weights are zero when no data gradient exceeds the penalty threshold.

### In Code

```python
"""
LassoCV: path-wise alpha selection with warm starts.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = make_regression(
    n_samples=200, n_features=50, n_informative=10,
    noise=10.0, random_state=42,
)
X = StandardScaler().fit_transform(X)

# ── LassoCV: path-wise cross-validation ─────────────────────────────
lasso_cv = LassoCV(cv=5, n_alphas=100, random_state=42)
lasso_cv.fit(X, y)

print(f"Best alpha:   {lasso_cv.alpha_:.6f}")
print(f"Nonzero coefs: {np.sum(np.abs(lasso_cv.coef_) > 1e-10)} / {X.shape[1]}")
print(f"Zeroed coefs:  {np.sum(np.abs(lasso_cv.coef_) < 1e-10)} / {X.shape[1]}")

# ── Verify alpha_max formula ────────────────────────────────────────
alpha_max = np.max(np.abs(X.T @ y)) / (2 * X.shape[0])
lasso_at_max = Lasso(alpha=alpha_max, fit_intercept=False).fit(X, y)
print(f"\nalpha_max (computed):   {alpha_max:.6f}")
print(f"Nonzero at alpha_max:  {np.sum(np.abs(lasso_at_max.coef_) > 1e-10)}")
# Should be 0 — all weights zeroed

# ── MSE at best alpha ──────────────────────────────────────────────
lasso_best = Lasso(alpha=lasso_cv.alpha_, fit_intercept=False)
scores = cross_val_score(lasso_best, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"\n5-fold CV MSE: {-scores.mean():.4f} ± {scores.std():.4f}")

# ── Regularization path: how coefficients evolve ────────────────────
print(f"\nRegularization path — {len(lasso_cv.alphas_)} alphas explored")
print(f"Alpha range: [{lasso_cv.alphas_[-1]:.6f}, {lasso_cv.alphas_[0]:.6f}]")
```

---

## Lasso vs Ridge vs Elastic Net — When to Use What

### Decision Guide

**Use Lasso when** you suspect many features are irrelevant and you want automatic feature selection. Lasso will zero them out.

**Use Ridge when** most features contribute some signal (even weakly) and you want to keep them all with shrunk coefficients. Also better when features are highly correlated — Lasso arbitrarily picks one from a correlated group and zeros the rest.

**Use [[Elastic Net]] when** you want the best of both: L1 for sparsity and L2 for stability with correlated features. Elastic Net's objective is $\text{MSE} + \alpha \cdot l_1\text{\_ratio} \cdot \|\mathbf{w}\|_1 + \alpha \cdot (1 - l_1\text{\_ratio}) \cdot \frac{1}{2}\|\mathbf{w}\|_2^2$. In practice, Elastic Net is the safer default unless you have a strong reason to prefer pure L1 or L2.

### In Code

```python
"""
Side-by-side comparison: Ridge, Lasso, Elastic Net on correlated features.
Lasso's instability with correlated features vs Ridge's stability.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ── Create data with correlated features ────────────────────────────
np.random.seed(42)
n_samples = 200
X_base = np.random.randn(n_samples, 5)

# Duplicate columns 0-2 with small noise → correlated pairs
X_noisy = X_base[:, :3] + np.random.randn(n_samples, 3) * 0.05
X = np.hstack([X_base, X_noisy])  # 8 features, 3 correlated pairs
X = StandardScaler().fit_transform(X)

y = 3 * X_base[:, 0] + 2 * X_base[:, 1] + 1 * X_base[:, 2] + np.random.randn(n_samples) * 2

models = {
    "Ridge":      Ridge(alpha=1.0, fit_intercept=False),
    "Lasso":      Lasso(alpha=0.1, fit_intercept=False),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False),
}

print(f"{'Model':>12s}  {'CV MSE':>10s}  {'Nonzero':>8s}  Coefficients")
print("-" * 80)

for name, model in models.items():
    model.fit(X, y)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
    print(f"{name:>12s}  {-scores.mean():10.4f}  {n_nonzero:8d}  "
          f"{np.round(model.coef_, 3)}")

# Notice: Lasso zeros out one feature from each correlated pair
# (arbitrary choice), while Ridge spreads the weight across both.
# Elastic Net is the compromise.
```

---

## Common Misconceptions

- **"Lasso is just Ridge with a different norm"** — The norms produce qualitatively different behavior. Ridge's L2 gradient vanishes at zero (weights settle at small values); Lasso's L1 subgradient is constant (weights hit exactly zero). They solve different problems: Ridge for shrinkage, Lasso for selection.
- **"Subgradient descent is how Lasso is solved"** — No. Sklearn uses coordinate descent with soft-thresholding, which produces exact zeros and converges much faster. Subgradient descent oscillates around zero and is only useful pedagogically.
- **"Lasso always picks the right features"** — With correlated features, Lasso arbitrarily selects one and zeros the others. The choice is unstable across bootstrap samples. Use [[Elastic Net]] or stability selection if you need reliable feature identification with correlations.
- **"The L1 penalty isn't differentiable, so gradient methods can't work"** — The non-differentiability at zero is actually the *mechanism* that produces zeros, not a bug. Coordinate descent sidesteps it entirely by solving each coordinate analytically. Proximal gradient methods (ISTA, FISTA) also handle it correctly via the proximal operator (which is just soft-thresholding).
- **"More features zeroed = better model"** — Overly aggressive sparsity (high $\alpha$) removes informative features. Use [[Sklearn Cross-Validation]] to find the $\alpha$ that minimizes prediction error, not the one that maximizes sparsity.
