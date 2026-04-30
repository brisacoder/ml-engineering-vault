---
type: concept
title: "Ridge Regression"
libs:
  - scikit-learn
  - numpy
tags:
  - concept/regularization
  - concept/optimization
related:
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Lasso Regression]]"
created: 2026-04-29
updated: 2026-04-29
---

# Ridge Regression

> Ridge regression (L2 regularization) adds a penalty proportional to the sum of squared weights to the loss function. This shrinks coefficients toward zero without eliminating them, controlling model complexity and reducing overfitting — especially when features are correlated or when you have more features than samples.

---

## The Ridge Objective

### Intuition

Ordinary least squares (OLS) minimizes only the prediction error. When features are correlated, small changes in the data can cause wild swings in the fitted coefficients — the model is unstable. Ridge adds a "tax" on large weights: you can still use them, but you pay a cost proportional to their squared magnitude. The optimizer finds the sweet spot between fitting the data and keeping weights small.

### Formal Definition

The Ridge loss function augments the standard MSE with an L2 penalty:

$$
L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \mathbf{w})^2 + \alpha \sum_{j=1}^{p} w_j^2
$$

where $\alpha > 0$ controls the regularization strength. Larger $\alpha$ means heavier penalty, smaller coefficients.

The closed-form solution is:

$$
\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

That $\alpha \mathbf{I}$ term is what makes the matrix invertible even when $\mathbf{X}^T \mathbf{X}$ is singular (hence the name "ridge" — you're adding a ridge along the diagonal).

### In Code

```python
"""
Ridge regression: closed-form vs sklearn, showing coefficient shrinkage.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ── Generate correlated features ────────────────────────────────────
X, y, true_coef = make_regression(
    n_samples=100,
    n_features=10,
    n_informative=5,
    noise=10.0,
    coef=True,
    random_state=42,
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Closed-form solution ────────────────────────────────────────────
alpha = 1.0
n_features = X_scaled.shape[1]
I = np.eye(n_features)

w_closed = np.linalg.solve(
    X_scaled.T @ X_scaled + alpha * I,
    X_scaled.T @ y,
)

# ── Sklearn Ridge ───────────────────────────────────────────────────
ridge = Ridge(alpha=alpha, fit_intercept=False)
ridge.fit(X_scaled, y)

print("Closed-form coefficients:")
print(np.round(w_closed, 4))
print("\nSklearn coefficients:")
print(np.round(ridge.coef_, 4))
print(f"\nMax difference: {np.max(np.abs(w_closed - ridge.coef_)):.2e}")
```

---

## How Ridge Shrinks Weights — The Gradient Descent View

### Intuition

The best way to understand *why* Ridge pushes weights toward zero (but not *to* zero) is to trace a single gradient descent step. The L2 penalty adds a term to the gradient that is proportional to the current weight value. This creates a multiplicative decay at every step — a constant fraction of the weight is shaved off before the data gradient even acts.

### The Math, Step by Step

Start with the Ridge loss:

$$
L = \text{MSE} + \alpha \sum_j w_j^2
$$

Take the partial derivative with respect to a single weight $w_j$:

$$
\frac{\partial L}{\partial w_j} = \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha \, w_j
$$

The gradient descent update rule with learning rate $\eta$:

$$
w_j \leftarrow w_j - \eta \left( \frac{\partial \text{MSE}}{\partial w_j} + 2\alpha \, w_j \right)
$$

Rearranging:

$$
w_j \leftarrow w_j \cdot (1 - 2\eta\alpha) - \eta \cdot \frac{\partial \text{MSE}}{\partial w_j}
$$

The factor $(1 - 2\eta\alpha)$ is the key. It is a number slightly less than 1 (for small $\eta$ and $\alpha$), so every update **multiplicatively decays** the weight before the data gradient pushes it toward the best-fit value.

**Why weights don't reach zero:** As $w_j$ shrinks, the penalty gradient $2\alpha w_j$ also shrinks. Eventually the shrinkage force and the data-fitting force reach equilibrium — the weight settles at a value smaller in magnitude than the OLS solution, but not zero.

**What if $2\eta\alpha \to 0$?** The factor approaches 1, meaning no shrinkage at all — exactly what you'd expect with no regularization.

### In Code

```python
"""
Manual gradient descent for Ridge regression — watch the weight
shrinkage factor (1 - 2*lr*alpha) in action.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# ── Setup ────────────────────────────────────────────────────────────
X, y = make_regression(
    n_samples=200, n_features=5, noise=10.0, random_state=42
)
X = StandardScaler().fit_transform(X)

alpha: float = 1.0
lr: float = 0.01
n_steps: int = 500
n_samples, n_features = X.shape

w = np.zeros(n_features)
shrinkage_factor = 1 - 2 * lr * alpha

print(f"Shrinkage factor per step: {shrinkage_factor:.4f}")
print(f"  (each step, weights are multiplied by {shrinkage_factor:.4f}"
      f" before the data gradient acts)\n")

# ── Gradient descent ─────────────────────────────────────────────────
history: list[tuple[int, float, float]] = []

for step in range(n_steps):
    residuals = X @ w - y                          # (n_samples,)
    grad_mse = (2 / n_samples) * (X.T @ residuals) # (n_features,)
    grad_penalty = 2 * alpha * w                    # (n_features,)

    w = w - lr * (grad_mse + grad_penalty)
    # Equivalent to: w = w * shrinkage_factor - lr * grad_mse

    if step % 100 == 0 or step == n_steps - 1:
        mse = np.mean(residuals ** 2)
        penalty = alpha * np.sum(w ** 2)
        history.append((step, mse, penalty))

print(f"{'Step':>5s}  {'MSE':>10s}  {'Penalty':>10s}  {'Total':>10s}")
for step, mse, pen in history:
    print(f"{step:5d}  {mse:10.4f}  {pen:10.4f}  {mse + pen:10.4f}")

print(f"\nFinal weights: {np.round(w, 4)}")
print(f"Weight norms — L2: {np.linalg.norm(w):.4f}")
```

---

## Ridge vs Lasso — Why Ridge Can't Zero Out Weights

### Intuition

The difference comes down to what the penalty gradient looks like as the weight approaches zero:

- **Ridge (L2):** penalty gradient is $2\alpha w_j$ — shrinks proportionally with the weight. As the weight approaches zero, the penalty force vanishes. The weight settles at a small nonzero value.
- **Lasso (L1):** penalty gradient is $\alpha \cdot \text{sign}(w_j)$ — a constant push regardless of magnitude. Even a tiny weight feels the full force of the penalty, so it gets driven all the way to exactly zero.

This is why [[Lasso Regression]] performs feature selection (some coefficients become exactly zero) while Ridge keeps all features in the model with reduced coefficients.

### In Code

```python
"""
Compare Ridge vs Lasso coefficient paths as alpha increases.
Ridge shrinks smoothly; Lasso zeros out features.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

X, y = make_regression(
    n_samples=200, n_features=10, n_informative=3,
    noise=5.0, random_state=42,
)
X = StandardScaler().fit_transform(X)

alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

print(f"{'alpha':>7s}  {'Ridge zeros':>12s}  {'Lasso zeros':>12s}  "
      f"{'Ridge L2':>10s}  {'Lasso L2':>10s}")
print("-" * 60)

for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False).fit(X, y)
    lasso = Lasso(alpha=a, fit_intercept=False, max_iter=10000).fit(X, y)

    r_zeros = np.sum(np.abs(ridge.coef_) < 1e-6)
    l_zeros = np.sum(np.abs(lasso.coef_) < 1e-6)
    r_norm = np.linalg.norm(ridge.coef_)
    l_norm = np.linalg.norm(lasso.coef_)

    print(f"{a:7.2f}  {r_zeros:12d}  {l_zeros:12d}  "
          f"{r_norm:10.4f}  {l_norm:10.4f}")
```

---

## Choosing Alpha With Cross-Validation

### Intuition

`RidgeCV` uses efficient leave-one-out cross-validation (LOO-CV) via a closed-form shortcut — it doesn't actually refit the model $n$ times. This makes it essentially free compared to a single Ridge fit. Always prefer `RidgeCV` over manually grid-searching alpha with `GridSearchCV` for Ridge.

### In Code

```python
"""
RidgeCV: efficient built-in alpha selection via LOO-CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

X, y = make_regression(
    n_samples=200, n_features=20, n_informative=5,
    noise=10.0, random_state=42,
)
X = StandardScaler().fit_transform(X)

# ── RidgeCV picks the best alpha automatically ──────────────────────
alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, scoring="neg_mean_squared_error")
ridge_cv.fit(X, y)

print(f"Best alpha: {ridge_cv.alpha_:.4f}")
print(f"Number of nonzero coefficients: {np.sum(np.abs(ridge_cv.coef_) > 1e-6)}")

# ── Verify with manual cross-validation ─────────────────────────────
ridge_manual = Ridge(alpha=ridge_cv.alpha_)
scores = cross_val_score(ridge_manual, X, y, cv=5, scoring="neg_mean_squared_error")
print(f"\n5-fold CV MSE with best alpha: {-scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Inside Sklearn — How Ridge Is Actually Solved

### Solver Dispatch Logic

When you call `Ridge(solver="auto").fit(X, y)`, sklearn picks a solver based on your data shape and sparsity. The `_BaseRidge.fit()` method centers `X` and `y` (subtracting means) so the intercept is never penalized, then dispatches to one of these internal functions.

The `auto` heuristic: sparse `X` → `sparse_cg`; dense with `n_samples >= n_features` → `cholesky`; dense with `n_features > n_samples` → `lsqr`.

### Cholesky Solver — `_solve_cholesky`

This is the default for dense data where `n_samples >= n_features`. It implements the closed-form solution, but **never explicitly inverts** the matrix. Instead it uses Cholesky factorization, which is ~2x faster than LU because $\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}$ is guaranteed symmetric positive definite (for $\alpha > 0$).

```python
"""
What sklearn's _solve_cholesky does under the hood.
Reconstructed from sklearn.linear_model._ridge source.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

def solve_cholesky(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Solve Ridge via Cholesky factorization of the Gram matrix.

    This is what sklearn dispatches to when solver='cholesky'.
    Cost: O(n·p² + p³) — dominated by Gram matrix formation.
    """
    # Step 1: Form the Gram matrix  X^T X  — O(n·p²)
    A = X.T @ X

    # Step 2: Add alpha to the diagonal.
    # sklearn uses A.flat[::n_features+1] += alpha as a stride trick
    # to hit exactly the diagonal elements without allocating np.eye.
    n_features = X.shape[1]
    A.flat[::n_features + 1] += alpha  # equivalent to A += alpha * I

    # Step 3: Compute X^T y
    Xy = X.T @ y

    # Step 4: Cholesky factorize A = L L^T (lower=False means upper)
    # Then solve via forward/back substitution — O(p²), not O(p³) inversion
    L = linalg.cholesky(A, lower=False)
    coef = linalg.cho_solve((L, False), Xy)

    return coef


# ── Verify against sklearn ──────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=42)
X = StandardScaler().fit_transform(X)

w_cholesky = solve_cholesky(X, y, alpha=1.0)
w_sklearn = Ridge(alpha=1.0, fit_intercept=False, solver="cholesky").fit(X, y).coef_

print(f"Max diff from sklearn: {np.max(np.abs(w_cholesky - w_sklearn)):.2e}")
# Should be ~1e-14 (machine epsilon)
```

**Key detail:** the `A.flat[::n_features + 1] += alpha` line. Instead of allocating a `p×p` identity matrix and multiplying, sklearn uses NumPy's flat iterator stride to add `alpha` directly to the diagonal elements in-place. The stride `n_features + 1` hits indices `0, p+1, 2(p+1), ...` which are exactly the diagonal.

### SVD Solver — `_solve_svd`

The numerically most stable path. Decomposes $\mathbf{X} = \mathbf{U} \mathbf{S} \mathbf{V}^T$ and applies a per-singular-value shrinkage factor.

```python
"""
What sklearn's _solve_svd does under the hood.
The shrinkage happens per singular value: s_i / (s_i² + alpha).
"""

from __future__ import annotations

import numpy as np

def solve_svd(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Solve Ridge via SVD with per-singular-value damping.

    Compare to OLS where each d_i = 1/s_i.
    Ridge replaces that with d_i = s_i / (s_i² + alpha), which
    continuously shrinks components with small singular values
    toward zero — this IS the variance reduction mechanism.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Ridge shrinkage factors — the heart of it
    # OLS would use d = 1/s; Ridge damps small singular values
    d = s / (s ** 2 + alpha)

    # coef = V @ diag(d) @ U^T @ y
    # sklearn computes this without forming the diagonal matrix:
    coef = (Vt.T * d) @ (U.T @ y)

    return coef


# ── Verify ──────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=42)
X = StandardScaler().fit_transform(X)

w_svd = solve_svd(X, y, alpha=1.0)
w_sklearn = Ridge(alpha=1.0, fit_intercept=False, solver="svd").fit(X, y).coef_

print(f"Max diff from sklearn: {np.max(np.abs(w_svd - w_sklearn)):.2e}")
```

**Why this matters geometrically:** each singular value $s_i$ represents the "spread" of the data along one principal direction. OLS divides by $s_i$, amplifying noise in directions with little data spread. Ridge's $s_i / (s_i^2 + \alpha)$ factor approaches $1/s_i$ for large $s_i$ (strong signal preserved) and approaches $s_i / \alpha \approx 0$ for small $s_i$ (noise suppressed).

### LSQR Solver — The Augmented Least Squares Trick

For sparse or very high-dimensional data, sklearn reformulates Ridge as an augmented OLS problem and solves it with an iterative Krylov method that never forms $\mathbf{X}^T \mathbf{X}$.

```python
"""
What sklearn's _solve_lsqr does: augmented least squares via scipy.

Ridge:  min ||Xw - y||² + alpha·||w||²

Is equivalent to:

        ┌  X       ┐       ┌ y ┐
  min  ││          │ w  -  │   │ ||²
        │√α · I    │       │ 0 │
        └          ┘       └   ┘

scipy.sparse.linalg.lsqr handles this with damp=√alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import lsqr

def solve_lsqr(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Solve Ridge via LSQR with damping.

    lsqr minimizes ||Xw - y||² + damp²·||w||²
    Setting damp = √alpha gives the Ridge solution.
    Never forms X^T X — ideal for sparse/high-dim data.
    """
    result = lsqr(X, y, damp=np.sqrt(alpha))
    coef = result[0]
    return coef


# ── Verify ──────────────────────────────────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=42)
X = StandardScaler().fit_transform(X)

w_lsqr = solve_lsqr(X, y, alpha=1.0)
w_sklearn = Ridge(alpha=1.0, fit_intercept=False, solver="lsqr").fit(X, y).coef_

print(f"Max diff from sklearn: {np.max(np.abs(w_lsqr - w_sklearn)):.2e}")
```

### SGD Path — Where Weight Decay Actually Happens

`SGDRegressor(penalty='l2')` lives in a separate Cython module (`_sgd_fast.pyx`). This is where the gradient descent shrinkage factor from the math section above shows up in real sklearn code. The per-sample inner loop does:

```python
"""
Simplified version of sklearn's SGD inner loop for L2 penalty.
The actual code is in Cython (_sgd_fast.pyx) for speed.
"""

from __future__ import annotations

import numpy as np

def sgd_ridge_one_epoch(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    alpha: float,
    eta: float,
) -> np.ndarray:
    """One epoch of SGD with L2 penalty — mirrors sklearn's Cython loop.

    The key line: w *= (1 - eta * alpha)
    This IS the L2 penalty expressed as weight decay.
    It's the same (1 - 2*eta*alpha) factor from full-batch GD,
    but sklearn's SGD convention uses alpha without the factor of 2
    (absorbed into the alpha hyperparameter).
    """
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    for i in indices:
        # Step 1: Weight decay — L2 shrinkage BEFORE the gradient step
        w *= (1 - eta * alpha)

        # Step 2: Compute prediction error for this sample
        y_pred = X[i] @ w
        gradient = (y_pred - y[i])

        # Step 3: SGD update from data gradient
        w -= eta * gradient * X[i]

    return w


# ── Verify against sklearn SGDRegressor ─────────────────────────────
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=500, n_features=5, noise=5.0, random_state=42)
X = StandardScaler().fit_transform(X)

sgd = SGDRegressor(
    loss="squared_error",
    penalty="l2",
    alpha=0.01,
    learning_rate="constant",
    eta0=0.001,
    max_iter=1000,
    random_state=42,
    fit_intercept=False,
)
sgd.fit(X, y)
print(f"SGDRegressor coefs: {np.round(sgd.coef_, 4)}")

# Compare with exact Ridge solution
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.01, fit_intercept=False).fit(X, y)
print(f"Ridge exact coefs:  {np.round(ridge.coef_, 4)}")
print(f"Max diff: {np.max(np.abs(sgd.coef_ - ridge.coef_)):.4f}")
# Won't be exact — SGD is stochastic — but should be close
```

**The `w *= (1 - eta * alpha)` line is the entire L2 penalty.** No explicit gradient computation of the penalty term, no addition — just multiplicative weight decay applied before each sample's gradient step. This is mathematically equivalent to the full-batch $(1 - 2\eta\alpha)$ factor but uses sklearn's convention where the factor of 2 is absorbed into `alpha`.

---

## Common Misconceptions

- **"Ridge drives weights to zero"** — No. The L2 penalty gradient is proportional to the weight itself ($2\alpha w_j$), so the shrinkage force vanishes as the weight approaches zero. Weights reach equilibrium at small but nonzero values. Only [[Lasso Regression]] (L1) can zero out weights because its penalty gradient is constant ($\alpha \cdot \text{sign}(w_j)$).
- **"You don't need to scale features for Ridge"** — Wrong. The penalty $\alpha \sum w_j^2$ treats all weights equally. If features have different scales, the penalty disproportionately penalizes weights on small-scale features. Always use [[Sklearn Feature Scaling]] before Ridge.
- **"Larger alpha means better generalization"** — Not necessarily. Too large an $\alpha$ kills the signal along with the noise — you underfit. Use `RidgeCV` or [[Sklearn Cross-Validation]] to find the right balance.
- **"Ridge is just OLS with a bigger matrix"** — Conceptually close (the closed form adds $\alpha \mathbf{I}$ to $\mathbf{X}^T \mathbf{X}$), but the geometric interpretation is different: Ridge constrains the solution to lie within a hypersphere around the origin in weight space. The OLS solution is projected onto this sphere.
