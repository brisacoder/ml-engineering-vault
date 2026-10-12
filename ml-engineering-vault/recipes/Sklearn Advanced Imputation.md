---
type: recipe
title: "Sklearn Advanced Imputation"
libs:
  - scikit-learn
tags:
  - task/feature-engineering
  - task/data-wrangling
related:
  - "[[NumPy Pandas Vectorization]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Matrix Completion and Soft-Impute]]"
created: 2026-04-22
updated: 2026-10-12
---

# Sklearn Advanced Imputation

> Recipes for handling missing numerical data beyond `SimpleImputer` — using neighbor-based and model-based strategies from `sklearn.impute`.

---

## KNNImputer

> Replace each missing value with the mean of its *k*-nearest neighbors (distance computed on all available features).

### Dependencies

```bash
uv add scikit-learn
```

### Steps

```python
import numpy as np
from sklearn.impute import KNNImputer

# Sample data — NaN represents missing values
X = np.array([
    [1.0,  2.0,  np.nan],
    [3.0,  4.0,  3.0],
    [np.nan, 6.0, 5.0],
    [8.0,  8.0,  7.0],
])

imputer = KNNImputer(n_neighbors=2, weights="uniform")
X_filled = imputer.fit_transform(X)
print(X_filled)
```

### Key Parameters

- `n_neighbors` — number of neighbors to consider (default 5). Smaller = more local, larger = smoother.
- `weights` — `"uniform"` (simple mean) or `"distance"` (inverse-distance weighted mean). Distance-weighted is usually better when feature scales vary.
- `metric` — distance metric, default `"nan_euclidean"` which ignores NaN pairs when computing distance.

### When to Use

- Dataset is small-to-medium (KNN computes a full distance matrix — **O(n²)** memory).
- Features have local structure (nearby samples are genuinely similar).
- You want a non-parametric approach that doesn't assume a particular model form.

### Gotchas

- **Scale your features first** — KNN distance is sensitive to magnitude. Pair with `StandardScaler` or `MinMaxScaler` inside a pipeline.
- Doesn't work with categorical features — numerical only.
- Slow on large datasets; consider `IterativeImputer` or domain-specific imputation instead.

### Variations

- Use `weights="distance"` for inverse-distance weighting when closer neighbors should matter more.
- Wrap in a `Pipeline` with scaling to avoid data leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("impute", KNNImputer(n_neighbors=5, weights="distance")),
])
X_filled = pipe.fit_transform(X)
```

---

## IterativeImputer

> Model-based imputation: trains a regression model per feature to predict missing values from the other features, then iterates until convergence. Inspired by the R `mice` package.

### Dependencies

```bash
uv add scikit-learn
```

### Steps

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # required!
from sklearn.impute import IterativeImputer

X = np.array([
    [1.0,  2.0,  np.nan],
    [3.0,  4.0,  3.0],
    [np.nan, 6.0, 5.0],
    [8.0,  8.0,  7.0],
])

imputer = IterativeImputer(max_iter=10, random_state=42)
X_filled = imputer.fit_transform(X)
print(X_filled)
```

### Key Parameters

- `estimator` — any sklearn regressor (default `BayesianRidge`). Swap in `RandomForestRegressor` or `ExtraTreesRegressor` for non-linear relationships.
- `max_iter` — number of imputation rounds (default 10). More iterations = better convergence but slower.
- `random_state` — set for reproducibility (the initial fill and iteration order are stochastic).
- `initial_strategy` — how to fill NaNs before the first round: `"mean"`, `"median"`, `"most_frequent"`, or `"constant"`.
- `imputation_order` — `"ascending"` (fewest missing first), `"descending"`, `"roman"`, `"arabic"`, or `"random"`.

### When to Use

- Features have complex inter-dependencies (e.g., height ↔ weight ↔ BMI).
- You want model-driven imputation that can capture non-linear patterns (with a tree-based estimator).
- Data is too large for KNNImputer's distance matrix.

### Gotchas

- **Still experimental** — you must run `from sklearn.experimental import enable_iterative_imputer` before importing it.
- Default `BayesianRidge` assumes linear relationships. Switch estimator for non-linear data.
- Numerical features only (same as `KNNImputer`).
- Can be slow with many features × many iterations × expensive estimator.

### Variations

- Use a tree-based estimator for non-linear imputation:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=100, random_state=42),
    max_iter=10,
    random_state=42,
)
X_filled = imputer.fit_transform(X)
```

- Multiple imputation (run several times with different seeds, pool the results) for uncertainty estimation:

```python
imputations: list[np.ndarray] = []
for seed in range(5):
    imp = IterativeImputer(max_iter=10, random_state=seed)
    imputations.append(imp.fit_transform(X))

# Average across imputations
X_pooled = np.mean(imputations, axis=0)
```

---

## Low-Rank Imputation - SoftImputer

> Fill holes with a low-rank matrix: alternate "fill holes, SVD, subtract $\lambda$ from every singular value, rebuild" until the holes stop moving. scikit-learn has no built-in low-rank imputer, so this is a small transformer you own. Theory, stopping rule, choosing $\lambda$: [[Matrix Completion and Soft-Impute]].

### Dependencies

```bash
uv add scikit-learn numpy
```

### Steps

```python
"""
A scikit-learn-compatible SoftImputer.
fit(X):       standardize on observed cells, run soft-impute, keep the
              learned right singular vectors V_r and shrink ratios
              (s_i - lam) / s_i.
transform(X): "fold in" new rows. At soft-impute's fixed point
              Z = U diag(s - lam) V^T = F V diag((s - lam)/s) V^T,
              so each row satisfies z = f V diag(ratio) V^T with V fixed.
              Iterating that map on the new rows' holes is a contraction
              (every ratio < 1), vectorized over all rows at once.
Observed cells are always returned unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_wine
from sklearn.impute import SimpleImputer


class SoftImputer(TransformerMixin, BaseEstimator):
    def __init__(self, lam: float = 4.0, max_iter: int = 500, tol: float = 1e-6) -> None:
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def _loop(self, Xs: np.ndarray, step) -> np.ndarray:
        observed = ~np.isnan(Xs)
        Z = np.zeros_like(Xs)
        for _ in range(self.max_iter):
            Z_new = step(np.where(observed, Xs, Z))
            done = np.sum((Z_new - Z) ** 2) < self.tol * max(np.sum(Z ** 2), 1e-12)
            Z = Z_new
            if done:
                break
        return np.where(observed, Xs, Z)

    def fit(self, X: np.ndarray, y=None) -> "SoftImputer":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        Xs = (X - self.mean_) / self.scale_

        def svt(F: np.ndarray) -> np.ndarray:
            U, s, Vt = np.linalg.svd(F, full_matrices=False)
            self._s, self._Vt = s, Vt                    # keep the last SVD
            return (U * np.maximum(s - self.lam, 0.0)) @ Vt

        self._loop(Xs, svt)
        keep = self._s > self.lam
        self.components_ = self._Vt[keep]                # (r, p)
        self.shrink_ratio_ = (self._s[keep] - self.lam) / self._s[keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.scale_
        V, ratio = self.components_, self.shrink_ratio_
        filled = self._loop(Xs, lambda F: ((F @ V.T) * ratio) @ V)
        return filled * self.scale_ + self.mean_


# --- Fit on 140 wines, impute 38 unseen wines with 15% of cells missing ---
rng = np.random.default_rng(0)
X = load_wine().data
perm = rng.permutation(len(X))
train, test = X[perm[:140]], X[perm[140:]]
train_missing = np.where(rng.random(train.shape) < 0.15, np.nan, train)
test_holes = rng.random(test.shape) < 0.15
test_missing = np.where(test_holes, np.nan, test)

sd = np.nanstd(train_missing, axis=0)
def rmse_sd_units(X_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((((X_hat - test) / sd)[test_holes]) ** 2)))

mean_imp = SimpleImputer(strategy="mean").fit(train_missing)
print(f"{'SimpleImputer(mean)':<24}test RMSE (sd units) = {rmse_sd_units(mean_imp.transform(test_missing)):.3f}")
for lam in (2.0, 4.0, 8.0):
    soft = SoftImputer(lam=lam).fit(train_missing)
    X_hat = soft.transform(test_missing)
    assert np.allclose(X_hat[~test_holes], test[~test_holes])
    print(f"{f'SoftImputer(lam={lam:g})':<24}test RMSE (sd units) = {rmse_sd_units(X_hat):.3f}"
          f"   rank = {soft.components_.shape[0]}")
```

```text
SimpleImputer(mean)     test RMSE (sd units) = 1.209
SoftImputer(lam=2)      test RMSE (sd units) = 0.881   rank = 13
SoftImputer(lam=4)      test RMSE (sd units) = 0.880   rank = 12
SoftImputer(lam=8)      test RMSE (sd units) = 0.942   rank = 6
```

Why fold-in is the right `transform`: fitting on a matrix and then transforming that same matrix reproduces the transductive soft-impute solution (checked to about $10^{-3}$ raw units with a tight `tol`), so training rows and new rows are imputed by the same rule.

### Key Parameters

- `lam`: threshold on singular values of the standardized matrix. Rank is an output (count of $\sigma_i>\lambda$). Express it relative to $\sigma_1$ of the zero-filled matrix; $\lambda\ge\sigma_1$ imputes the column means.
- `tol`: on $\lVert Z_{new}-Z\rVert^2/\lVert Z\rVert^2$, the change of the iterate. Never test $\lVert Z_{new}-F\rVert$, which has a floor of $r\lambda^2$.
- `max_iter`: small $\lambda$ converges slowly; when tuning over a grid, warm-start from the previous $\lambda$'s solution.

### When to Use

- Columns are correlated through a few latent factors (chemistry panels, survey batteries, ratings, sensor arrays).
- Many columns, moderate missingness, and you want one model of the whole table rather than one regressor per column.
- Wide data where `KNNImputer`'s distances become unreliable.

### Gotchas

- **Standardize first** (done inside `fit` here). The SVD chases squared error, so a column in big units owns $\sigma_1$.
- **Shrinkage bias**: imputed values are pulled toward the column means. Choose `lam` by masked validation, and consider debiasing (refit the kept singular values by least squares on the observed cells).
- **Missing not at random** breaks it, as it breaks every imputer that only looks at observed values.

### Variations

Compare against the built-in imputers inside one leakage-free pipeline:

```python
"""
SoftImputer next to the built-in imputers, each inside the same Pipeline
(scale -> impute -> classify) and scored by 5-fold CV on Wine with 15% of
the cells knocked out. Uses the SoftImputer class defined above.
Leakage-free by construction: every imputer is fitted on the training
folds only.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_wine
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required before import)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
rng = np.random.default_rng(0)
X_missing = np.where(rng.random(X.shape) < 0.15, np.nan, X)

imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "knn (k=5)": KNNImputer(n_neighbors=5),
    "iterative": IterativeImputer(max_iter=20, random_state=0),
    "soft (lam=4)": SoftImputer(lam=4.0),
}
for name, imputer in imputers.items():
    pipe = make_pipeline(StandardScaler(), imputer, LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipe, X_missing, y, cv=5)
    print(f"{name:<14} accuracy = {scores.mean():.3f} +/- {scores.std():.3f}")
```

---

## Quick Comparison

| Aspect | `KNNImputer` | `IterativeImputer` | `SoftImputer` (custom) |
|---|---|---|---|
| Strategy | Neighbor mean | Regression per feature | Low-rank SVD with shrunk singular values |
| Handles non-linearity | Naturally (local) | Depends on estimator | No (linear, global) |
| Scalability | O(n²) distance matrix | Scales better | One SVD per iteration; sparse tricks at scale |
| Iteration | Single pass | Multi-round convergence | Multi-round, convex (unique answer) |
| Experimental? | No | Yes (needs explicit enable) | Not in sklearn; your own class |
