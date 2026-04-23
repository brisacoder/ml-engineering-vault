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
created: 2026-04-22
updated: 2026-04-22
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

## Quick Comparison

| Aspect | `KNNImputer` | `IterativeImputer` |
|---|---|---|
| Strategy | Neighbor mean | Regression per feature |
| Handles non-linearity | Naturally (local) | Depends on estimator |
| Scalability | O(n²) distance matrix | Scales better |
| Iteration | Single pass | Multi-round convergence |
| Experimental? | No | Yes (needs explicit enable) |
