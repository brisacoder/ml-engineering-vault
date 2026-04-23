---
type: recipe
title: "Sklearn Feature Scaling"
libs:
  - scikit-learn
tags:
  - task/feature-engineering
  - task/data-wrangling
related:
  - "[[Sklearn Advanced Imputation]]"
  - "[[Sklearn validate_data]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn Feature Scaling

> Recipes for scaling and transforming numerical features — from standard scaling to taming heavy-tailed distributions. Scaling should generally happen **after** imputation but **before** model training.

---

## Taming Heavy-Tailed Features Before Scaling

> When a feature's distribution has a **heavy tail** (values far from the mean are not exponentially rare), both min-max scaling and standardization will squash most values into a tiny range. Transform first to make the distribution roughly symmetrical, *then* scale.

### The Ladder of Transformations

Pick the transform based on how heavy the tail is:

| Tail severity | Transform | Code |
|---|---|---|
| Mild right skew | Square root | `np.sqrt(X)` |
| Moderate skew | Power (0 < p < 1) | `np.power(X, 0.3)` |
| Heavy tail / power-law | Logarithm | `np.log1p(X)` |

**Example**: the `population` feature in housing data roughly follows a power law — districts with 10,000 inhabitants are only 10× less frequent than districts with 1,000 (not exponentially less frequent). Taking `log(population)` produces a near-Gaussian distribution.

### Steps

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Log-transform heavy-tailed columns, then standard scale everything
log_transform = FunctionTransformer(np.log1p, validate=True)

preprocessor = ColumnTransformer([
    ("log_heavy", Pipeline([
        ("log", log_transform),
        ("scale", StandardScaler()),
    ]), ["population", "total_rooms"]),   # heavy-tailed columns
    ("scale_rest", StandardScaler(), ["median_income", "housing_median_age"]),
])
```

### Gotchas

- `np.log1p` (i.e. log(1 + x)) avoids log(0) = -∞ for zero values.
- These transforms only work on **positive** features. For features with negatives, use `PowerTransformer` (see below).
- Always transform **before** scaling — scaling a heavy tail just compresses the bulk of data near zero.

---

## PowerTransformer (Automatic Gaussianization)

> Sklearn's `PowerTransformer` automatically finds the best power transform to make each feature as Gaussian as possible. Handles the "which transform?" decision for you.

### Steps

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox: only works with strictly positive data
pt_boxcox = PowerTransformer(method="box-cox", standardize=True)

# Yeo-Johnson: works with positive, negative, and zero values
pt_yeojohnson = PowerTransformer(method="yeo-johnson", standardize=True)

X_transformed = pt_yeojohnson.fit_transform(X)
```

### Key Parameters

- `method` — `"yeo-johnson"` (default, handles any sign) or `"box-cox"` (strictly positive data only, sometimes fits better).
- `standardize` — if `True` (default), zero-mean unit-variance scaling is applied after the power transform.

### When to Use

- You have several skewed features and don't want to manually pick sqrt vs. log vs. power for each.
- You want a single transformer that both reshapes and standardizes.

### Gotchas

- Box-Cox will raise an error if any value ≤ 0. Use Yeo-Johnson as the safe default.
- The learned lambda parameter can overfit on small samples — inspect `pt.lambdas_` after fitting.

---

## StandardScaler

> Centers to mean=0, scales to std=1. The go-to scaler for algorithms that assume Gaussian-like inputs (linear models, SVMs, neural nets).

### Steps

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use train stats!
```

### When to Use

- Features are roughly symmetrical (bell-shaped) — if they're skewed, transform first.
- Algorithm uses distance or gradient-based optimization.

### Gotchas

- Sensitive to outliers (they inflate mean/std). Consider `RobustScaler` if outliers are present.
- Always `fit` on training data only, then `transform` both train and test.

---

## MinMaxScaler

> Rescales features to a fixed range, default [0, 1].

### Steps

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

### When to Use

- Algorithm expects bounded inputs (e.g., neural nets with sigmoid outputs, image pixel pipelines).
- You want to preserve zero entries in sparse data (use `MaxAbsScaler` instead for truly sparse matrices).

### Gotchas

- Extremely sensitive to outliers — a single extreme value compresses everything else.
- Same train-only fitting rule as `StandardScaler`.

---

## RobustScaler

> Uses median and IQR instead of mean and std — robust to outliers.

### Steps

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(quantile_range=(25.0, 75.0))
X_scaled = scaler.fit_transform(X_train)
```

### When to Use

- Data has significant outliers you want to keep (not remove).
- You want centering + scaling without letting outliers dominate.

---

## Scaling the Target (Labels)

> Features aren't the only thing that can be heavy-tailed — the **target variable** may need transforming too. If you log-transform the target, the model predicts log(y), so you must `inverse_transform` predictions back to the original scale.

### Steps

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Scale the target with its own scaler
target_scaler = StandardScaler()
# .to_frame() because StandardScaler expects 2D input
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# Train on raw feature(s), scaled target
model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)

# Predict → inverse_transform back to original scale
some_new_data = housing[["median_income"]].iloc[:5]
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
```

### Key Ideas

- Use a **separate scaler instance** for the target — don't reuse the feature scaler.
- Most sklearn transformers expose `inverse_transform()`, so you can always map predictions back to the original scale.
- For heavy-tailed targets (e.g., house prices, population), log-transform before scaling: the model predicts log(y), then `np.expm1()` or `inverse_transform()` recovers y.

### With TransformedTargetRegressor (cleaner)

Sklearn provides a wrapper that handles the forward/inverse transform automatically:

```python
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    transformer=StandardScaler(),
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # already in original scale
```

For log-transforming a heavy-tailed target:

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p,          # forward: log(1 + y)
    inverse_func=np.expm1,  # inverse: exp(pred) - 1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # back in original scale
```

### Gotchas

- **Forgetting `inverse_transform`** is the #1 mistake — your metrics (RMSE, MAE) will be meaningless if computed on scaled predictions.
- `StandardScaler` expects 2D input, so convert a Pandas Series with `.to_frame()` or `.values.reshape(-1, 1)`.
- `TransformedTargetRegressor` eliminates the manual inverse step entirely — prefer it.

---

## Quick Comparison

| Scaler | Formula | Outlier robust? | Output range |
|---|---|---|---|
| `StandardScaler` | (x − μ) / σ | No | Unbounded |
| `MinMaxScaler` | (x − min) / (max − min) | No | [0, 1] |
| `RobustScaler` | (x − median) / IQR | Yes | Unbounded |
| `PowerTransformer` | Gaussianize + standardize | Moderate | Unbounded |

## Decision Flow

1. **Heavy tail or strong skew?** → `PowerTransformer(method="yeo-johnson")` or manual log/sqrt first
2. **Outliers you want to keep?** → `RobustScaler`
3. **Need bounded [0,1] range?** → `MinMaxScaler`
4. **Otherwise** → `StandardScaler`
