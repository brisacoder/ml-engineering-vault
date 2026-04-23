---
type: recipe
title: "Sklearn validate_data"
libs:
  - scikit-learn
tags:
  - task/data-wrangling
  - concept/architecture
related:
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Advanced Imputation]]"
  - "[[Sklearn Datasets]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn validate_data

> How to use `sklearn.utils.validation.validate_data` to validate incoming data in custom estimators — both at fit time and at predict/transform time. Introduced as a **public function in scikit-learn 1.6**, replacing the private `estimator._validate_data()` method.

---

## Full Working Example — Custom Classifier on Iris

> A minimal custom classifier that uses `validate_data` correctly during `fit()` (with `reset=True`) and during `predict()` (with `reset=False`). Uses the Iris dataset so you can run it end-to-end.

### Dependencies

```bash
uv add scikit-learn numpy
```

### Complete Code

```python
"""
Demonstrates sklearn.utils.validation.validate_data on new data.

validate_data(estimator, X, ..., reset=True)   → called during fit()
    - Validates X (dtype, shape, NaNs, sparsity, etc.)
    - Records n_features_in_ and feature_names_in_ on the estimator

validate_data(estimator, X, ..., reset=False)   → called during predict/transform
    - Validates X against the contract stored at fit time
    - Raises ValueError if n_features doesn't match
    - Does NOT overwrite the stored metadata

sklearn >= 1.6 required (the function is public since 1.6;
prior versions only had the private estimator._validate_data method).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data


class CentroidClassifier(BaseEstimator, ClassifierMixin):
    """Nearest-centroid classifier — dead simple, but enough to show the
    validate_data lifecycle."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> CentroidClassifier:
        # reset=True: store n_features_in_ (and feature_names_in_ if DataFrame)
        X, y = validate_data(self, X, y, reset=True)

        self.classes_ = np.unique(y)
        self.centroids_ = np.array([
            X[y == c].mean(axis=0) for c in self.classes_
        ])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        # reset=False: validate new data against the fit-time contract
        # — raises if n_features differs from training
        X = validate_data(self, X, reset=False)

        distances = np.linalg.norm(
            X[:, np.newaxis] - self.centroids_, axis=2
        )
        return self.classes_[distances.argmin(axis=1)]


# ── Load Iris and split ──────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,
)

clf = CentroidClassifier()
clf.fit(X_train, y_train)

# ── Validate new data automatically on predict ───────────────────────
predictions = clf.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
print(f"n_features_in_: {clf.n_features_in_}")  # 4 (set by reset=True)

# ── What happens with bad data ───────────────────────────────────────

# Wrong number of features → ValueError
try:
    clf.predict(X_test[:, :2])  # only 2 of 4 features
except ValueError as e:
    print(f"\nFeature mismatch caught: {e}")

# Calling predict before fit → NotFittedError
try:
    CentroidClassifier().predict(X_test)
except Exception as e:
    print(f"\nNot fitted caught: {e}")
```

### Expected Output

```
Accuracy: 92.00%
n_features_in_: 4

Feature mismatch caught: X has 2 features, but CentroidClassifier is expecting
4 features as seen during fit.

Not fitted caught: This CentroidClassifier instance is not fitted yet. Call 'fit'
with appropriate arguments before using this estimator.
```

---

## validate_data Key Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `_estimator` | — | The estimator instance (`self`) |
| `X` | — | Input array |
| `y` | `None` | Target array (optional) |
| `reset` | `True` | `True` at fit → store contract; `False` at predict → enforce it |
| `dtype` | `"numeric"` | Cast/check dtype (`"numeric"`, specific dtype, or `None` to skip) |
| `ensure_2d` | `True` | Reject 1D arrays |
| `accept_sparse` | `False` | Accept sparse matrices (`True`, or format string like `"csr"`) |
| `force_all_finite` | `"allow-nan"` | `True` = reject NaN/Inf; `"allow-nan"` = reject only Inf |
| `ensure_min_samples` | `1` | Minimum number of samples required |
| `ensure_min_features` | `1` | Minimum number of features required |

---

## Migration from _validate_data

```python
# Old (deprecated in sklearn 1.6, will be removed in 1.8)
X = self._validate_data(X, reset=False)

# New
from sklearn.utils.validation import validate_data
X = validate_data(self, X, reset=False)
```

The semantics are identical — the only change is that `self` moves from method receiver to first argument.
