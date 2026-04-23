---
type: recipe
title: "Sklearn Pipelines and ColumnTransformer"
libs:
  - scikit-learn
tags:
  - task/feature-engineering
  - task/data-wrangling
  - concept/architecture
related:
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Advanced Imputation]]"
  - "[[Sklearn validate_data]]"
  - "[[Sklearn Metrics]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn Pipelines and ColumnTransformer

> Patterns for composing preprocessing steps with `Pipeline`, `ColumnTransformer`, and `make_pipeline` — from per-column mini-pipelines to full end-to-end train-predict chains.

---

## Per-Column Pipelines inside ColumnTransformer

> Each column (or group of columns) can have its **own pipeline** of steps. `ColumnTransformer` routes columns to the right pipeline, runs them in parallel, and horizontally concatenates the results.

### Complete Code

```python
"""
Per-column pipelines inside a ColumnTransformer.

Pattern: different columns need different preprocessing chains.
ColumnTransformer lets you assign a dedicated Pipeline to each group.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


# ── Helper: custom feature as a FunctionTransformer ──────────────────
def rooms_per_household(X: np.ndarray) -> np.ndarray:
    """AveRooms / AveOccup — a derived feature."""
    return (X[:, [0]] / X[:, [1]])  # keep 2D


# ── Per-column pipelines ─────────────────────────────────────────────

# Columns that need log transform (heavy-tailed)
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler(),
)

# Columns that just need impute + scale
default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

# Derived-feature pipeline: compute ratio, then impute + scale
ratio_pipeline = make_pipeline(
    FunctionTransformer(rooms_per_household),
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

# ── Assemble into ColumnTransformer ──────────────────────────────────
preprocessing = ColumnTransformer([
    # (name,        pipeline,              columns)
    ("log",         log_pipeline,          ["Population", "AveRooms"]),
    ("ratio",       ratio_pipeline,        ["AveRooms", "AveOccup"]),
    ("default_num", default_num_pipeline,  ["MedInc", "HouseAge"]),
    ],
    remainder="passthrough",  # pass leftover columns unchanged
)

# ── Try it ───────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True).frame
X = housing.drop(columns=["MedHouseVal"])

X_prepared = preprocessing.fit_transform(X)
print(f"Input shape:  {X.shape}")          # (20640, 8)
print(f"Output shape: {X_prepared.shape}")  # columns expanded

# Inspect generated feature names
print(preprocessing.get_feature_names_out())
```

### How It Works

The `ColumnTransformer` constructor takes a list of `(name, transformer, columns)` tuples. The "transformer" can be a single transformer *or* an entire `Pipeline`. Each pipeline processes only its assigned columns, and the outputs are concatenated column-wise.

```
ColumnTransformer
├── "log"         → [Impute → Log → Scale]   → Population, AveRooms
├── "ratio"       → [Ratio → Impute → Scale]  → AveRooms, AveOccup
├── "default_num" → [Impute → Scale]           → MedInc, HouseAge
└── remainder="passthrough"                    → everything else
```

### Key Points

- **A column can appear in multiple pipelines** — `AveRooms` feeds both `log` and `ratio` above. `ColumnTransformer` slices the input independently per pipeline.
- **`remainder`** controls leftover columns: `"drop"` (default) discards them, `"passthrough"` keeps them raw, or pass a transformer/pipeline to process them.
- **`make_column_selector`** can replace explicit column lists — select by dtype:

```python
preprocessing = ColumnTransformer([
    ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
])
```

### Gotchas

- Column order in the output follows the order of the tuples, then `remainder`. Don't assume original column order is preserved.
- `get_feature_names_out()` gives you the full output column names — use it to map back to meaningful names.
- If a `FunctionTransformer` changes the number of columns, set `feature_names_out` to a callable or the names won't track.

---

## Nesting ColumnTransformer Inside a Model Pipeline

> The real power: wrap the entire `ColumnTransformer` preprocessing + a model into a single `Pipeline`. Now `fit()` preprocesses and trains, `predict()` preprocesses and infers — one object, no data leakage, fully serializable.

### Complete Code

```python
"""
End-to-end pipeline: ColumnTransformer (preprocessing) → Model.

The outer Pipeline chains preprocessing and the estimator.
Calling fit/predict on the pipeline runs the full chain.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# ── Preprocessing (same per-column pipelines as above) ───────────────
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
    StandardScaler(),
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)

preprocessing = ColumnTransformer([
    ("log",  log_pipeline,         ["Population", "AveRooms", "AveOccup"]),
    ("num",  default_num_pipeline, ["MedInc", "HouseAge", "AveBedrms",
                                    "Latitude", "Longitude"]),
])

# ── End-to-end pipeline: preprocessing → model ──────────────────────
lin_reg = make_pipeline(preprocessing, LinearRegression())
rf_reg  = make_pipeline(preprocessing, RandomForestRegressor(
    n_estimators=100, random_state=42,
))

# ── Load data and split ──────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True).frame
X = housing.drop(columns=["MedHouseVal"])
y = housing["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)

# ── Fit + predict — one call does preprocessing + model ──────────────
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)  # auto-preprocesses X_test
rmse = root_mean_squared_error(y_test, predictions)
print(f"LinearRegression RMSE: {rmse:.4f}")

# ── Cross-validation works on the full pipeline too ──────────────────
rf_scores = cross_val_score(
    rf_reg, X_train, y_train,
    scoring="neg_root_mean_squared_error", cv=5,
)
print(f"RandomForest CV RMSE: {-rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
```

### How It Works

```
make_pipeline(preprocessing, LinearRegression())

  Pipeline
  ├── Step 1: ColumnTransformer (preprocessing)
  │   ├── "log" → [Impute → Log → Scale]
  │   └── "num" → [Impute → Scale]
  └── Step 2: LinearRegression
```

When you call `pipeline.fit(X_train, y_train)`:
1. `preprocessing.fit_transform(X_train)` runs all sub-pipelines
2. The transformed array feeds into `LinearRegression.fit()`

When you call `pipeline.predict(X_test)`:
1. `preprocessing.transform(X_test)` — uses stats learned from training
2. `LinearRegression.predict()` on the result

**No leakage**: the scaler/imputer only ever `fit` on training data, even during cross-validation.

### Why This Matters

- **Serialization**: `joblib.dump(pipeline, "model.pkl")` saves preprocessing + model as one artifact. Deploy it and call `.predict()` on raw data.
- **Cross-validation**: `cross_val_score` on the pipeline means each fold properly fits the preprocessing on the fold's training split — no data leakage.
- **Grid search**: `GridSearchCV` can tune both preprocessing and model hyperparameters with `param_grid={"columntransformer__log__standardscaler__with_mean": [True, False], "linearregression__fit_intercept": [True, False]}`.

### Accessing Pipeline Steps

```python
# By index
pipeline[0]                # ColumnTransformer
pipeline[1]                # LinearRegression

# By name (make_pipeline auto-names from class)
pipeline.named_steps["columntransformer"]
pipeline.named_steps["linearregression"]

# Explicit names with Pipeline() constructor
pipe = Pipeline([
    ("preprocess", preprocessing),
    ("model", LinearRegression()),
])
pipe.named_steps["model"].coef_  # after fitting
```

### Gotchas

- `make_pipeline` auto-generates step names from class names (lowercased). For clearer names, use `Pipeline([("name", step), ...])` explicitly.
- Nested parameter access in `GridSearchCV` uses double underscores: `"preprocess__log__simpleimputer__strategy"`.
- If you swap models often, keep `preprocessing` as a separate variable and build new pipelines with `make_pipeline(preprocessing, NewModel())` — the `ColumnTransformer` is reusable.

---

## make_pipeline vs Pipeline

> `make_pipeline` is syntactic sugar — it auto-generates step names. Use `Pipeline` when you need explicit names (for grid search or readability).

```python
# These are equivalent:
pipe_a = make_pipeline(SimpleImputer(), StandardScaler(), LinearRegression())
pipe_b = Pipeline([
    ("simpleimputer",    SimpleImputer()),
    ("standardscaler",   StandardScaler()),
    ("linearregression", LinearRegression()),
])

# Explicit names are clearer for grid search:
pipe_c = Pipeline([
    ("impute", SimpleImputer()),
    ("scale",  StandardScaler()),
    ("model",  LinearRegression()),
])
# pipe_c param: "model__fit_intercept"  (readable)
# pipe_a param: "linearregression__fit_intercept"  (auto-generated)
```
