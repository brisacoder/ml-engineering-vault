---
type: recipe
title: "Sklearn Datasets"
libs:
  - scikit-learn
tags:
  - task/data-wrangling
related:
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Pandas Plotting]]"
  - "[[Sklearn Metrics]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn Datasets

> How to load, download, and generate datasets with `sklearn.datasets`. Three families of functions, two return formats, and the `as_frame` toggle.

---

## The Three Function Families

| Prefix | What it does | Network? | Example |
|---|---|---|---|
| `load_*` | Bundled toy datasets shipped with sklearn | No | `load_iris()`, `load_digits()` |
| `fetch_*` | Downloads real-world datasets from the internet | Yes (first call) | `fetch_openml()`, `fetch_california_housing()` |
| `make_*` | Generates synthetic datasets procedurally | No | `make_classification()`, `make_moons()` |

### Code — All Three Families

```python
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_iris,
    make_classification,
)

# ── load_* — bundled, no download ────────────────────────────────────
iris = load_iris()
print(type(iris))  # <class 'sklearn.utils.Bunch'>

# ── fetch_* — downloads on first call, cached after that ─────────────
housing = fetch_california_housing()
mnist = fetch_openml("mnist_784", as_frame=False)

# ── make_* — synthetic, returned as (X, y) tuples ────────────────────
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
print(X.shape, y.shape)  # (1000, 20) (1000,)
```

---

## Return Format: Bunch Object

> `load_*` and `fetch_*` return a `sklearn.utils.Bunch` — a dict subclass whose keys are also accessible as attributes. This is the standard container for sklearn datasets.

### Bunch Attributes

```python
from sklearn.datasets import load_iris

iris = load_iris()

# These are dict keys AND attributes — both work:
print(iris.keys())
# dict_keys(['data', 'target', 'frame', 'target_names',
#            'DESCR', 'feature_names', 'filename'])

print(iris["data"].shape)    # dict-style access
print(iris.data.shape)       # attribute-style access — same thing
# (150, 4)
```

### Standard Bunch Entries

| Key | Type | Description |
|---|---|---|
| `DESCR` | `str` | Human-readable description of the dataset |
| `data` | `ndarray` | Input features, shape `(n_samples, n_features)` |
| `target` | `ndarray` | Labels, shape `(n_samples,)` |
| `feature_names` | `list[str]` | Column names for `data` |
| `target_names` | `ndarray` | Class names (classification) or target description |
| `frame` | `DataFrame \| None` | Full DataFrame (only when `as_frame=True`) |
| `filename` | `str` | Path to the cached CSV on disk |

```python
from sklearn.datasets import load_iris

iris = load_iris()

# Description
print(iris.DESCR[:200])

# Data + target as NumPy arrays (default)
X, y = iris.data, iris.target
print(f"X: {X.shape}, dtype={X.dtype}")  # (150, 4), float64
print(f"y: {y.shape}, dtype={y.dtype}")  # (150,), int64

# Feature and target names
print(iris.feature_names)
# ['sepal length (cm)', 'sepal width (cm)',
#  'petal length (cm)', 'petal width (cm)']
print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']
```

---

## The `as_frame` Toggle

> Controls whether you get NumPy arrays or Pandas DataFrames. Most `load_*` and `fetch_*` functions support it.

```python
from sklearn.datasets import fetch_california_housing

# as_frame=False (default for some) → NumPy arrays
housing_np = fetch_california_housing(as_frame=False)
print(type(housing_np.data))    # <class 'numpy.ndarray'>
print(type(housing_np.target))  # <class 'numpy.ndarray'>

# as_frame=True → Pandas DataFrame / Series
housing_df = fetch_california_housing(as_frame=True)
print(type(housing_df.data))    # <class 'pandas.core.frame.DataFrame'>
print(type(housing_df.target))  # <class 'pandas.core.series.Series'>

# .frame gives the FULL DataFrame (features + target combined)
print(housing_df.frame.columns.tolist())
# ['MedInc', 'HouseAge', ..., 'MedHouseVal']
```

### Shortcut: `return_X_y`

Skip the Bunch entirely and get `(X, y)` directly:

```python
from sklearn.datasets import load_iris

# Combines with as_frame for maximum convenience
X, y = load_iris(return_X_y=True, as_frame=True)
print(type(X))  # DataFrame
print(type(y))  # Series
```

---

## fetch_openml — The Universal Downloader

> `fetch_openml` can download any dataset from [OpenML](https://www.openml.org). It's the most flexible fetcher but has quirks.

```python
from sklearn.datasets import fetch_openml

# By name
mnist = fetch_openml("mnist_784", as_frame=False)

# By ID (more reliable — names can be ambiguous)
mnist = fetch_openml(data_id=554, as_frame=False)

print(mnist.data.shape)    # (70000, 784)
print(mnist.target.shape)  # (70000,)
print(mnist.target.dtype)  # object (strings like '5', '0', '4')
```

### Gotchas with fetch_openml

- **Targets are strings by default** — MNIST labels come back as `dtype=object` (`'5'`, `'0'`, etc.), not integers. Cast them:

```python
import numpy as np

mnist = fetch_openml("mnist_784", as_frame=False)
y = mnist.target.astype(np.int8)
print(y.dtype)  # int8
```

- **Default is `as_frame=True`** (unlike most other fetch functions), which returns a DataFrame. For image data this is wasteful — use `as_frame=False`.
- **First download is slow** — data is cached in `~/scikit_learn_data/` after that.
- **Version pinning** — some datasets have multiple versions. Pin with `version=`:

```python
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
```

---

## Where Data Is Cached

> Downloaded datasets live in `~/scikit_learn_data/` by default. Override with the `data_home` parameter or the `SCIKIT_LEARN_DATA` environment variable.

```python
from sklearn.datasets import get_data_home

print(get_data_home())
# /home/user/scikit_learn_data   (default)

# Override per-call
housing = fetch_california_housing(data_home="/tmp/my_ml_data")

# Or globally via environment variable
# export SCIKIT_LEARN_DATA=/data/ml_cache
```

---

## make_* — Synthetic Data Generators

> Generate controlled datasets for testing, prototyping, or benchmarking. No files, no downloads — pure NumPy output as `(X, y)` tuples.

```python
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_moons,
    make_regression,
)

# Classification with informative + redundant features
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=3,
    random_state=42,
)

# Regression with noise
X, y = make_regression(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42,
)

# Non-linear 2D datasets for visual demos
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X, y = make_blobs(n_samples=300, centers=4, random_state=42)
```

### When to Use

- **Unit testing** custom estimators — deterministic with `random_state`.
- **Benchmarking** model speed — control `n_samples` and `n_features` precisely.
- **Teaching / demos** — `make_moons` and `make_blobs` produce visually clear clusters.
