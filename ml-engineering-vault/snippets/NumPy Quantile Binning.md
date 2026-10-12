---
type: snippet
title: "NumPy Quantile Binning"
libs:
  - numpy
  - pandas
tags:
  - task/data-wrangling
  - task/feature-engineering
related:
  - "[[NumPy Pandas Vectorization]]"
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Datasets]]"
created: 2026-08-27
updated: 2026-08-27
---

# NumPy Quantile Binning

> Splitting a sorted array into quantile groups (quartiles, deciles, ...) means one of two different things depending on whether you want equal *counts* per group or groups bounded by quantile *values*. NumPy alone covers both — no SciPy required. Dataset: Iris `sepal length (cm)`, chosen because its 150 values collapse to only 35 uniques, which is exactly what breaks the naive approach.

---

## Equal-Frequency Groups — `np.array_split`

> The rank-based reading of "quartile": each group holds the same *count* of elements, regardless of value. This is what "top quartile of students" means — top 25% by rank. Because the input is already sorted, this is pure slicing: no comparisons, no computation.

```python
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
sepal_length: NDArray[np.float64] = np.sort(
    iris.frame["sepal length (cm)"].to_numpy()
)


def equal_frequency_groups(
    sorted_values: NDArray[np.float64], n_groups: int
) -> list[NDArray[np.float64]]:
    """Split a pre-sorted 1-D array into n_groups equal-count chunks."""
    return np.array_split(sorted_values, n_groups)


quartiles = equal_frequency_groups(sepal_length, n_groups=4)
for i, group in enumerate(quartiles, start=1):
    print(f"Q{i}: n={len(group)}  range=[{group[0]}, {group[-1]}]")

# Q1: n=38  range=[4.3, 5.1]
# Q2: n=38  range=[5.1, 5.8]
# Q3: n=37  range=[5.8, 6.4]
# Q4: n=37  range=[6.4, 7.9]
```

**Gotchas:** `array_split` (unlike `np.split`) accepts a group count that doesn't divide `len(sorted_values)` evenly — the remainder is spread one-per-group across the first groups, so sizes never differ by more than one. It is completely blind to ties: two elements with the same value can land in different groups purely because of where the rank-based cut falls. That's intentional here — see the next section for what happens when a *value*-based cut is used instead.

---

## Value-Threshold Groups — `np.quantile` + `np.searchsorted`

> The other reading of "quantile": groups bounded by the actual quantile *values* (e.g. "everything below the 25th-percentile value"). Compute the boundary values with `np.quantile`, then binary-search them back into cut indices with `np.searchsorted` — this is the one place `searchsorted` earns its keep, since a sorted array lets it do log-n binary search instead of a linear scan.

```python
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
sepal_length: NDArray[np.float64] = np.sort(
    iris.frame["sepal length (cm)"].to_numpy()
)


def value_threshold_groups(
    sorted_values: NDArray[np.float64],
    n_groups: int,
    side: str = "left",
) -> list[NDArray[np.float64]]:
    """Split a pre-sorted 1-D array at empirical quantile *values*.

    `side` controls which group a value tied with a boundary falls into:
    "left" sends it to the upper group, "right" to the lower group.
    """
    edges = np.quantile(sorted_values, np.linspace(0.0, 1.0, n_groups + 1))
    cut_points = np.searchsorted(sorted_values, edges[1:-1], side=side)
    return np.split(sorted_values, cut_points)


left = value_threshold_groups(sepal_length, n_groups=4, side="left")
right = value_threshold_groups(sepal_length, n_groups=4, side="right")
print([len(g) for g in left])   # [32, 41, 35, 42]
print([len(g) for g in right])  # [41, 39, 35, 35]

# Ties get worse with more groups: at n_groups=20, one group comes back empty
sizes_20 = [len(g) for g in value_threshold_groups(sepal_length, n_groups=20)]
print(sizes_20.count(0))  # 1
```

**Gotchas:** Iris `sepal_length` has 150 values but only 35 distinct ones. `array_split` gave a clean `[38, 38, 37, 37]`; this method gives `[32, 41, 35, 42]` for `side="left"` and a *different* `[41, 39, 35, 35]` for `side="right"` — same data, same quantile edges, two different partitions, because `side` decides which side of a boundary a tied value falls on. Push `n_groups` high enough relative to the number of unique values and `np.split` will hand back a zero-length array for a group whose entire span got absorbed by a tie (empirically: 1 empty group at `n_groups=20`, 69 out of 100 at `n_groups=100` for this data) — always check `len(group)` before indexing into a group downstream.

---

## Matching `pd.qcut`'s Convention (and Its Duplicate-Edge Trap)

> `pd.qcut` is the batteries-included version of the pattern above — it also has to decide which side of a boundary a tie falls on, and its answer is fixed: intervals are left-open, right-closed (`(a, b]`), which is exactly `side="right"` from the previous section.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
sepal_length = iris.frame["sepal length (cm)"].to_numpy()  # unsorted is fine — qcut ranks internally

edges = np.quantile(np.sort(sepal_length), [0.0, 0.25, 0.5, 0.75, 1.0])
cuts_right = np.searchsorted(np.sort(sepal_length), edges[1:-1], side="right")
manual = [len(g) for g in np.split(np.sort(sepal_length), cuts_right)]

bucket = pd.qcut(sepal_length, q=4)
via_qcut = pd.Series(sepal_length).groupby(bucket, observed=True).size().tolist()

print(manual)    # [41, 39, 35, 35]
print(via_qcut)  # [41, 39, 35, 35]  -- identical, side="right" is qcut's rule
```

```python
# The duplicate-edge trap: ask for more bins than the data can distinguish
try:
    pd.qcut(sepal_length, q=20)
except ValueError as e:
    print(e)
# ValueError: Bin edges must be unique: Index([4.3, 4.6, 4.8, 5.0, 5.0, 5.1, ...
#             You can drop duplicate edges by setting the 'duplicates' kwarg

# duplicates="drop" fixes the crash but silently returns FEWER bins than requested
binned = pd.qcut(sepal_length, q=20, duplicates="drop")
print(binned.categories.size)  # 19, not 20
```

**Gotchas:** `duplicates="drop"` is the standard fix, but it changes your bin count without telling you — read back `binned.categories.size` instead of assuming you got what you asked for. To get subarrays (not just labels) out of `pd.qcut`, group by the returned categorical: `series.groupby(bucket, observed=True).apply(lambda g: g.to_numpy())`.

---

## Which One Do I Want?

> The three methods produce different answers on the same data, so the choice isn't cosmetic:

- Reporting "top/bottom 25% of records" → equal-frequency (`np.array_split`). It is what people mean by rank-based quartiles, and it is the cheapest option since the array is already sorted.
- Bucketing by a specific statistical threshold (e.g. "flag anything above the 95th-percentile value") → value-threshold (`np.quantile` + `np.searchsorted`), with `side` chosen deliberately rather than left at the default.
- Need pandas-native labels for a `groupby`/`assign` pipeline, and can tolerate `duplicates="drop"` silently changing the bin count → `pd.qcut`.

No SciPy needed for any of this: `np.quantile`/`np.percentile` already cover quantile *values*. `scipy.stats.mstats.mquantiles` only earns a place over `np.quantile` when the input is a masked array (values to exclude rather than propagate as NaN).

---

<!-- Copy a section above to add more patterns to this topic -->
