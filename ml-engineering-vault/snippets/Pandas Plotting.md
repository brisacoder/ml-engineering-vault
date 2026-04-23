---
type: snippet
title: "Pandas Plotting"
libs:
  - "pandas"
  - "matplotlib"
tags:
  - "task/visualization"
related:
  - "[[Matplotlib Colormaps]]"
  - "[[Seaborn vs Pandas Plotting]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[Accelerated Python - CPU Vectorization]]"
created: 2026-04-22
updated: 2026-04-22
---

# Pandas Plotting

> Patterns for `DataFrame.plot()` — pandas' built-in matplotlib wrapper. These cover the cases where you don't need to drop down to raw matplotlib or reach for seaborn.

---

## Hue-Style Scatter via `c` and `cmap`

> Achieve seaborn's `hue` behavior without importing seaborn. The `c` parameter maps a numeric column to a colormap, and `s` scales point size by another column — giving you four dimensions (x, y, color, size) in one call.

```python
df.plot(
    kind="scatter",
    x="col_x",
    y="col_y",
    s=df["size_col"] / scale_factor,  # point size from a column
    c="color_col",                     # color by this column (like hue)
    cmap="viridis",                    # any matplotlib colormap
    colorbar=True,
    alpha=0.6,
    figsize=(10, 7),
)
```

Under the hood, `DataFrame.plot()` delegates to `matplotlib.axes.Axes.scatter`, so all `scatter` kwargs work — `c`, `s`, `cmap`, `norm`, `vmin`, `vmax`, `edgecolors`, etc.

**Gotchas:**

- `c` must be a **numeric** column for colormaps to work. For categorical color encoding, use [[Seaborn vs Pandas Plotting]] or map categories to integers manually.
- `s` expects **absolute pixel sizes**, not data units. Dividing by a scale factor is almost always necessary to keep points readable.
- The colorbar is tied to the `Axes` — calling `plt.colorbar()` separately will create a duplicate. Use `colorbar=True` in the plot call.
- `cmap="viridis"` is perceptually uniform and colorblind-friendly. Avoid `"jet"` — see [[Matplotlib Colormaps]].

**Source:** *Hands-on Machine Learning with Scikit-Learn and PyTorch* — Aurélien Géron

---

## Scatter Matrix (Pairwise Feature Exploration)

> `pandas.plotting.scatter_matrix` plots every numerical feature against every other in a grid, with histograms (or KDEs) on the diagonal. One call to spot correlations, clusters, and skewed distributions across multiple features at once.

```python
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import fetch_california_housing

# Load California housing as a DataFrame
housing = fetch_california_housing(as_frame=True).frame

attributes = ["MedHouseVal", "MedInc", "AveRooms", "HouseAge"]
scatter_matrix(
    housing[attributes],
    figsize=(12, 8),
    alpha=0.1,        # transparency — essential for large datasets
    diagonal="hist",  # "hist" (default) or "kde"
    grid=True,
)
plt.suptitle("Scatter Matrix — California Housing", y=1.02)
plt.tight_layout()
plt.show()
```

**Key Parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `frame` | — | DataFrame (or subset of columns) to plot |
| `alpha` | `0.5` | Point transparency — lower for large datasets |
| `figsize` | `None` | Figure size in inches `(width, height)` |
| `diagonal` | `"hist"` | `"hist"` for histograms, `"kde"` for density curves |
| `grid` | `False` | Show grid lines on each subplot |
| `density_kwds` | `None` | Dict of kwargs passed to KDE plots |
| `hist_kwds` | `None` | Dict of kwargs passed to histograms |
| `range_padding` | `0.05` | Fraction of padding on axis limits |

**What to look for:**

- **Strong linear bands** between two features → high correlation (e.g., `MedInc` vs `MedHouseVal`).
- **Clusters or gaps** → potential subpopulations worth segmenting.
- **Skewed histograms on the diagonal** → candidate for log/power transform before scaling (see [[Sklearn Feature Scaling]]).

**Gotchas:**

- Only works with **numerical** columns — drop or encode categoricals before calling.
- Scales poorly beyond ~8 features (the grid becomes unreadable). For higher dimensionality, use `seaborn.pairplot` with `vars=` to select a subset, or compute a correlation matrix instead.
- `alpha` is critical for large datasets — without it the scatter plots are just solid blobs.
- Returns a NumPy array of `Axes` objects, so you can fine-tune individual subplots if needed: `axes[0, 1].set_ylabel(...)`.

**Source:** *Hands-on Machine Learning with Scikit-Learn and PyTorch* — Aurélien Géron

---

<!-- Add more pandas plotting patterns below this line -->
