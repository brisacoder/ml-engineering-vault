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

<!-- Add more pandas plotting patterns below this line -->
