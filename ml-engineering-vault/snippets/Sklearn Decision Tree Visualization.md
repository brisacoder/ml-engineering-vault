---
type: snippet
title: "Sklearn Decision Tree Visualization"
libs:
  - "scikit-learn"
  - "matplotlib"
tags:
  - "task/visualization"
  - "task/classification"
related:
  - "[[Sklearn Decision Function and Thresholds]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Display API]]"
  - "[[Sklearn Datasets]]"
  - "[[Pandas Plotting]]"
created: 2026-05-13
updated: 2026-05-13
---

# Sklearn Decision Tree Visualization

> Two complementary ways to inspect a fitted `DecisionTreeClassifier` (or `Regressor`): a **text dump** for quick terminal inspection, and a **matplotlib plot** for visual exploration of splits, impurity, and class distributions.

---

## Text Representation with `export_text`

> Print the tree as indented if/else rules — ideal for logging, quick debugging, or pasting into documentation where graphics aren't practical.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

X, y = load_iris(return_X_y=True, as_frame=True)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

print(export_text(dt, feature_names=list(X.columns), max_depth=3))
```

**Key Parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `decision_tree` | — | A fitted tree estimator |
| `feature_names` | `None` | Human-readable names instead of `feature_0`, `feature_1`, … |
| `max_depth` | `10` | Truncate display beyond this depth |
| `spacing` | `3` | Number of spaces per indentation level |
| `decimals` | `2` | Decimal places for threshold values |
| `show_weights` | `False` | Show sample weights instead of counts |

**Gotchas:**

- `feature_names` expects a plain `list[str]`. Passing an `Index` or `ndarray` directly works, but `export_text` will call `str()` on each element — if your column names have unusual types, convert explicitly.
- `max_depth` here controls **display** depth only — the tree itself may be deeper. Useful for summarizing a large tree without retraining.
- For regression trees the leaf values are means, not class labels.

---

## Graphical Plot with `plot_tree`

> Render the full tree as a matplotlib figure — nodes are colored by majority class (classification) or value magnitude (regression), with saturation proportional to purity. Best for presentations, notebooks, and understanding the geometry of the splits.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

X, y = load_iris(return_X_y=True, as_frame=True)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(
    dt,
    feature_names=list(X.columns),
    class_names=list(load_iris().target_names),
    filled=True,       # color nodes by majority class
    rounded=True,       # rounded box corners
    proportion=False,   # show absolute counts (True → percentages)
    fontsize=10,
    ax=ax,
)
fig.suptitle("Decision Tree — Iris (max_depth=3)", fontsize=14)
plt.tight_layout()
plt.show()
```

**Key Parameters:**

| Parameter | Default | Purpose |
|---|---|---|
| `feature_names` | `None` | Label splits with column names |
| `class_names` | `None` | Label leaf classes with human-readable names |
| `filled` | `False` | Color nodes — hue = class, saturation = purity |
| `rounded` | `False` | Rounded corners on boxes |
| `proportion` | `False` | `True` shows class fractions instead of counts |
| `impurity` | `True` | Display Gini / entropy at each node |
| `fontsize` | `None` | Fixed font size; `None` auto-scales (can get tiny on deep trees) |
| `ax` | `None` | Target `Axes`; creates a new figure if `None` |
| `max_depth` | `None` | Truncate **display** to this depth |
| `precision` | `3` | Decimal places for impurity and threshold |

**Gotchas:**

- Deep trees (depth > 5) become unreadable even on large figures. Use `max_depth` to limit the display, or pair with `export_text` for the full picture.
- `filled=True` requires classification or regression — it doesn't work on un-fitted estimators (raises `NotFittedError`).
- `class_names` order must match the label encoding order (`dt.classes_`). For [[Sklearn Datasets]] loaders, `target_names` is already in the right order.
- Always pass an explicit `ax` when you need to control layout — relying on `plt.gcf()` inside notebooks can produce double figures.
- `plot_tree` returns a list of `matplotlib.text.Annotation` objects, so you can post-process individual node labels if needed.

---

## Exporting to Graphviz (Alternative)

> For publication-quality trees or PDF/SVG export, `export_graphviz` produces DOT source that Graphviz renders. More control over styling, but requires the `graphviz` system package.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

X, y = load_iris(return_X_y=True, as_frame=True)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

# Export DOT source as a string
dot_source: str = export_graphviz(
    dt,
    out_file=None,               # return string instead of writing file
    feature_names=list(X.columns),
    class_names=list(load_iris().target_names),
    filled=True,
    rounded=True,
    special_characters=True,
)

# Render with graphviz (requires: uv pip install graphviz + system graphviz)
import graphviz
graph = graphviz.Source(dot_source)
graph.render("iris_tree", format="png", cleanup=True)  # saves iris_tree.png
```

**When to prefer Graphviz over `plot_tree`:**

- You need vector output (SVG/PDF) for papers or slides.
- You want finer control over node colors, edge labels, or graph layout.
- The tree is moderately deep (5–8) — Graphviz's layout engine handles spacing better than matplotlib.

**Gotchas:**

- Requires both the Python `graphviz` package (`uv pip install graphviz`) **and** the system `graphviz` binary (`apt install graphviz` / `brew install graphviz`). Missing the system package is the most common failure mode.
- `out_file=None` returns a string; without it, `export_graphviz` writes to `"tree.dot"` in the current directory.

---

<!-- Add more decision tree visualization patterns below this line -->
