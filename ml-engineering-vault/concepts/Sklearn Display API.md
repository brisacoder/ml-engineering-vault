---
type: concept
title: "Sklearn Display API"
libs:
  - scikit-learn
tags:
  - concept/other
  - task/evaluation
  - task/visualization
related:
  - "[[Sklearn Metrics]]"
  - "[[Sklearn Decision Function and Thresholds]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Pandas Plotting]]"
created: 2026-04-23
updated: 2026-04-23
---

# Sklearn Display API

> Scikit-learn ships **10 Display classes** that produce publication-quality plots without calling matplotlib directly. Each Display is a thin container holding computed values plus a reference to the matplotlib `ax_` and `figure_`. They all follow the same pattern: classmethods compute values and call `.plot()`, which returns `self`. You only touch matplotlib if you want to customize further.

---

## The Architecture — How Every Display Class Works

> All Display classes share a three-tier API: `from_estimator` (give me a model + data), `from_predictions` (give me pre-computed predictions), and the raw constructor + `.plot()`. Not every class has all three.

```
                  ┌─────────────────────┐
  fitted model ──►│  from_estimator()   │──► calls predict internally
  + X, y          └─────────────────────┘         │
                                                  ▼
                  ┌─────────────────────┐    ┌──────────┐    ┌──────────┐
  y_true,      ──►│  from_predictions() │───►│ __init__ │───►│  .plot() │──► Display obj
  y_pred/proba    └─────────────────────┘    └──────────┘    └──────────┘
                                                  ▲                │
  raw arrays   ──►── constructor directly ────────┘          returns self
  (fpr, tpr…)                                                     │
                                                            .ax_  .figure_
```

**Key insight**: `from_estimator` calls `predict` / `predict_proba` internally, then delegates to `from_predictions`. Use `from_predictions` when you already have predictions (e.g., from [[Sklearn Cross-Validation|cross_val_predict]]) to avoid redundant computation.

### Complete Inventory

| # | Class | Module | `from_predictions`? | Purpose |
|---|---|---|---|---|
| 1 | `ConfusionMatrixDisplay` | `metrics` | Yes | Heatmap of classification errors |
| 2 | `RocCurveDisplay` | `metrics` | Yes | TPR vs FPR |
| 3 | `PrecisionRecallDisplay` | `metrics` | Yes | Precision vs Recall |
| 4 | `DetCurveDisplay` | `metrics` | Yes | Detection Error Tradeoff |
| 5 | `CalibrationDisplay` | `metrics` | Yes | Probability calibration |
| 6 | `PredictionErrorDisplay` | `metrics` | Yes | Regression residual plots |
| 7 | `PartialDependenceDisplay` | `inspection` | No | Feature effect on prediction |
| 8 | `DecisionBoundaryDisplay` | `inspection` | No | 2D decision regions |
| 9 | `LearningCurveDisplay` | `model_selection` | No | Score vs training set size |
| 10 | `ValidationCurveDisplay` | `model_selection` | No | Score vs hyperparameter value |

Classes 7–10 lack `from_predictions` because they inherently need a fitted estimator to compute their output (e.g., evaluating on a grid, retraining at different sizes).

---

## Common Setup — Shared Across All Examples

> Run this once. We train classifiers and a regressor on real sklearn datasets so every example below is self-contained.

```python
"""Shared setup for all Display examples."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ── Multiclass: Digits (8×8 MNIST, 10 classes) ────────────────────
digits = load_digits()
X_clf, y_clf = digits.data, digits.target
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42,
)

clf_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
clf_rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
clf_lr.fit(X_clf_train, y_clf_train)
clf_rf.fit(X_clf_train, y_clf_train)

# ── Binary: digit-5 vs not-5 ──────────────────────────────────────
y_bin_train = (y_clf_train == 5).astype(int)
y_bin_test = (y_clf_test == 5).astype(int)
clf_bin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
clf_bin.fit(X_clf_train, y_bin_train)

# ── 2D classification for decision boundaries: Iris (first 2 features) ─
iris = load_iris()
X_iris_2d = iris.data[:, :2]
y_iris = iris.target
clf_iris = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
clf_iris.fit(X_iris_2d, y_iris)

# ── Regression: California Housing ────────────────────────────────
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42,
)
reg = make_pipeline(StandardScaler(), Ridge(random_state=42))
reg.fit(X_reg_train, y_reg_train)

print("Setup complete.")
```

---

## ConfusionMatrixDisplay

> Heatmap of true vs predicted labels. The `from_predictions` classmethod is the cleanest way — no need to call `confusion_matrix()` first.

```python
from sklearn.metrics import ConfusionMatrixDisplay

# ── from_predictions: pass y_true and y_pred directly ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw counts
ConfusionMatrixDisplay.from_predictions(
    y_clf_test,
    clf_lr.predict(X_clf_test),
    display_labels=digits.target_names,
    cmap="Blues",
    colorbar=False,
    ax=axes[0],
)
axes[0].set_title("Confusion Matrix (counts)")

# Normalized by true label → each row sums to 1 → shows recall per class
ConfusionMatrixDisplay.from_predictions(
    y_clf_test,
    clf_lr.predict(X_clf_test),
    display_labels=digits.target_names,
    normalize="true",
    cmap="Blues",
    values_format=".2f",
    colorbar=False,
    ax=axes[1],
)
axes[1].set_title("Normalized (recall per class)")

plt.tight_layout()
plt.show()

# ── from_estimator: pass the model + data, it calls predict for you ─
ConfusionMatrixDisplay.from_estimator(
    clf_lr,
    X_clf_test,
    y_clf_test,
    display_labels=digits.target_names,
    cmap="Purples",
    colorbar=False,
)
plt.title("from_estimator — same result, fewer steps")
plt.tight_layout()
plt.show()
```

### normalize Options

| Value | Each row/column sums to | Shows |
|---|---|---|
| `"true"` | Rows sum to 1 | Recall per class |
| `"pred"` | Columns sum to 1 | Precision per class |
| `"all"` | Entire matrix sums to 1 | Joint probability |
| `None` (default) | Raw counts | Absolute numbers |

---

## RocCurveDisplay

> Plot TPR vs FPR. Overlay multiple models by passing the same `ax`.

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(7, 6))

# Binary classifiers — overlay two models on the same axes
RocCurveDisplay.from_estimator(
    clf_bin, X_clf_test, y_bin_test,
    name="LogisticRegression",
    ax=ax,
)

clf_bin_rf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42),
)
clf_bin_rf.fit(X_clf_train, y_bin_train)

RocCurveDisplay.from_estimator(
    clf_bin_rf, X_clf_test, y_bin_test,
    name="RandomForest",
    ax=ax,
)

ax.plot([0, 1], [0, 1], "k--", label="Chance level")
ax.set_title("ROC Curve — Digit-5 Detector")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## PrecisionRecallDisplay

> More informative than ROC on imbalanced datasets (digit-5 is ~10% of samples). Same overlay pattern.

```python
from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize=(7, 6))

PrecisionRecallDisplay.from_estimator(
    clf_bin, X_clf_test, y_bin_test,
    name="LogisticRegression",
    ax=ax,
)
PrecisionRecallDisplay.from_estimator(
    clf_bin_rf, X_clf_test, y_bin_test,
    name="RandomForest",
    ax=ax,
)

ax.set_title("Precision-Recall — Digit-5 Detector")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## CalibrationDisplay

> Shows whether predicted probabilities match actual frequencies. A well-calibrated model follows the diagonal.

```python
from sklearn.metrics import CalibrationDisplay

fig, ax = plt.subplots(figsize=(7, 6))

CalibrationDisplay.from_estimator(
    clf_bin, X_clf_test, y_bin_test,
    n_bins=10,
    name="LogisticRegression",
    ax=ax,
)
CalibrationDisplay.from_estimator(
    clf_bin_rf, X_clf_test, y_bin_test,
    n_bins=10,
    name="RandomForest",
    ax=ax,
)

ax.set_title("Calibration Curve — Digit-5 Detector")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## PredictionErrorDisplay (Regression)

> Replaces hand-rolled residual plots. Two modes: actual-vs-predicted scatter and residual-vs-predicted.

```python
from sklearn.metrics import PredictionErrorDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: actual vs predicted — perfect model lies on the diagonal
PredictionErrorDisplay.from_estimator(
    reg, X_reg_test, y_reg_test,
    kind="actual_vs_predicted",
    ax=axes[0],
)
axes[0].set_title("Actual vs Predicted")

# Right: residuals — should be centered around 0 with no pattern
PredictionErrorDisplay.from_estimator(
    reg, X_reg_test, y_reg_test,
    kind="residual_vs_predicted",
    ax=axes[1],
)
axes[1].set_title("Residuals vs Predicted")

plt.tight_layout()
plt.show()
```

---

## DetCurveDisplay

> Detection Error Tradeoff: plots FPR vs FNR on a normal deviance scale. Linear curves on a DET plot indicate normally-distributed scores — useful in biometrics and signal detection.

```python
from sklearn.metrics import DetCurveDisplay

fig, ax = plt.subplots(figsize=(7, 6))

DetCurveDisplay.from_estimator(
    clf_bin, X_clf_test, y_bin_test,
    name="LogisticRegression",
    ax=ax,
)
DetCurveDisplay.from_estimator(
    clf_bin_rf, X_clf_test, y_bin_test,
    name="RandomForest",
    ax=ax,
)

ax.set_title("DET Curve — Digit-5 Detector")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## DecisionBoundaryDisplay (2D only)

> Visualizes decision regions across a 2D feature space. Only works with exactly 2 features.

```python
from sklearn.inspection import DecisionBoundaryDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filled contour (default)
DecisionBoundaryDisplay.from_estimator(
    clf_iris, X_iris_2d, ax=axes[0],
    response_method="predict",
    plot_method="contourf",
    alpha=0.4,
)
axes[0].scatter(
    X_iris_2d[:, 0], X_iris_2d[:, 1],
    c=y_iris, edgecolors="k", cmap="viridis", s=20,
)
axes[0].set_title("Decision Boundary (predict)")
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])

# Probability surface for one class
DecisionBoundaryDisplay.from_estimator(
    clf_iris, X_iris_2d, ax=axes[1],
    response_method="predict_proba",
    plot_method="pcolormesh",
    alpha=0.6,
)
axes[1].scatter(
    X_iris_2d[:, 0], X_iris_2d[:, 1],
    c=y_iris, edgecolors="k", cmap="viridis", s=20,
)
axes[1].set_title("Decision Boundary (predict_proba)")
axes[1].set_xlabel(iris.feature_names[0])
axes[1].set_ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()
```

---

## LearningCurveDisplay

> Score vs training set size — diagnoses whether more data would help. Converging curves = more data won't help (high bias). Large gap = overfitting (high variance).

```python
from sklearn.model_selection import LearningCurveDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

LearningCurveDisplay.from_estimator(
    clf_lr, X_clf, y_clf,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    score_type="both",
    ax=axes[0],
)
axes[0].set_title("Learning Curve — LogisticRegression")

LearningCurveDisplay.from_estimator(
    clf_rf, X_clf, y_clf,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    score_type="both",
    ax=axes[1],
)
axes[1].set_title("Learning Curve — RandomForest")

plt.tight_layout()
plt.show()
```

---

## ValidationCurveDisplay

> Score vs a hyperparameter value — finds the sweet spot before overfitting kicks in.

```python
from sklearn.model_selection import ValidationCurveDisplay

ValidationCurveDisplay.from_estimator(
    LogisticRegression(max_iter=5000, random_state=42),
    StandardScaler().fit_transform(X_clf),
    y_clf,
    param_name="C",
    param_range=np.logspace(-4, 4, 9),
    cv=5,
    scoring="accuracy",
    score_type="both",
)
plt.title("Validation Curve — LogisticRegression C parameter")
plt.xscale("log")
plt.tight_layout()
plt.show()
```

---

## PartialDependenceDisplay

> Shows the marginal effect of one or two features on the model's predictions. Supports PDP (average effect) and ICE (individual conditional expectation) plots.

```python
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import GradientBoostingRegressor

# Use gradient boosting on housing — tree models show interesting PDPs
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
gbr.fit(X_reg_train, y_reg_train)

fig, ax = plt.subplots(figsize=(14, 5))

PartialDependenceDisplay.from_estimator(
    gbr,
    X_reg_train,
    features=[0, 5, (0, 5)],  # MedInc, AveOccup, and their 2D interaction
    feature_names=housing.feature_names,
    kind="both",  # PDP line + ICE lines
    ax=ax,
)
fig.suptitle("Partial Dependence — California Housing")
plt.tight_layout()
plt.show()
```

---

## Composability — Overlaying Multiple Displays

> The `ax` parameter is the key to composability. Pass `display.ax_` from one Display into the next to overlay them on the same plot.

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(7, 6))

models = {
    "LogReg": clf_bin,
    "RandomForest": clf_bin_rf,
}

for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_clf_test, y_bin_test, name=name, ax=ax)

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
ax.legend()
ax.set_title("Model Comparison — ROC")
plt.tight_layout()
plt.show()
```

This pattern works with **any** Display class that accepts `ax`: ROC, PR, DET, Calibration, etc.

---

## Customizing After the Fact

> Every Display stores the matplotlib objects it created. You can modify them after `.plot()` returns.

```python
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_estimator(
    clf_lr, X_clf_test, y_clf_test,
    display_labels=digits.target_names,
    cmap="Blues",
    colorbar=False,
)

# disp.ax_ is the matplotlib Axes — customize as usual
disp.ax_.set_title("Custom Title", fontsize=16, fontweight="bold")
disp.ax_.set_xlabel("Predicted Label", fontsize=12)
disp.ax_.set_ylabel("True Label", fontsize=12)

# disp.im_ is the AxesImage (the heatmap itself)
disp.im_.set_clim(0, 80)  # set color scale limits

# disp.text_ is the ndarray of Text objects (the numbers in cells)
# Make the diagonal bold
for i in range(len(digits.target_names)):
    disp.text_[i, i].set_fontweight("bold")

disp.figure_.set_size_inches(8, 7)
plt.tight_layout()
plt.show()
```

---

## Common Misconceptions

- **"I need matplotlib to plot a confusion matrix"** — Not since sklearn 0.22. `ConfusionMatrixDisplay.from_predictions()` is one line. You only import matplotlib if you want `plt.show()` or need to customize axes.
- **"from_estimator is better than from_predictions"** — Not necessarily. `from_predictions` avoids redundant predict calls when you already have predictions (e.g., from `cross_val_predict`). It's also the only option when predictions come from an external source.
- **"These are just convenience wrappers"** — They're also composable data containers. The Display object stores all computed values (`fpr`, `tpr`, `confusion_matrix`, etc.) so you can inspect them programmatically, not just visually.
- **"DecisionBoundaryDisplay works with any number of features"** — It only works with exactly 2 features. For higher dimensions, reduce to 2 with PCA first.
