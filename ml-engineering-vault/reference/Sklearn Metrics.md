---
type: reference
title: "Sklearn Metrics"
libs:
  - scikit-learn
tags:
  - task/evaluation
  - task/classification
  - task/regression
  - task/clustering
related:
  - "[[Sklearn Datasets]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Sklearn Decision Function and Thresholds]]"
  - "[[Sklearn Display API]]"
  - "[[Sklearn Cross-Validation]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn Metrics

> The `sklearn.metrics` package provides scoring functions for evaluating model performance: classification, regression, clustering, and ranking. Every example below uses real sklearn datasets so you can run them end-to-end.

---

## Common Setup — Models We'll Evaluate

> All examples below share this setup. We train on MNIST (classification) and California Housing (regression) so we have predictions to score.

```python
"""Shared setup — run this cell first."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ── Classification: Digits (8×8 version of MNIST, bundled) ───────────
digits = load_digits()
X_clf, y_clf = digits.data, digits.target  # 10 classes (0–9)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42,
)

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
clf.fit(X_clf_train, y_clf_train)
y_clf_pred = clf.predict(X_clf_test)
y_clf_proba = clf.predict_proba(X_clf_test)  # needed for ROC/PR curves

# Binary classification variant (digit 5 vs not-5)
y_bin_train = (y_clf_train == 5).astype(int)
y_bin_test = (y_clf_test == 5).astype(int)
clf_bin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
clf_bin.fit(X_clf_train, y_bin_train)
y_bin_pred = clf_bin.predict(X_clf_test)
y_bin_proba = clf_bin.predict_proba(X_clf_test)[:, 1]  # P(digit=5)
y_bin_decision = clf_bin.decision_function(X_clf_test)

# ── Regression: California Housing ───────────────────────────────────
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42,
)

reg = make_pipeline(StandardScaler(), Ridge(random_state=42))
reg.fit(X_reg_train, y_reg_train)
y_reg_pred = reg.predict(X_reg_test)

print("Setup complete.")
print(f"  Classification test: {X_clf_test.shape[0]} samples, {len(digits.target_names)} classes")
print(f"  Regression test:     {X_reg_test.shape[0]} samples")
```

---

## Classification Metrics

### Accuracy

> Fraction of predictions that are correct. Simple, but misleading on imbalanced datasets.

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_clf_test, y_clf_pred)
print(f"Multiclass accuracy: {acc:.4f}")

# Normalize=False → raw count of correct predictions
correct = accuracy_score(y_clf_test, y_clf_pred, normalize=False)
print(f"Correct predictions: {correct} / {len(y_clf_test)}")
```

### Confusion Matrix

> Shows where misclassifications happen — rows are true labels, columns are predicted.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_clf_test, y_clf_pred)
print(cm)
# Each cell cm[i, j] = count of true class i predicted as class j

# Visual version
disp = ConfusionMatrixDisplay(cm, display_labels=digits.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Digits Confusion Matrix")
plt.tight_layout()
plt.show()

# Normalized by true class (shows recall per class)
cm_norm = confusion_matrix(y_clf_test, y_clf_pred, normalize="true")
ConfusionMatrixDisplay(cm_norm, display_labels=digits.target_names).plot(
    cmap="Blues", values_format=".2f",
)
plt.title("Normalized Confusion Matrix (recall per class)")
plt.tight_layout()
plt.show()
```

### Precision, Recall, F1-Score

> Precision = "of all I predicted positive, how many are actually positive?"
> Recall = "of all actual positives, how many did I catch?"
> F1 = harmonic mean of precision and recall.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Binary (5 vs not-5) ─────────────────────────────────────────────
p = precision_score(y_bin_test, y_bin_pred)
r = recall_score(y_bin_test, y_bin_pred)
f1 = f1_score(y_bin_test, y_bin_pred)
print(f"Binary — Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

# ── Multiclass — averaging strategies ────────────────────────────────
for avg in ("micro", "macro", "weighted"):
    f1 = f1_score(y_clf_test, y_clf_pred, average=avg)
    print(f"  F1 ({avg:>8s}): {f1:.4f}")

# micro   = global TP / (TP + FP) — same as accuracy for multiclass
# macro   = unweighted mean across classes — treats rare classes equally
# weighted = weighted mean by class support — accounts for imbalance
```

### Classification Report

> One-stop summary: precision, recall, F1, and support for every class.

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_clf_test, y_clf_pred,
    target_names=[str(d) for d in digits.target_names],
)
print(report)
# Returns a formatted string. For programmatic use:
report_dict = classification_report(
    y_clf_test, y_clf_pred, output_dict=True,
)
print(f"Class '5' recall: {report_dict['5']['recall']:.4f}")
```

### Precision-Recall Curve (Binary)

> Plots precision vs recall at every decision threshold. More informative than ROC on imbalanced datasets.

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
import matplotlib.pyplot as plt

ap = average_precision_score(y_bin_test, y_bin_proba)
print(f"Average Precision (AP): {ap:.4f}")

PrecisionRecallDisplay.from_predictions(
    y_bin_test, y_bin_proba, name=f"Digit-5 detector (AP={ap:.2f})",
)
plt.title("Precision-Recall Curve — 5 vs not-5")
plt.tight_layout()
plt.show()
```

### ROC Curve and AUC (Binary)

> Plots True Positive Rate vs False Positive Rate. AUC = area under this curve (1.0 = perfect, 0.5 = random).

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt

auc = roc_auc_score(y_bin_test, y_bin_proba)
print(f"ROC AUC: {auc:.4f}")

RocCurveDisplay.from_predictions(
    y_bin_test, y_bin_proba, name=f"Digit-5 (AUC={auc:.2f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("ROC Curve — 5 vs not-5")
plt.legend()
plt.tight_layout()
plt.show()
```

### ROC AUC — Multiclass

> Extends ROC AUC to multiclass via one-vs-rest (OVR) or one-vs-one (OVO).

```python
from sklearn.metrics import roc_auc_score

# OVR: compute AUC for each class vs rest, then average
auc_ovr = roc_auc_score(
    y_clf_test, y_clf_proba, multi_class="ovr", average="weighted",
)
print(f"Multiclass ROC AUC (OVR, weighted): {auc_ovr:.4f}")

# OVO: compute AUC for each pair of classes, then average
auc_ovo = roc_auc_score(
    y_clf_test, y_clf_proba, multi_class="ovo", average="macro",
)
print(f"Multiclass ROC AUC (OVO, macro):    {auc_ovo:.4f}")
```

### Log Loss (Cross-Entropy)

> Penalizes confident wrong predictions harshly. The loss function behind logistic regression.

```python
from sklearn.metrics import log_loss

ll = log_loss(y_clf_test, y_clf_proba)
print(f"Log loss: {ll:.4f}")

# Binary version
ll_bin = log_loss(y_bin_test, y_bin_proba)
print(f"Binary log loss: {ll_bin:.4f}")
```

### Matthews Correlation Coefficient (MCC)

> Balanced metric that works even with very imbalanced classes. Ranges from -1 (total disagreement) to +1 (perfect).

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_bin_test, y_bin_pred)
print(f"MCC (binary): {mcc:.4f}")

mcc_multi = matthews_corrcoef(y_clf_test, y_clf_pred)
print(f"MCC (multiclass): {mcc_multi:.4f}")
```

---

## Regression Metrics

### Mean Squared Error / Root Mean Squared Error

> MSE penalizes large errors quadratically. RMSE is in the same units as the target.

```python
from sklearn.metrics import mean_squared_error, root_mean_squared_error

mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = root_mean_squared_error(y_reg_test, y_reg_pred)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
# root_mean_squared_error was added in sklearn 1.4;
# on older versions: np.sqrt(mean_squared_error(...))
```

### Mean Absolute Error

> Average absolute deviation — less sensitive to outliers than MSE.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f"MAE: {mae:.4f}")
```

### Mean Absolute Percentage Error

> Error as a percentage of the true value — intuitive but undefined when y_true = 0.

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_reg_test, y_reg_pred)
print(f"MAPE: {mape:.2%}")
```

### R² Score (Coefficient of Determination)

> Proportion of variance explained. 1.0 = perfect, 0.0 = constant-mean predictor, negative = worse than the mean.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_reg_test, y_reg_pred)
print(f"R²: {r2:.4f}")
```

### Comparing Multiple Regression Metrics at Once

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

metrics: dict[str, float] = {
    "MAE":  mean_absolute_error(y_reg_test, y_reg_pred),
    "RMSE": root_mean_squared_error(y_reg_test, y_reg_pred),
    "MSE":  mean_squared_error(y_reg_test, y_reg_pred),
    "R²":   r2_score(y_reg_test, y_reg_pred),
}
for name, value in metrics.items():
    print(f"  {name:>5s}: {value:.4f}")
```

---

## Using Metrics with Cross-Validation

> Pass a scoring string to `cross_val_score` instead of computing metrics manually. Sklearn negates loss-style metrics (lower-is-better) so that higher is always better in the CV API.

```python
from sklearn.model_selection import cross_val_score

# Classification — accuracy (default) and F1
acc_scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring="accuracy")
f1_scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring="f1_macro")
print(f"CV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"CV F1 macro: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# Regression — note the "neg_" prefix (sklearn convention)
rmse_scores = cross_val_score(
    reg, X_reg, y_reg, cv=5, scoring="neg_root_mean_squared_error",
)
print(f"CV RMSE: {-rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
# Negate because sklearn returns negative so that "higher is better"
```

### Common Scoring Strings

| Task | String | What it computes |
|---|---|---|
| Classification | `"accuracy"` | Accuracy |
| Classification | `"f1"` | F1 (binary) |
| Classification | `"f1_macro"` | F1 macro-averaged |
| Classification | `"f1_weighted"` | F1 weighted by support |
| Classification | `"precision"` | Precision (binary) |
| Classification | `"recall"` | Recall (binary) |
| Classification | `"roc_auc"` | ROC AUC (binary) |
| Classification | `"roc_auc_ovr"` | ROC AUC one-vs-rest |
| Classification | `"log_loss"` | Negative log loss |
| Regression | `"neg_mean_squared_error"` | Negative MSE |
| Regression | `"neg_root_mean_squared_error"` | Negative RMSE |
| Regression | `"neg_mean_absolute_error"` | Negative MAE |
| Regression | `"r2"` | R² score |

Full list: `sklearn.metrics.get_scorer_names()`

---

## Custom Scorers with make_scorer

> Wrap any metric function into a scorer object for use with `cross_val_score` and `GridSearchCV`.

```python
from sklearn.metrics import make_scorer, f1_score, mean_squared_error

# Custom classification scorer: F1 for class "5" specifically
f1_class5 = make_scorer(
    f1_score,
    pos_label=5,
    average="binary",
)

# Custom regression scorer: RMSE (greater_is_better=False → auto-negated)
rmse_scorer = make_scorer(
    mean_squared_error,
    greater_is_better=False,  # lower MSE is better
    squared=False,            # return RMSE not MSE
)

# Use in cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_clf, y_clf, cv=5, scoring=f1_class5)
print(f"CV F1 for digit 5: {scores.mean():.4f}")

scores = cross_val_score(reg, X_reg, y_reg, cv=5, scoring=rmse_scorer)
print(f"CV RMSE (custom): {-scores.mean():.4f}")
```

---

## Clustering Metrics

> Evaluate clusterings either against known labels (extrinsic) or purely from geometry (intrinsic).

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)
y_true = iris.target

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
y_pred = kmeans.fit_predict(X_scaled)

# Extrinsic — need ground-truth labels
ari = adjusted_rand_score(y_true, y_pred)
homo = homogeneity_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari:.4f}")   # 1.0 = perfect match
print(f"Homogeneity:         {homo:.4f}")   # 1.0 = each cluster has one class

# Intrinsic — no ground truth needed
sil = silhouette_score(X_scaled, y_pred)
print(f"Silhouette Score:    {sil:.4f}")    # [-1, 1], higher = tighter clusters
```

---

## Pairwise Distances and Kernels

> `sklearn.metrics.pairwise` computes distance matrices and kernel matrices efficiently.

```python
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    rbf_kernel,
)
from sklearn.datasets import load_digits

digits = load_digits()
X_sample = digits.data[:5]  # 5 digit images

# Distance matrix (5×5)
dists = euclidean_distances(X_sample)
print(f"Euclidean distances shape: {dists.shape}")
print(dists.round(1))

# Cosine similarity matrix
cos_sim = cosine_similarity(X_sample)
print(f"\nCosine similarity (first 2 vs first 2):")
print(cos_sim[:2, :2].round(3))

# RBF kernel matrix (used by SVM under the hood)
K = rbf_kernel(X_sample, gamma=0.001)
print(f"\nRBF kernel shape: {K.shape}")
```

---

## Quick Reference: Choosing the Right Metric

| Scenario | Recommended metric | Why |
|---|---|---|
| Balanced classification | Accuracy, F1 macro | Simple, fair across classes |
| Imbalanced classification | F1, PR AUC, MCC | Accuracy is misleading |
| Ranking / probabilities | ROC AUC, Log loss | Evaluates confidence, not just threshold |
| Regression (same-unit error) | RMSE, MAE | Interpretable in target units |
| Regression (outlier-heavy) | MAE, MedAE | Less sensitive to outliers |
| Regression (explained variance) | R² | Quick "how good" summary |
| Clustering (with labels) | Adjusted Rand, Homogeneity | Corrects for chance |
| Clustering (no labels) | Silhouette | Geometry-based |
