---
type: concept
title: "Sklearn Decision Function and Thresholds"
libs:
  - scikit-learn
tags:
  - concept/optimization
  - task/classification
related:
  - "[[Sklearn Metrics]]"
  - "[[Sklearn Datasets]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
created: 2026-04-22
updated: 2026-04-22
---

# Sklearn Decision Function and Thresholds

> When you call `clf.predict(X)`, the classifier doesn't go directly from features to a label. Internally it computes a **raw score** for each sample, then applies a **threshold** to decide the class. `decision_function()` gives you those raw scores so you can apply your own threshold — trading precision for recall or vice versa.

---

## What predict() Actually Does Under the Hood

> `predict()` is just `decision_function()` + a fixed threshold. Understanding this split is the key.

```
                      ┌────────────────────┐
   X (features) ────►│  decision_function  │────► raw score (float)
                      └────────────────────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  score > threshold? │────► predicted class (True/False)
                      └────────────────────┘
```

For most binary classifiers (SGDClassifier, SVC, LogisticRegression), the default threshold is **0**. Scores above 0 → positive class, scores below 0 → negative class. `predict()` is literally just:

```python
predicted = (clf.decision_function(X) > 0)
```

The score itself tells you **how confident** the model is. A score of 2164 means "very confident positive". A score of -500 means "fairly confident negative". A score of 3 means "barely positive".

---

## Seeing It in Action — Digit 5 Detector

> Train a binary classifier on MNIST digits, then explore what `decision_function()` returns and how thresholds change predictions.

```python
"""
decision_function() returns the raw confidence score.
predict() applies threshold=0 to that score.
You can pick any threshold you want.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# ── Load digits, create binary task: "is it a 5?" ───────────────────
digits = load_digits()
X, y = digits.data, digits.target

y_binary = (y == 5).astype(int)  # 1 = digit is 5, 0 = not 5

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42,
)

clf = SGDClassifier(random_state=42)
clf.fit(X_train, y_train)

# ── Pick a single sample we know is a 5 ─────────────────────────────
idx_fives = np.where(y_test == 1)[0]
some_digit = X_test[idx_fives[0]].reshape(1, -1)

# ── decision_function → raw score ────────────────────────────────────
score = clf.decision_function(some_digit)
print(f"Raw score: {score[0]:.2f}")
# e.g. 1247.53 — a large positive number = "confident it's a 5"

# ── predict() is just: score > 0 ────────────────────────────────────
prediction = clf.predict(some_digit)
manual_prediction = (score > 0)
print(f"predict():          {prediction[0]}")
print(f"score > 0:          {manual_prediction[0]}")
print(f"Same result:        {prediction[0] == manual_prediction[0]}")
# True — they're identical
```

---

## Why You'd Want a Custom Threshold

> The default threshold (0) balances precision and recall in a generic way. But in practice you often care more about one than the other.

**Example scenarios:**

- **Spam filter**: you'd rather let some spam through (lower recall) than send a real email to spam (high precision). → **Raise the threshold**.
- **Cancer screening**: you'd rather flag too many patients for follow-up (lower precision) than miss a cancer case (high recall). → **Lower the threshold**.

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Get scores for all test samples
scores = clf.decision_function(X_test)

# ── Default threshold = 0 ────────────────────────────────────────────
y_pred_default = (scores > 0).astype(int)
print("Threshold = 0 (default)")
print(f"  Precision: {precision_score(y_test, y_pred_default):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_default):.4f}")

# ── High threshold → fewer positives → higher precision, lower recall
y_pred_strict = (scores > 500).astype(int)
print("\nThreshold = 500 (strict / high-precision)")
print(f"  Precision: {precision_score(y_test, y_pred_strict, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_strict):.4f}")

# ── Low threshold → more positives → lower precision, higher recall
y_pred_lenient = (scores > -500).astype(int)
print("\nThreshold = -500 (lenient / high-recall)")
print(f"  Precision: {precision_score(y_test, y_pred_lenient):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_lenient):.4f}")
```

```
Threshold = 0 (default)
  Precision: ~0.82
  Recall:    ~0.75

Threshold = 500 (strict / high-precision)
  Precision: ~0.95    ← fewer false positives
  Recall:    ~0.55    ← misses more actual 5s

Threshold = -500 (lenient / high-recall)
  Precision: ~0.50    ← more false positives
  Recall:    ~0.95    ← catches almost all 5s
```

---

## Finding the Right Threshold Systematically

> Don't guess thresholds. Use `precision_recall_curve` to see precision and recall at **every possible threshold**, then pick the one that matches your requirements.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

# Get scores via cross-validation (avoids overfitting to the training set)
y_scores_cv = cross_val_predict(
    clf, X_train, y_train, cv=5, method="decision_function",
)

# precision_recall_curve returns arrays for every threshold
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_cv)

# ── Plot precision & recall as functions of the threshold ────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: precision and recall vs threshold
ax1.plot(thresholds, precisions[:-1], "b-", label="Precision")
ax1.plot(thresholds, recalls[:-1], "r-", label="Recall")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Score")
ax1.set_title("Precision & Recall vs Threshold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: precision vs recall (the PR curve)
ax2.plot(recalls, precisions, "g-")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── Pick a threshold for 90% precision ───────────────────────────────
target_precision = 0.90
# Find the lowest threshold that achieves >= 90% precision
idx = np.argmax(precisions >= target_precision)
chosen_threshold = thresholds[idx]
print(f"\nFor {target_precision:.0%} precision:")
print(f"  Threshold: {chosen_threshold:.2f}")
print(f"  Precision: {precisions[idx]:.4f}")
print(f"  Recall:    {recalls[idx]:.4f}")

# ── Apply the chosen threshold ───────────────────────────────────────
y_pred_90prec = (clf.decision_function(X_test) > chosen_threshold).astype(int)
print(f"\n  Actual precision on test set: "
      f"{precision_score(y_test, y_pred_90prec, zero_division=0):.4f}")
print(f"  Actual recall on test set:    "
      f"{recall_score(y_test, y_pred_90prec):.4f}")
```

---

## Why the Precision Curve Is Bumpy but Recall Is Smooth

> When you plot precision and recall vs threshold, recall decreases smoothly while precision zigzags. This is not noise — it's a fundamental asymmetry in how their denominators behave.

### The Core Insight

**Recall** = TP / (all actual positives). The denominator is **fixed** — it's the total number of real positives in your dataset, and it never changes no matter what threshold you pick. Raising the threshold can only remove samples from the "predicted positive" group, so you can only *lose* TPs, never gain them. Recall can only go down or stay flat. Hence: smooth.

**Precision** = TP / (all predicted positives). The denominator **shrinks by 1** every time you raise the threshold past a sample. What happens to precision depends on *which* sample you just dropped:

- **Dropped a FP** → numerator stays, denominator shrinks → precision **goes up**
- **Dropped a TP** → numerator AND denominator both shrink → precision can **go down**

The direction at each step depends on whether the next sample in the score ranking is a true positive or a false positive — which varies unpredictably. Hence: bumpy.

### Walkthrough With Concrete Numbers

Imagine 6 actual positives in the dataset, and at some threshold you have these samples above it:

```
Threshold T₁:  4 TP, 1 FP above threshold
  Precision = 4/5 = 80%
  Recall    = 4/6 = 67%
```

Raise the threshold one notch. The lowest-scoring sample above T₁ gets dropped.

**Case A — the dropped sample is the FP:**

```
Threshold T₂:  4 TP, 0 FP above threshold
  Precision = 4/4 = 100%  ↑ (removed a mistake)
  Recall    = 4/6 = 67%   → (unchanged — no TP lost)
```

**Case B — the dropped sample is a TP:**

```
Threshold T₂:  3 TP, 1 FP above threshold
  Precision = 3/4 = 75%   ↓ (removed a correct prediction, kept the mistake)
  Recall    = 3/6 = 50%   ↓ (lost a TP)
```

Case B is the counterintuitive one: you made the classifier *stricter* and precision *dropped*. The FP that's still above the threshold is dragging precision down, and you just removed a TP that was helping.

### Code — See It Happen Sample by Sample

```python
"""
Walk through threshold changes one sample at a time
to see exactly when precision drops vs rises.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# ── Setup: binary digit-5 detector ──────────────────────────────────
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, (digits.target == 5).astype(int),
    test_size=0.3, random_state=42,
)

clf = SGDClassifier(random_state=42)
clf.fit(X_train, y_train)

scores = clf.decision_function(X_test)
labels = y_test

# Sort samples by score descending (highest confidence first)
order = np.argsort(scores)[::-1]
sorted_labels = labels[order]
sorted_scores = scores[order]

# Walk through, adding one sample at a time from highest to lowest score
# (equivalent to lowering the threshold one step at a time)
tp = 0
fp = 0
total_positives = labels.sum()

print(f"{'Step':>4}  {'Score':>8}  {'Label':>5}  {'TP':>3}  {'FP':>3}  "
      f"{'Prec':>6}  {'Recall':>6}  {'Prec Δ':>7}")

prev_precision = None
for i in range(min(30, len(sorted_labels))):  # first 30 steps
    if sorted_labels[i] == 1:
        tp += 1
    else:
        fp += 1

    precision = tp / (tp + fp)
    recall = tp / total_positives
    delta = ""
    if prev_precision is not None:
        diff = precision - prev_precision
        if diff < 0:
            delta = f" {diff:+.3f} ↓"  # precision DROPPED
        elif diff > 0:
            delta = f" {diff:+.3f} ↑"
        else:
            delta = f" {diff:+.3f} →"

    label_str = "TP ✓" if sorted_labels[i] == 1 else "FP ✗"
    print(f"{i+1:>4}  {sorted_scores[i]:>8.1f}  {label_str:>5}  "
          f"{tp:>3}  {fp:>3}  {precision:>6.3f}  {recall:>6.3f}  {delta}")
    prev_precision = precision
```

### Expected Output (excerpt)

```
Step     Score  Label   TP   FP    Prec  Recall   Prec Δ
   1    1523.4   TP ✓    1    0   1.000   0.019   
   2    1498.2   TP ✓    2    0   1.000   0.038   +0.000 →
   3    1301.7   TP ✓    3    0   1.000   0.058   +0.000 →
   4    1105.3   FP ✗    3    1   0.750   0.058   -0.250 ↓  ← FP appeared!
   5    1089.1   TP ✓    4    1   0.800   0.077   +0.050 ↑  ← TP recovered it
   6     998.4   TP ✓    5    1   0.833   0.096   +0.033 ↑
   7     954.2   FP ✗    5    2   0.714   0.096   -0.119 ↓  ← another FP
   ...
```

Every time an FP appears in the ranking, precision drops. Every time a TP appears after an FP, precision partially recovers. The interleaving of TPs and FPs along the score axis is what creates the bumps. Recall, meanwhile, only ever increases (we're walking from high to low threshold here) — it has no mechanism to go down.

---

## decision_function vs predict_proba

> Some classifiers offer both. They return different things.

| Method | Returns | Range | Available on |
|---|---|---|---|
| `decision_function(X)` | Raw signed score | (-∞, +∞) | SVM, SGD, LinearSVC, LogisticRegression |
| `predict_proba(X)` | Calibrated probabilities | [0, 1] per class | LogisticRegression, RandomForest, GradientBoosting |

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

# LogisticRegression has BOTH
lr = LogisticRegression(max_iter=5000, random_state=42)
lr.fit(X_train, y_train)

score = lr.decision_function(some_digit)
proba = lr.predict_proba(some_digit)
print(f"decision_function: {score[0]:>8.2f}")
print(f"predict_proba:     {proba[0]}  (class 0, class 1)")
# predict_proba sums to 1.0 — it's a proper probability distribution

# SGDClassifier has decision_function but NOT predict_proba (by default)
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train)
score = sgd.decision_function(some_digit)
print(f"\nSGD decision_function: {score[0]:.2f}")
# sgd.predict_proba(some_digit)  # ← AttributeError!

# To get probabilities from SGD, use loss="modified_huber" or CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV

sgd_cal = CalibratedClassifierCV(sgd, cv=5)
sgd_cal.fit(X_train, y_train)
proba_cal = sgd_cal.predict_proba(some_digit)
print(f"SGD calibrated proba:  {proba_cal[0]}")
```

### When to Use Which

- **`decision_function`**: when you need raw scores for threshold tuning, ROC/PR curves, or ranking. Works with SVMs and linear models that don't naturally output probabilities.
- **`predict_proba`**: when you need actual probabilities (e.g., "there's a 73% chance this is spam"). Required for `log_loss`, calibration analysis, and multi-class ROC AUC.
- **Both available?** Use `predict_proba` for probability-based metrics (log loss, calibration), `decision_function` for threshold exploration.

---

## Multiclass decision_function

> For multiclass problems, `decision_function` returns one score **per class**. The predicted class is the one with the highest score.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42,
)

clf_multi = SGDClassifier(random_state=42)
clf_multi.fit(X_train, y_train)

sample = X_test[:1]
scores = clf_multi.decision_function(sample)
print(f"Scores shape: {scores.shape}")  # (1, 10) — one score per digit class
print(f"Scores: {scores[0].round(1)}")
print(f"Highest score class: {scores.argmax(axis=1)[0]}")
print(f"predict() says:      {clf_multi.predict(sample)[0]}")
# Same — predict() just picks the class with the highest score

# Map scores to class labels
for cls, score in zip(clf_multi.classes_, scores[0]):
    bar = "█" * max(0, int(score / 100))
    print(f"  Class {cls}: {score:>8.1f}  {bar}")
```

---

## Common Misconceptions

- **"The score is a probability"** — No. Decision function scores are unbounded. A score of 2000 doesn't mean "2000% sure". Use `predict_proba` or `CalibratedClassifierCV` for actual probabilities.
- **"Threshold 0 is always best"** — It's just the default. The optimal threshold depends entirely on your precision/recall requirements.
- **"Higher score = always correct"** — Higher means more confident, not necessarily right. The model can be confidently wrong.
