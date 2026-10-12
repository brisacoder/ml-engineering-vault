---
type: concept
title: "Multinomial Logistic Regression"
libs:
  - scikit-learn
  - statsmodels
  - numpy
tags:
  - concept/stats
  - concept/optimization
  - concept/math
related:
  - "[[Ridge Regression]]"
  - "[[Lasso Regression]]"
  - "[[Sklearn Cross-Validation]]"
  - "[[Sklearn Datasets]]"
  - "[[Sklearn Decision Function and Thresholds]]"
  - "[[Sklearn Metrics]]"
  - "[[Linear Discriminant Analysis]]"
created: 2026-08-15
updated: 2026-08-15
---

# Multinomial Logistic Regression

> The extension of (binary) [[Sklearn Decision Function and Thresholds|logistic regression]] to $K>2$ classes (ISLP §4.6, eqs. 4.10–4.12). One class is designated the *baseline*; the log-odds of every other class against the baseline is modeled as linear in $x$. Everything else — the softmax formula, the "symmetric" all-$K$-classes parameterization, why the baseline choice doesn't matter, why you need regularization on separable data — falls out of that one assumption plus the constraint that probabilities sum to 1.

> [!note] What's verified here vs. what isn't
> Every algebraic identity below (the ISLP 4.10↔4.11↔4.12 equivalence, the gauge-shift invariance, the gradient/Hessian formulas, the from-scratch Newton–Raphson solver) was checked end-to-end in a pure-NumPy harness in the session this note was written in — including catching and fixing a real ravel/reshape ordering bug in the first draft of the Newton–Raphson code (see the callout in that section). The `sklearn` / `statsmodels` snippets use current, stable APIs (`LogisticRegression` with no `multi_class` arg — deprecated in 1.5, removed in 1.7, multinomial is now the only behavior for non-`liblinear` solvers; `MNLogit`'s reference category is the lowest-sorted label) but weren't executable in the sandbox this note was drafted in (no package-index network access). Run them once before you trust them blindly — that's the whole point of this rule.

---

## From Binary to $K$-Class Logistic Regression (ISLP 4.10–4.12)

### Intuition

Pick one class — say class $K$ — to be the *baseline*. For every other class $k$, model how much more (or less) likely class $k$ is than the baseline, in log-odds, as a linear function of $x$. That single assumption, plus "probabilities sum to 1," is the entire model. Everything ISLP labels (4.10), (4.11), (4.12) is a restatement of it.

### Formal Definition

**The assumption.** For $k=1,\dots,K-1$:

$$
\log\left(\frac{\Pr(Y=k\mid X=x)}{\Pr(Y=K\mid X=x)}\right) = \eta_k(x) := \beta_{k0}+\beta_{k1}x_1+\cdots+\beta_{kp}x_p \tag{*}
$$

**Deriving (4.11).** Exponentiate: $\Pr(Y=k\mid x)=\Pr(Y=K\mid x)\,e^{\eta_k(x)}$ for $k<K$. Substitute into $\sum_{k=1}^K \Pr(Y=k\mid x)=1$:

$$
\begin{aligned}
\Pr(Y=K\mid x) + \sum_{k=1}^{K-1}\Pr(Y=K\mid x)\,e^{\eta_k(x)} &= 1 \\
\Pr(Y=K\mid x)\left[1+\sum_{l=1}^{K-1}e^{\eta_l(x)}\right] &= 1 \\
\Pr(Y=K\mid x) &= \frac{1}{1+\sum_{l=1}^{K-1}e^{\eta_l(x)}}
\end{aligned}
$$

which is exactly (4.11).

**Deriving (4.10).** Substitute (4.11) back into $\Pr(Y=k\mid x)=\Pr(Y=K\mid x)e^{\eta_k(x)}$:

$$
\Pr(Y=k\mid x) = \frac{e^{\eta_k(x)}}{1+\sum_{l=1}^{K-1}e^{\eta_l(x)}}, \qquad k=1,\dots,K-1
$$

Normalization isn't a separate thing to verify — the sum was solved *for*, so it telescopes back to 1 by construction.

**Recovering (4.12).** Divide (4.10) by (4.11): the denominators cancel, leaving $\Pr(Y=k\mid x)/\Pr(Y=K\mid x)=e^{\eta_k(x)}$ — i.e. $(*)$ again. (4.10)+(4.11) and (4.12) are the same model stated two ways, not two separate facts to reconcile.

### In Code

```python
"""
ISLP 4.10-4.12 on Iris: fit sklearn's multinomial logistic regression,
then reproduce its probabilities by hand from the baseline-form formula.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X: np.ndarray = iris.data          # (150, 4)
y: np.ndarray = iris.target        # (150,) in {0, 1, 2} = setosa, versicolor, virginica
class_names: list[str] = list(iris.target_names)

# ── Fit sklearn's multinomial logistic regression ────────────────────
# No `multi_class` kwarg: it's deprecated since 1.5 / removed in 1.7 --
# lbfgs (the default solver) always fits the full softmax loss for K>2.
clf = LogisticRegression(penalty=None, max_iter=5_000)
clf.fit(X, y)

# sklearn stores the SYMMETRIC (all-K-classes) parameterization:
print(f"coef_ shape:      {clf.coef_.shape}")       # (3, 4) -- one row per class
print(f"intercept_ shape: {clf.intercept_.shape}")   # (3,)

# ── Convert to ISLP's baseline form: pick virginica (index 2) as baseline ──
baseline = 2
eta = X @ (clf.coef_[:baseline] - clf.coef_[baseline]).T \
      + (clf.intercept_[:baseline] - clf.intercept_[baseline])   # (150, 2)

denom = 1.0 + np.exp(eta).sum(axis=1)                # (4.11) denominator
p_baseline = 1.0 / denom                              # (4.11)
p_others = np.exp(eta) / denom[:, None]               # (4.10)
p_manual = np.hstack([p_others, p_baseline[:, None]]) # reorder: [k=0, k=1, K=baseline]

p_sklearn = clf.predict_proba(X)

max_diff = np.max(np.abs(p_manual - p_sklearn))
print(f"\nMax |manual (4.10/4.11) - sklearn predict_proba|: {max_diff:.2e}")
assert max_diff < 1e-8, "baseline-form derivation should exactly match sklearn"
```

---

## The Softmax (Symmetric) Parameterization and Gauge Freedom

### Intuition

ISLP never gives class $K$ its own coefficient vector — and that omission is load-bearing, not cosmetic. If you tried to give *every* class, including the baseline, its own linear score (the "symmetric" softmax form), the model would have more parameters than are actually identifiable from data: shifting every class's coefficients by the same constant vector changes nothing about the predicted probabilities. ISLP's baseline trick is one specific way of removing that redundancy.

### Formal Definition

**The symmetric form.** Give every class $k=1,\dots,K$ its own linear score $\gamma_{k0}+\gamma_k\cdot x$:

$$
\Pr(Y=k\mid x) = \frac{e^{\gamma_{k0}+\gamma_k\cdot x}}{\sum_{l=1}^K e^{\gamma_{l0}+\gamma_l\cdot x}}, \qquad k=1,\dots,K
$$

**The gauge symmetry.** Add any constant vector $c=(c_0,c_1,\dots,c_p)$ to *every* class's coefficients: $\gamma_k \to \gamma_k+c$ (and $\gamma_{k0}\to\gamma_{k0}+c_0$). Every term in numerator and denominator picks up the identical factor $e^{c_0+c\cdot x}$:

$$
\frac{e^{c_0+c\cdot x}\,e^{\gamma_{k0}+\gamma_k\cdot x}}{e^{c_0+c\cdot x}\sum_l e^{\gamma_{l0}+\gamma_l\cdot x}} = \frac{e^{\gamma_{k0}+\gamma_k\cdot x}}{\sum_l e^{\gamma_{l0}+\gamma_l\cdot x}}
$$

unchanged. So the symmetric form has $K(p+1)$ parameters, but only $(K-1)(p+1)$ are identifiable — an entire $(p+1)$-dimensional family of parameter settings gives identical probabilities.

**Pinning the gauge.** Set $\gamma_K\equiv 0$ (both intercept and slope). Then $e^{\gamma_{K0}+\gamma_K\cdot x}=1$, and the symmetric formula collapses to exactly (4.10)/(4.11) with $\eta_k=\gamma_k$. Baseline-class logistic regression *is* softmax regression with the redundant degree of freedom removed, not a different model.

### In Code

```python
"""
Gauge freedom, demonstrated two ways:
1. Shifting sklearn's own fitted coefficients by an arbitrary constant
   leaves predict_proba unchanged.
2. A from-scratch softmax fit, gradient-descended from zero, never
   drifts out of the "centered" (sum-to-zero) representative of its
   equivalence class -- because the gradient's per-class columns
   always sum to zero.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
iris = load_iris()
X, y = iris.data, iris.target

clf = LogisticRegression(penalty=None, max_iter=5_000).fit(X, y)

# ── 1. Shift-invariance of sklearn's own coefficients ─────────────────
c_slope = rng.normal(size=X.shape[1])
c_intercept = rng.normal()

coef_shifted = clf.coef_ + c_slope           # same c added to EVERY class
intercept_shifted = clf.intercept_ + c_intercept

def softmax(eta: np.ndarray) -> np.ndarray:
    z = eta - eta.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

p_original = softmax(X @ clf.coef_.T + clf.intercept_)
p_shifted = softmax(X @ coef_shifted.T + intercept_shifted)

print(f"Max |p_original - p_shifted|: {np.max(np.abs(p_original - p_shifted)):.2e}")
assert np.allclose(p_original, p_shifted, atol=1e-10)

# ── 2. Zero-init gradient descent stays centered, with no penalty ────
def fit_softmax_gd(
    X: np.ndarray, y: np.ndarray, n_classes: int,
    lr: float = 0.1, n_steps: int = 5_000,
) -> np.ndarray:
    """Vanilla softmax regression, zero-initialized, unpenalized."""
    n_features = X.shape[1]
    Y_onehot = np.eye(n_classes)[y]
    B = np.zeros((n_features + 1, n_classes))       # row 0 is the intercept
    Xd = np.hstack([np.ones((X.shape[0], 1)), X])
    for _ in range(n_steps):
        P = softmax(Xd @ B)
        grad = Xd.T @ (P - Y_onehot) / Xd.shape[0]   # columns sum to 0 -- see below
        B -= lr * grad
    return B

B_gd = fit_softmax_gd(X, y, n_classes=3)
col_sums = B_gd.sum(axis=1)
print(f"Max |sum_k beta_k| after unpenalized GD: {np.max(np.abs(col_sums)):.2e}")
assert np.max(np.abs(col_sums)) < 1e-8, (
    "gradient columns X^T(P - Y) always sum to 0 across classes, "
    "so a zero start can never leave the centered slice"
)
```

---

## Pairwise Log-Odds and Baseline Invariance

### Intuition

Nothing restricts the linear log-odds property to comparisons against the baseline — it holds between *any* two classes. And since the choice of which class is "the" baseline was arbitrary to begin with, the model's actual predictions can't depend on it. What changes is the numeric value of the coefficients; what doesn't change is the fitted probability for every class, at every $x$.

### Formal Definition

**General pairwise log-odds.** For any $k,m\le K-1$:

$$
\log\frac{\Pr(Y=k\mid x)}{\Pr(Y=m\mid x)} = \log\frac{\Pr(Y=k\mid x)}{\Pr(Y=K\mid x)} - \log\frac{\Pr(Y=m\mid x)}{\Pr(Y=K\mid x)} = \eta_k(x)-\eta_m(x)
$$

still linear, with coefficient vector $\beta_k-\beta_m$ — this is ISLP's generalization right after (4.12).

**Baseline invariance.** Suppose instead you baseline on some other class $K'\neq K$. Using the identity above with $m=K'$:

$$
\log\frac{\Pr(Y=k\mid x)}{\Pr(Y=K'\mid x)} = \eta_k(x)-\eta_{K'}(x) = (\beta_k-\beta_{K'})\cdot\tilde x
$$

so "the coefficient for class $k$ under baseline $K'$" is exactly $\beta_k-\beta_{K'}$ — precisely the gauge shift by $c=\beta_{K'}$ from the previous section. Changing baseline just picks a different representative of the same equivalence class: coefficients differ, but $\Pr(Y=k\mid x)$ for every $k$ and every $x$ is identical.

### In Code

```python
"""
Fit the same model two ways -- sklearn's symmetric softmax form, and
statsmodels' MNLogit (baseline = lowest-sorted category, i.e. setosa=0
for Iris's default integer labels) -- and confirm they agree.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

# ── sklearn: symmetric parameterization, unregularized to match MLE ──
clf = LogisticRegression(penalty=None, max_iter=5_000).fit(X, y)

# ── statsmodels: baseline-form parameterization ───────────────────────
X_const = sm.add_constant(X)
mnlogit_res = sm.MNLogit(y, X_const).fit(disp=0)

# Don't assume which class is the baseline -- discover it: MNLogit's
# params DataFrame has one column per NON-reference category.
all_classes = set(np.unique(y))
fitted_classes = set(mnlogit_res.params.columns)
baseline_class = (all_classes - fitted_classes).pop()
print(f"MNLogit reference category: {baseline_class} ({iris.target_names[baseline_class]})")

# ── Compare fitted probabilities ──────────────────────────────────────
p_sklearn = clf.predict_proba(X)
p_statsmodels = mnlogit_res.predict(X_const)  # already full (n, K) array

max_diff = np.max(np.abs(p_sklearn - p_statsmodels.values))
print(f"Max |sklearn - statsmodels| probabilities: {max_diff:.2e}")

# ── Pairwise log-odds between two NON-baseline classes agree ─────────
non_baseline = sorted(fitted_classes)
k, m = non_baseline[0], non_baseline[1]

lo_sklearn = X @ (clf.coef_[k] - clf.coef_[m]) + (clf.intercept_[k] - clf.intercept_[m])
lo_statsmodels = X_const @ (mnlogit_res.params[k] - mnlogit_res.params[m])

print(f"Max |pairwise log-odds diff| (classes {k} vs {m}): "
      f"{np.max(np.abs(lo_sklearn - lo_statsmodels.values)):.2e}")
```

---

## $K=2$ Reduces to Ordinary Logistic Regression (ISLP 4.7)

### Intuition

With only two classes, there's exactly one non-baseline class, so the model collapses to a single log-odds equation — the ordinary binary logistic regression formula.

### Formal Definition

For $K=2$, only class 1 gets a coefficient vector (class 2 is baseline). (4.10) and (4.11) become

$$
\Pr(Y=1\mid x) = \frac{e^{\eta_1(x)}}{1+e^{\eta_1(x)}}, \qquad \Pr(Y=2\mid x)=\frac{1}{1+e^{\eta_1(x)}}
$$

which is (4.7) with $\beta=\beta_1$. Multinomial logistic regression is the direct generalization of binary logistic regression, not a separately-derived model that happens to look similar.

### In Code

```python
"""
Iris restricted to versicolor vs. virginica (the two classes that
actually overlap -- setosa is trivially separable and not a useful
binary-reduction example): confirm the K=2 case is exactly the
standard sigmoid.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
mask = iris.target != 0                       # drop setosa
X2, y2 = iris.data[mask], iris.target[mask]   # y2 in {1, 2}

clf_binary = LogisticRegression(penalty=None, max_iter=5_000).fit(X2, y2)

p_versicolor_sklearn = clf_binary.predict_proba(X2)[:, 0]
p_versicolor_sigmoid = 1.0 / (1.0 + np.exp(-(X2 @ clf_binary.coef_[0] + clf_binary.intercept_[0])))

max_diff = np.max(np.abs(p_versicolor_sklearn - p_versicolor_sigmoid))
print(f"Max |sklearn predict_proba - manual sigmoid|: {max_diff:.2e}")
assert max_diff < 1e-8
```

---

## Maximum Likelihood: Score Equations, Hessian, Newton–Raphson

### Intuition

There's no closed form for the MLE (unlike OLS), so ISLP just says "fit by maximum likelihood" and moves on. The gradient of the log-likelihood has a clean form — residual (observed minus predicted) times feature — and setting it to zero gives the same "fitted marginal matches observed marginal" moment condition binary logistic regression has, just $K-1$ times over.

### Formal Definition

Let $\tilde x_i=(1,x_{i1},\dots,x_{ip})$, $\eta_{ik}=\beta_k\cdot\tilde x_i$ for $k<K$, $\eta_{iK}\equiv 0$, $p_{ik}=\Pr(Y=k\mid x_i)$. The log-likelihood over $n$ i.i.d. observations:

$$
\ell(\beta) = \sum_{i=1}^n\left[\eta_{i,y_i} - \log\sum_{l=1}^K e^{\eta_{il}}\right]
$$

**Score (gradient).** Differentiate w.r.t. $\beta_{kj}$ for $k<K$ (there is no $\beta_{Kj}$ — it isn't a free parameter):

$$
\frac{\partial \ell}{\partial\beta_{kj}} = \sum_i x_{ij}\big(\mathbb 1\{y_i=k\}-p_{ik}\big)
$$

Setting this to zero: $\sum_i x_{ij}\,\mathbb 1\{y_i=k\} = \sum_i x_{ij}\,p_{ik}$ for every $j$ — including the intercept column, which forces $\sum_i p_{ik} = n_k$ (fitted class counts match observed counts exactly at the MLE).

**Hessian.**

$$
\frac{\partial^2\ell}{\partial\beta_{kj}\,\partial\beta_{ml}} = -\sum_i x_{ij}x_{il}\,p_{ik}\big(\delta_{km}-p_{im}\big)
$$

Negative semi-definite in general; strictly negative definite on the $(K-1)(p+1)$-dimensional identified parameter space when $X$ has full column rank and the classes aren't (quasi-)separated — so $\ell$ is concave there and Newton–Raphson converges to a unique MLE.

> [!warning] A real indexing bug, left in on purpose
> The first draft of the Newton–Raphson code below built the score as `(X.T @ residual).ravel()` where `X.T @ residual` has shape `(d, K-1)` — features on axis 0, classes on axis 1. The Hessian, though, is built class-major (block `k` holds all `d` features of class `k`). `.ravel()` defaults to C-order, which flattens `(d, K-1)` as *feature-major* — so the flat score vector didn't line up with the Hessian's block layout. `np.linalg.solve` didn't complain about a shape mismatch (the total size was still right), it just silently returned a nonsense step. Symptom: looked like slow convergence, then an eventual `LinAlgError: Singular matrix` a few iterations in — which reads like a numerical/separation problem but was actually a pure indexing bug. Fix: transpose before raveling, `(X.T @ residual).T.ravel()`, so both score and Hessian use the same class-major flat layout. Caught by comparing the analytic gradient against finite differences before trusting it.

### In Code

```python
"""
Newton-Raphson MLE for multinomial logistic regression, from scratch,
checked against sklearn's unregularized fit.

Uses only sepal length/width (2 of Iris's 4 features) -- with all 4
features, setosa is perfectly linearly separable and unregularized MLE
has no finite optimum (see the next section). This section is about
verifying the optimizer on a well-posed problem first.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def log_softmax(eta: np.ndarray) -> np.ndarray:
    z = eta - eta.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def softmax(eta: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(eta))


def fit_baseline_newton(
    X: np.ndarray, y: np.ndarray, n_classes: int,
    tol: float = 1e-10, max_iter: int = 100,
) -> tuple[np.ndarray, int]:
    """Newton-Raphson on the (K-1)-block baseline parameterization.

    Returns coefficients of shape (n_features, n_classes - 1) and the
    number of iterations used. Class `n_classes - 1` is the baseline.
    """
    n_features = X.shape[1]
    Y_onehot = np.eye(n_classes)[y]
    beta = np.zeros((n_features, n_classes - 1))

    for iteration in range(max_iter):
        eta = np.hstack([X @ beta, np.zeros((X.shape[0], 1))])
        P = softmax(eta)

        # Score: transpose BEFORE ravel -- see the callout above.
        residual = Y_onehot[:, : n_classes - 1] - P[:, : n_classes - 1]
        score = (X.T @ residual).T.ravel()

        hessian = np.zeros((n_features * (n_classes - 1),) * 2)
        for k in range(n_classes - 1):
            for m in range(n_classes - 1):
                weight = P[:, k] * ((k == m) - P[:, m])
                block = X.T @ (X * weight[:, None])
                hessian[k * n_features:(k + 1) * n_features,
                         m * n_features:(m + 1) * n_features] = block

        step = np.linalg.solve(hessian, score)
        beta += step.reshape(n_classes - 1, n_features).T
        if np.abs(step).max() < tol:
            return beta, iteration

    return beta, max_iter


iris = load_iris()
X = iris.data[:, :2]   # sepal length, sepal width only
y = iris.target
X_design = np.hstack([np.ones((X.shape[0], 1)), X])   # intercept column

beta_manual, n_iter = fit_baseline_newton(X_design, y, n_classes=3)
print(f"Converged in {n_iter + 1} Newton steps")

eta_manual = np.hstack([X_design @ beta_manual, np.zeros((X.shape[0], 1))])
p_manual = softmax(eta_manual)

# sklearn, unregularized, for comparison. Its symmetric coefficients
# have to be converted to baseline form (class 2 = baseline) first.
clf = LogisticRegression(penalty=None, max_iter=5_000).fit(X, y)
beta_sklearn = (clf.coef_[:2] - clf.coef_[2]).T
intercept_sklearn = clf.intercept_[:2] - clf.intercept_[2]

max_diff = np.max(np.abs(beta_manual[1:] - beta_sklearn))
max_diff_intercept = np.max(np.abs(beta_manual[0] - intercept_sklearn))
print(f"Max |slope diff| vs sklearn:     {max_diff:.2e}")
print(f"Max |intercept diff| vs sklearn: {max_diff_intercept:.2e}")

score_at_optimum = (X_design.T @ (np.eye(3)[y][:, :2] - p_manual[:, :2])).T.ravel()
print(f"Max |score| at the fitted optimum: {np.abs(score_at_optimum).max():.2e}")
```

---

## Separation: Why Unregularized MLE Can Diverge

### Intuition

If some hyperplane perfectly separates a class from the rest, the score equations from the previous section are never satisfied at any *finite* $\beta$ — pushing the separating coefficient toward $+\infty$ keeps increasing the likelihood. This is the exact same failure mode as unpenalized binary logistic regression on separable data, and Iris has a real, well-known instance of it: setosa's petal length/width ranges don't overlap at all with versicolor/virginica's.

### Formal Definition

If $x^\top w > 0$ for every observation in class $k$ and $x^\top w < 0$ for every observation not in class $k$ (perfect separation along direction $w$), then scaling $\beta_k \mathrel{+}= t\,w$ and letting $t\to\infty$ drives $p_{ik}\to 1$ for class-$k$ points and $p_{ik}\to 0$ for the rest — monotonically increasing $\ell(\beta)$ without bound. The supremum of the likelihood is approached only in the limit; it's never attained. Practically: the Hessian's relevant eigenvalues shrink toward 0 as probabilities saturate toward 0/1, so Newton–Raphson's Hessian solve becomes ill-conditioned (and can look identical to the indexing bug above — always check the score norm, not just "did it converge").

### In Code

```python
"""
Unregularized multinomial logistic regression on FULL 4-feature Iris:
setosa is perfectly separable by petal length/width, so penalty=None
should trigger a convergence warning and inflated coefficients.
penalty='l2' (sklearn's default) keeps the fit finite.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target   # all 4 features, all 3 classes

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    clf_unpenalized = LogisticRegression(penalty=None, max_iter=1_000).fit(X, y)
    n_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))

print(f"Convergence warnings with penalty=None: {n_warnings}")
print(f"Max |coef_| with penalty=None: {np.abs(clf_unpenalized.coef_).max():.1f}")

clf_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=1_000).fit(X, y)
print(f"Max |coef_| with penalty='l2', C=1.0: {np.abs(clf_l2.coef_).max():.3f}")

# Both still classify training data ~perfectly -- the model isn't wrong,
# the *unregularized point estimate* is just unbounded. See [[Ridge Regression]]
# and [[Lasso Regression]] for why adding a penalty term fixes this exactly
# the same way it fixes ill-conditioning in linear regression.
print(f"\nTrain accuracy (penalty=None): {clf_unpenalized.score(X, y):.3f}")
print(f"Train accuracy (penalty='l2'):  {clf_l2.score(X, y):.3f}")
```

---

## Why $(*)$ Is the Right Assumption, Not an Arbitrary One

### Intuition

ISLP states $(*)$ by analogy with binary logistic regression and moves on. There are two independent justifications for why it's *the* natural choice rather than *a* choice: the log-odds vector is literally the multinomial's natural parameter in the GLM/exponential-family sense, and it's exactly what falls out of Bayes' rule when class-conditional densities are Gaussian with a shared covariance matrix — i.e. multinomial logistic regression and [[Linear Discriminant Analysis]] fit the *same functional form*, just via discriminative vs. generative estimation.

### Formal Definition

**1. The log-odds vector is the multinomial's natural parameter.** Write $Y\in\{1,\dots,K\}$'s pmf using indicators $t_j=\mathbb 1\{Y=j\}$, $j=1,\dots,K-1$ (drop $t_K=1-\sum_{j<K}t_j$, redundant):

$$
\log \Pr(Y=y) = \sum_{j=1}^{K-1} t_j\log\frac{\pi_j}{\pi_K} + \log\pi_K
$$

This is the canonical exponential-family form $p(y;\theta)=\exp(\theta^\top t(y)-A(\theta))$ with $\theta_j=\log(\pi_j/\pi_K)$ and $A(\theta)=\log(1+\sum_j e^{\theta_j})$. The log-odds against baseline *is* the natural parameter. In GLM theory, the canonical link sets the natural parameter linear in the covariates — exactly $(*)$. (Binary logistic regression does the identical move for the Bernoulli's natural parameter $\log(p/(1-p))$; OLS does it trivially for the Gaussian's natural parameter, the mean.)

**2. LDA with shared covariance derives $(*)$.** Assume $X\mid Y=k\sim N(\mu_k,\Sigma)$, shared $\Sigma$, priors $\pi_k$. Bayes' rule:

$$
\log\frac{\Pr(Y=k\mid x)}{\Pr(Y=K\mid x)} = \log\frac{\pi_k}{\pi_K} + \log f_k(x) - \log f_K(x)
$$

Expanding the Gaussian quadratic forms, the $x^\top\Sigma^{-1}x$ terms are *identical* for both classes (same $\Sigma$) and cancel exactly:

$$
\begin{aligned}
\log f_k(x)-\log f_K(x) &= -\tfrac12(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k) + \tfrac12(x-\mu_K)^\top\Sigma^{-1}(x-\mu_K) \\
&= (\mu_k-\mu_K)^\top\Sigma^{-1}x - \tfrac12(\mu_k-\mu_K)^\top\Sigma^{-1}(\mu_k+\mu_K)
\end{aligned}
$$

giving

$$
\log\frac{\Pr(Y=k\mid x)}{\Pr(Y=K\mid x)} = \underbrace{\left[\log\frac{\pi_k}{\pi_K}-\tfrac12(\mu_k-\mu_K)^\top\Sigma^{-1}(\mu_k+\mu_K)\right]}_{\beta_{k0}} + \underbrace{\Sigma^{-1}(\mu_k-\mu_K)}_{\beta_k}{}^{\!\top}x
$$

— linear, i.e. $(*)$, with closed-form coefficients in terms of the generative parameters. **If each class instead gets its own covariance $\Sigma_k$ (QDA)**, the quadratic terms no longer match and don't cancel — log-odds become quadratic in $x$, and $(*)$ fails. Linear log-odds is quietly equivalent to assuming homoscedastic class-conditional densities.

### In Code

```python
"""
LDA, multinomial logistic regression, and QDA on Iris: LDA and
multinomial logistic regression fit the same linear-log-odds form
(generative vs. discriminative estimation of it); QDA doesn't share
that form and should diverge more as class covariances differ.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

lda = LinearDiscriminantAnalysis()
logreg = LogisticRegression(penalty="l2", C=1.0, max_iter=1_000)
qda = QuadraticDiscriminantAnalysis()

for name, model in [("LDA", lda), ("Multinomial LogReg", logreg), ("QDA", qda)]:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name:>20s}: accuracy = {scores.mean():.3f} +/- {scores.std():.3f}")

# LDA and logistic regression should agree closely on which points they
# classify -- both assume linear log-odds, whether or not the Gaussian
# generative assumption underneath LDA is exactly right for Iris.
lda.fit(X, y)
logreg.fit(X, y)
agreement = (lda.predict(X) == logreg.predict(X)).mean()
print(f"\nLDA / logistic regression label agreement: {agreement:.3f}")

# Per-class sample covariances: how far from "shared" is Iris really?
for name, cls in zip(iris.target_names, [0, 1, 2]):
    cov = np.cov(X[y == cls], rowvar=False)
    print(f"{name:>12s} covariance trace: {np.trace(cov):.3f}")
```

---

## Common Misconceptions

- **"The baseline class is special, or the model prefers it"** — no. It's purely an identifiability convenience. $\Pr(Y=k\mid x)$ for every class, at every $x$, is identical no matter which class you baseline on — only the numeric coefficient values change (see the pairwise log-odds / baseline invariance section above).

- **"You need $K$ independent models, one per class"** — no, $K-1$ suffice; the symmetric all-$K$-classes form is deliberately over-parameterized (by exactly $p+1$ redundant dimensions) for notational convenience, not because it's more expressive. It's the same model with a gauge freedom baked in.

- **"Ridge/Lasso don't make sense for multinomial logistic regression because there's no single coefficient vector to penalize"** — they apply directly to the symmetric parameterization (which is what `sklearn` fits by default), and the penalty does double duty: it controls overfitting *and* selects the minimum-Frobenius-norm representative out of the otherwise-infinite gauge-equivalent solutions. See [[Ridge Regression]] and [[Lasso Regression]].

- **"Multinomial logistic regression assumes Gaussian features"** — no. It only assumes linear log-odds, which is *implied by* (not equivalent to) shared-covariance Gaussian class-conditionals. Many other generative pairs of class densities produce the same linear log-odds property; Gaussian-with-shared-$\Sigma$ (i.e. [[Linear Discriminant Analysis]]) is just the standard, checkable example.

- **"An unregularized fit that throws a convergence warning means the implementation is buggy"** — not necessarily; on (quasi-)separable data (real Iris with all 4 features and the setosa class is a genuine example) the MLE simply doesn't exist at any finite $\beta$. That's a property of the data and the loss, not a bug — the fix is the same one used for [[Ridge Regression]]'s ill-conditioning: add an L2 (or L1) penalty.
