---
type: concept
title: "Principal Component Analysis"
libs:
  - numpy
  - pandas
  - scikit-learn
tags:
  - concept/math
  - concept/stats
related:
  - "[[Eigenvalues and Eigenvectors]]"
  - "[[Sklearn Datasets]]"
  - "[[Sklearn Feature Scaling]]"
  - "[[Singular Value Decomposition]]"
  - "[[K-Means Clustering]]"
  - "[[Matrix Completion and Soft-Impute]]"
created: 2026-09-20
updated: 2026-10-12
---

# Principal Component Analysis

> PCA finds new axes — linear combinations of the original features — chosen so the data varies as much as possible along each one, in decreasing order. It's unsupervised: no target $y$ enters the objective, only the feature matrix $X$. This note starts with the data-matrix notation ($X_j$, $x_i$, $x_{ij}$) that every PCA derivation leans on, then follows it through to the first principal component itself. For how PCA's cumulative explained variance bounds what a k-means partition can explain, see [[K-Means Clustering]].

### Quick Reference: ISLR Terms ↔ Linear Algebra

Checked directly against the source (James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, §12.2.1, pp. 505–506) rather than recalled from memory: **ISLR never writes a generic $\phi_k$.** It only ever instantiates a specific, literal component number — $\phi_1$ for the first component (eq. 12.1: $Z_1=\phi_{11}X_1+\cdots+\phi_{p1}X_p$), $\phi_2$ for the second (eq. 12.4: $z_{i2}=\phi_{12}x_{i1}+\cdots+\phi_{p2}x_{ip}$) — and when it does generalize, it writes out an explicit sequence rather than a running index: "the principal component directions $\phi_1,\phi_2,\phi_3,\ldots$ are given by the ordered sequence of eigenvectors of the matrix $X^TX$" (footnote 2, p. 506). A single letter-indexed symbol like $\phi_k$ or $\phi_m$ doesn't appear in this section at all — that was my own shorthand in an earlier answer, not something lifted from the book, and conflating the two was a fair thing to catch.

The table below keeps $v$ (plain eigenvector notation, textbook-agnostic) for the general/abstract role, and reserves $\phi$ only for the concrete, verified instances:

| Concept | Standard linear algebra | ISLR, 1st component | ISLR, 2nd component |
|---|---|---|---|
| Loading vector | eigenvector $v$ | $\phi_1=(\phi_{11},\ldots,\phi_{p1})^T$ | $\phi_2=(\phi_{12},\ldots,\phi_{p2})^T$ |
| Score | $x_i \cdot v$ | $z_{i1}=\phi_{11}x_{i1}+\cdots+\phi_{p1}x_{ip}$ | $z_{i2}=\phi_{12}x_{i1}+\cdots+\phi_{p2}x_{ip}$ |

Loading = a fixed direction, shared by every observation. Score = one observation's position along that direction. A third, fourth, ... component follows the same ordered-eigenvector pattern (that footnote again) — ISLR just never gives "the $k$-th one in general" its own symbol; it names them as they come.

---

## Data Matrix Notation: $X_j$, $x_i$, $x_{ij}$

### Intuition

Three different "grain sizes" of looking at the same table get three different symbols, and mixing them up is the single most common trip-up when reading this notation for the first time. Picture the Iris data as a spreadsheet: 150 rows (flowers), 4 columns (measurements).

- $\mathbf{X}$ is the *whole spreadsheet* — the $n \times p$ matrix.
- $X_j$ (capital, subscript $j$) is a *column* — one feature, e.g. $X_3$ = petal length, taken across all 150 flowers. In the stricter statistical reading, $X_j$ is the random variable itself; a column of numbers is one realization of it.
- $x_i$ (lowercase, subscript $i$) is a *row* — one observation, e.g. $x_1$ = flower #1's full 4-number measurement. It's a vector.
- $x_{ij}$ (lowercase, two subscripts) is a single *cell* — the value of feature $j$ for observation $i$, e.g. $x_{13}$ = flower #1's petal length.

Notice the index convention: $i$ always walks over observations ($i = 1, \dots, n$), $j$ always walks over features ($j = 1, \dots, p$). There is no standard "$X_i$" in this system — if you see a capital $X$ with an $i$ subscript, either the source is using $i$ loosely to mean "the $i$-th thing in a generic list" (common outside stats), or it's a typo for $x_i$ or $X_j$. In ISLR's PCA chapter specifically, capital letters always carry the feature index $j$, never the observation index $i$.

### Formal Definition

Let $n$ be the number of observations and $p$ the number of features. The data matrix is

$$
X = \begin{pmatrix} x_{11} & x_{12} & \cdots & x_{1p} \\ x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix} \in \mathbb{R}^{n \times p}
$$

Row $i$, read as a column vector, is the observation $x_i = (x_{i1}, x_{i2}, \ldots, x_{ip})^T \in \mathbb{R}^p$. Column $j$ is the feature $X_j = (x_{1j}, x_{2j}, \ldots, x_{nj})^T \in \mathbb{R}^n$. The scalar $x_{ij}$ sits at the intersection: row $i$, column $j$, matching the usual matrix-entry convention $A_{ij}$.

### In Code

```python
"""
Data matrix notation on the Iris dataset: X (matrix), X_j (feature/column),
x_i (observation/row), x_ij (scalar entry). Fisher's classic 150 x 4 dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X: np.ndarray = iris.data  # the full n x p data matrix
n, p = X.shape

print(f"n (observations) = {n}, p (features) = {p}")
print(f"X_1 .. X_p (feature names): {iris.feature_names}")

# --- x_i: the i-th OBSERVATION -- a row, a p-vector -------------------
# Textbook indexing is 1-based (x_1 = first flower); numpy is 0-based,
# so x_1 in the book is X[0] in code. This off-by-one is the most
# common bug when translating stats notation straight into array code.
i = 1
x_i = X[i - 1]
print(f"\nx_{i} (flower #{i}'s full measurement): {x_i}")

# --- x_ij: a single SCALAR entry -- feature j of observation i --------
j = 3
x_ij = X[i - 1, j - 1]
print(f"x_{i}{j} ({iris.feature_names[j - 1]} of flower {i}): {x_ij}")

# --- X_j: the j-th FEATURE -- a column, all n observations ------------
X_j = X[:, j - 1]
print(f"\nX_{j} ({iris.feature_names[j - 1]}), shape {X_j.shape}: {X_j[:5]}...")

# A labeled pandas view makes the row/column correspondence explicit:
# DataFrame rows ARE the x_i's, DataFrame columns ARE the X_j's.
df = pd.DataFrame(X, columns=[f"X{k + 1}" for k in range(p)])
df.index = [f"x_{k + 1}" for k in range(n)]
print("\nRows = observations (x_i), columns = features (X_j):")
print(df.head(3))
```

### Common Misconceptions

- **"$X_i$ is the $i$-th row."** No — in this notation the observation index $i$ lives on lowercase $x_i$ and $x_{ij}$. Capital $X_j$ is indexed by $j$, the feature. Seeing a capital letter with an $i$ subscript is the tell that two different conventions have been mixed together.
- **"$X_j$ is a single number."** No — $X_j$ denotes an entire column ($n$ values), or, read as a random variable, the abstract feature itself. Pinning down one number needs both indices: $x_{ij}$.
- **"The order of subscripts in $x_{ij}$ is arbitrary."** It's a fixed convention (row index, then column index), matching standard matrix notation $A_{ij}$. Flipping it silently would flip which axis a sum like $\sum_j$ or $\sum_i$ runs over.
- **"PCA needs a target $y$."** No — PCA is unsupervised. Equations 12.1–12.3 only ever reference $X_1, \ldots, X_p$; no $y$ appears anywhere in the objective.

---

## From Notation to the First Principal Component

### Intuition

With $x_{ij}$ pinned down, the textbook's first principal component is just: for every flower $i$, take a weighted sum across its $p$ measurements — one weight $\phi_{j1}$ per feature — producing one number $z_{i1}$ per flower. PCA searches over *all* possible weight vectors of length 1 (the normalization constraint) and keeps the one that makes those $n$ output numbers vary the most. That search turns out to have a closed-form answer: it's an eigenvector of the covariance matrix, which is why [[Eigenvalues and Eigenvectors]] and PCA are the same linear-algebra problem wearing different clothes.

### Formal Definition

The first principal component is the normalized linear combination

$$
Z_1 = \phi_{11}X_1 + \phi_{21}X_2 + \cdots + \phi_{p1}X_p, \qquad \sum_{j=1}^{p}\phi_{j1}^2 = 1
$$

Applied to one observation $i$, this becomes a scalar **score**:

$$
z_{i1} = \phi_{11}x_{i1} + \phi_{21}x_{i2} + \cdots + \phi_{p1}x_{ip}
$$

and $\phi_1 = (\phi_{11}, \ldots, \phi_{p1})^T$ is called the **loading vector** — fixed weights, one per feature, the same for every observation. The loadings are chosen to solve

$$
\max_{\phi_{11},\ldots,\phi_{p1}} \left\{ \frac{1}{n}\sum_{i=1}^{n} z_{i1}^2 \right\} \quad \text{subject to} \quad \sum_{j=1}^{p}\phi_{j1}^2 = 1
$$

(the data is assumed centered, so the mean of $z_{i1}$ is already zero and this objective is exactly the sample variance of the scores). This maximization is solved by an eigen-decomposition of the covariance matrix $\Sigma = \frac{1}{n-1}X_c^TX_c$: $\phi_1$ is the eigenvector belonging to the largest eigenvalue — see the full derivation in [[Eigenvalues and Eigenvectors#Eigen-decomposition of a Real Covariance Matrix (PCA)]].

### In Code

```python
"""
Equations 12.1-12.3: turning x_ij into the first principal component
score z_i1 via the loading vector phi_1. Same Iris matrix as above --
[[Eigenvalues and Eigenvectors#Eigen-decomposition of a Real Covariance Matrix (PCA)]]
derives why this reduces to an eigen-decomposition of the covariance matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape

# PCA maximizes variance, so the eq. 12.3 objective assumes the mean of
# x_ij over i is already zero for every feature j -- center first.
X_centered = X - X.mean(axis=0)
cov = (X_centered.T @ X_centered) / (n - 1)

# eigh exploits covariance-matrix symmetry; eigenvalues come back
# ascending, so flip the order to match PCA's descending convention.
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
phi_1 = eigvecs[:, order[0]]  # loading vector, eq. 12.1

print(f"phi_1 (loadings): {np.round(phi_1, 4)}")
print(f"sum(phi_j1^2) = {np.sum(phi_1 ** 2):.6f}  <- the eq. 12.3 constraint")

# z_i1 = phi_11*x_i1 + phi_21*x_i2 + ... + phi_p1*x_ip, for EVERY i at
# once -- eq. 12.2 is exactly this dot product, applied row by row.
z_scores = X_centered @ phi_1
print(f"\nz_1 (flower #1's PC1 score): {z_scores[0]:.4f}")
print(f"Sample variance of z_i1, the eq. 12.3 objective: {np.mean(z_scores ** 2):.4f}")

# Cross-check against sklearn -- the same eigh-vs-PCA equivalence is
# verified in detail in [[Eigenvalues and Eigenvectors]].
pca = PCA(n_components=1).fit(X)
z_sklearn = pca.transform(X).ravel()

# An eigenvector and its negation both solve Sigma @ v = lambda * v, so
# sklearn's sign convention can flip the whole component -- compare magnitudes.
print(f"\nMatches sklearn PCA (up to sign): "
      f"{np.allclose(np.abs(z_scores), np.abs(z_sklearn))}")
print(f"explained_variance_ratio_ (PC1): {pca.explained_variance_ratio_[0]:.4f}")
```

Running this on raw Iris: `phi_1 ≈ [-0.36, 0.08, -0.86, -0.36]` and PC1 explains about 92.5% of total variance — a value dominated almost entirely by petal length. That's not a coincidence; see the misconceptions below.

### Common Misconceptions

- **"The first principal component is one of the original features."** No — $Z_1$ is a new, synthetic variable, a weighted blend of all $p$ original features. None of the $\phi_{j1}$ are typically zero or one.
- **"Loadings and scores are the same thing."** They're not, and the equations above show why: $\phi_1$ has no observation index — it's $p$ fixed numbers shared by every flower. $z_{i1}$ has an observation index — it's a different number for each of the $n$ flowers. Loadings describe the direction; scores describe where each point lands on it.
- **"PCA on unstandardized features finds the most biologically important axis."** In this run, PC1's largest-magnitude loading (-0.86) landed on petal length — and petal length's raw variance (≈3.12) dwarfs sepal width's (≈0.19). PCA maximizes variance in whatever units you hand it, so a feature measured on a larger scale mechanically dominates the first component regardless of its actual relevance. Standardizing first (see [[Sklearn Feature Scaling]]) changes PC1's explained variance ratio from ≈92.5% down to ≈73%, spreading more of the signal onto PC2 — same data, different question being asked.

---

## Raw vs. Standardized: Why Scale Changes the Answer

### Intuition

Section 12.3's optimization has no notion of "units" built in — it purely maximizes $\frac{1}{n}\sum_i z_{i1}^2$. If one feature happens to be measured in units that produce a numerically larger variance, the optimizer will happily assign it a large loading just to chase that variance, whether or not it's the most scientifically meaningful axis. This is why `StandardScaler` so often appears immediately before `PCA` in a pipeline.

### In Code

```python
"""
PCA's variance-maximization objective is scale-dependent: whichever
feature has the largest raw variance dominates the first component.
Standardizing (mean 0, unit variance per feature) removes that bias.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


def explained_variance_ratio(data: np.ndarray) -> np.ndarray:
    """Eigen-decomposition-based explained variance ratio, descending."""
    centered = data - data.mean(axis=0)
    cov = (centered.T @ centered) / (data.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return eigvals / eigvals.sum()


iris = load_iris()
X = iris.data

print("Per-feature variance (raw, unstandardized):")
for name, var in zip(iris.feature_names, X.var(axis=0, ddof=1)):
    print(f"  {name:20s}: {var:.4f}")

raw_ratio = explained_variance_ratio(X)
print(f"\nRaw explained variance ratio:          {np.round(raw_ratio, 4)}")

X_std = StandardScaler().fit_transform(X)
std_ratio = explained_variance_ratio(X_std)
print(f"Standardized explained variance ratio: {np.round(std_ratio, 4)}")
```

Petal length's variance (≈3.12 cm²) is over 16x sepal width's (≈0.19 cm²) purely from the raw measurement scale — nothing biological about it. That gap is exactly why the raw and standardized explained-variance ratios differ.

### Common Misconceptions

- **"There's one universally correct choice between raw and standardized PCA."** Neither is unconditionally right. Standardize when features are in different units or scales that shouldn't dictate importance (the usual case). Skip it when the units are already comparable and the scale genuinely carries meaning you want PCA to respect (e.g., all features already in the same physical unit and you want the natural variance to drive the analysis).

---

## Eigenvectors, Singular Vectors, and What sklearn Actually Computes

### Intuition

Eigendecomposition works on the $p \times p$ covariance matrix. SVD works directly on the $n \times p$ centered data matrix. These aren't merely two roads to similar answers — for PCA they are provably the *same* computation, viewed through two different matrix factorizations. That equivalence is also why sklearn can afford to switch between the two behind your back. Forming $X_c^TX_c$ squares the condition number and costs floating-point precision, so the numerically safe default is SVD on the centered data — but since sklearn 1.5 the `svd_solver="auto"` policy (read from `sklearn/decomposition/_pca.py`, `_fit`) picks `covariance_eigh` — literally build the covariance matrix and call `eigh` — whenever $p \le 1000$ and $n \ge 10p$, because on tall-skinny data the $p\times p$ eigenproblem is far cheaper and the precision loss is tolerable. Iris (150 × 4) takes that path. Wide or square data falls through to `full` SVD or `randomized`.

### Formal Definition

Let $X_c \in \mathbb{R}^{n\times p}$ be centered, with (economy) SVD $X_c = U S V^T$: $U \in \mathbb{R}^{n\times p}$ has orthonormal columns, $S = \operatorname{diag}(s_1 \geq \cdots \geq s_p \geq 0)$, and $V \in \mathbb{R}^{p\times p}$ is orthogonal. Then

$$
X_c^TX_c = VS^2V^T \quad\Longrightarrow\quad \Sigma = \frac{X_c^TX_c}{n-1} = V\left(\frac{S^2}{n-1}\right)V^T
$$

Comparing this to the eigendecomposition $\Sigma = V\Lambda V^T$ identifies the two factorizations term for term: $\Lambda = S^2/(n-1)$, i.e. $\lambda_k = s_k^2/(n-1)$, and — this is the point — **the same $V$** is both "the right-singular-vector matrix of $X_c$" and "the eigenvector matrix of $\Sigma$." Not analogous. Identical, given consistent ordering by magnitude (both $s_k$ and $\lambda_k$ are sorted descending) and given that sign is never fixed by either factorization ($v$ and $-v$ solve $\Sigma v = \lambda v$ equally well).

The scores fall out of the same identity: $X_cV = US V^TV = US$ (since $V^TV=I$), so $Z = X_cV = US$. The left singular vectors $U$ alone are *not* the scores — they need the $S$ rescaling first. $U$'s columns carry the same role as $x_i$ vs $X_j$ from the notation section above: $U$ is indexed per-observation (like the scores), $V$ is indexed per-feature (like the loadings). Swapping them is the SVD-flavored version of that same $X_j$/$x_i$ mix-up.

### In Code

```python
"""
Three routes to the same principal axes on Iris: eigh(covariance),
np.linalg.svd(centered X) directly, and sklearn's PCA. All three must
agree up to a +-1 sign per component -- SVD is what sklearn actually
runs for wide data; for tall-skinny data like Iris it actually takes Route 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape
X_centered = X - X.mean(axis=0)

# --- Route 1: eigendecomposition of the covariance matrix -------------
cov = (X_centered.T @ X_centered) / (n - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

# --- Route 2: SVD directly on the centered data matrix -----------------
# X_centered = U @ diag(S) @ Vt  ->  X_centered.T @ X_centered = V @ S^2 @ V.T
# so V (= Vt.T) IS the covariance matrix's eigenvector matrix, and
# S**2 / (n - 1) IS its eigenvalues. No covariance matrix ever formed.
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
svd_eigvals = S ** 2 / (n - 1)

# --- Route 3: sklearn -- Route 1 (covariance_eigh) for Iris's shape, Route 2 otherwise
pca = PCA(n_components=p).fit(X)

print("Eigenvalues (eigh):        ", np.round(eigvals, 4))
print("Eigenvalues (S^2/(n-1)):   ", np.round(svd_eigvals, 4))
print("Eigenvalues (sklearn):     ", np.round(pca.explained_variance_, 4))

# Sign is arbitrary per eigenvector (v and -v both solve Sigma@v = lambda*v),
# so compare magnitudes, not raw values, when checking agreement.
print("\neigh vs SVD     max |diff|:", np.max(np.abs(np.abs(eigvecs) - np.abs(Vt.T))))
print("eigh vs sklearn  max |diff|:", np.max(np.abs(np.abs(eigvecs) - np.abs(pca.components_.T))))

# --- Scores: Z = X_centered @ V, but SVD hands you U @ diag(S) directly ---
Z_via_eigh = X_centered @ eigvecs
Z_via_svd = U * S
print(f"\nScores match (up to per-column sign): "
      f"{np.allclose(np.abs(Z_via_eigh), np.abs(Z_via_svd))}")
```

### Common Misconceptions

- **"Eigendecomposition and SVD give similar results for PCA."** Not similar — identical. $V$, the right-singular-vector matrix of the centered data, *is* the covariance matrix's eigenvector matrix, exactly, given consistent sign and ordering conventions.
- **"`sklearn.decomposition.PCA` always runs an SVD."** Not since 1.5. With `svd_solver="auto"` it forms the covariance matrix and calls `eigh` when $p\le1000$ and $n\ge10p$ (`covariance_eigh`), and uses SVD (`full` or `randomized`) otherwise. The docstring itself warns that `covariance_eigh` "effectively doubles the condition number" — the same stability argument behind the SVD solver in [[Ridge Regression]] — so if your features span many orders of magnitude, pass `svd_solver="full"` explicitly. Either route yields identical `components_` up to sign, which is the whole point of this section.
- **"$U$ (left singular vectors) are the loadings."** No — $U$ pairs with the *scores* ($Z = US$), not the loadings. $V$ (right singular vectors, one column per **feature**) are the loadings/eigenvectors; $U$'s columns are indexed per **observation**. This is the SVD-side echo of the $X_j$-vs-$x_i$ confusion from the first section of this note.
- **"There's one unique eigenvector per eigenvalue."** Only when eigenvalues are distinct. A repeated eigenvalue makes its entire eigenspace valid — any orthonormal basis of that subspace is an equally correct set of loadings, not a single fixed vector. See [[Eigenvalues and Eigenvectors#Common Misconceptions]] for the general case.

---

## Two Things Called "Loadings"

### Intuition

"Loadings" is overloaded across fields, and the two meanings can differ by an order of magnitude for the same dataset. ISLR (eq. 12.1) and sklearn's `components_` both mean the raw, unit-norm eigenvector — $\sum_j \phi_{j1}^2 = 1$, no exceptions. Classical multivariate statistics and factor analysis (and packages like R's `principal()`) instead scale each eigenvector by the square root of its eigenvalue. That rescaled version has a concrete interpretation the unit-norm version doesn't: on standardized data, it equals the correlation between the original feature and the component score. Neither definition is "wrong" — they answer different questions — but silently assuming they're interchangeable is how a loadings table copied from R stops matching `pca.components_` from sklearn.

### Formal Definition

Given eigenvector $v_k$ and its eigenvalue $\lambda_k$ (standard notation — ISLR's $\phi$ only ever labels a specific, instantiated component like $\phi_1$ or $\phi_2$, never a general $k$; see the note above), define the **scaled loading**

$$
a_{jk} = v_{jk}\sqrt{\lambda_k}
$$

For observation $i$, $\operatorname{Cov}(X_j, Z_k) = \lambda_k v_{jk}$ always holds (derivable directly from $Z_k = X_cv_k$ and $\Sigma v_k=\lambda_k v_k$), so

$$
\operatorname{Corr}(X_j, Z_k) = \frac{\operatorname{Cov}(X_j,Z_k)}{\operatorname{sd}(X_j)\operatorname{sd}(Z_k)} = \frac{\sqrt{\lambda_k}\,v_{jk}}{\operatorname{sd}(X_j)} = \frac{a_{jk}}{\operatorname{sd}(X_j)}
$$

which collapses to $\operatorname{Corr}(X_j, Z_k) = a_{jk}$ exactly when $\operatorname{sd}(X_j) = 1$ for every $j$ — i.e., when PCA is run on standardized features. That's the entire justification for the classical convention: it's the eigenvector rescaled so its entries read directly as correlations.

### In Code

```python
"""
Two different things are called 'loadings' depending on the source:
(1) ISLR / sklearn's components_ -- the unit-norm eigenvector itself.
(2) Classical / factor-analysis convention -- eigenvector * sqrt(eigenvalue),
    which equals Corr(X_j, Z_k) -- but ONLY if the eigenvalue and the
    standardization step use the SAME variance convention (ddof). Mixing
    sklearn's StandardScaler (ddof=0) with PCA's explained_variance_
    (ddof=1) introduces a tiny, exactly-quantifiable mismatch -- shown below.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape

# Standardize by hand with ddof=1, matching the ddof PCA itself uses
# internally for explained_variance_ (a sample, not population, variance).
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

pca = PCA(n_components=p).fit(X_std)
eigenvectors = pca.components_.T                                   # unit-norm columns
scaled_loadings = eigenvectors * np.sqrt(pca.explained_variance_)  # v_k * sqrt(lambda_k)

scores = pca.transform(X_std)
corr_Xj_Zk = np.corrcoef(np.hstack([X_std, scores]).T)[:p, p:]

print("Scaled loading, PC1:           ", np.round(scaled_loadings[:, 0], 4))
print("Direct Corr(X_j, Z_1), PC1:    ", np.round(corr_Xj_Zk[:, 0], 4))
print("Exact match (ddof-consistent): ", np.allclose(scaled_loadings, corr_Xj_Zk))

# --- The gotcha: sklearn's StandardScaler uses ddof=0, PCA uses ddof=1 ---
from sklearn.preprocessing import StandardScaler

X_std_sklearn = StandardScaler().fit_transform(X)  # ddof=0 internally
pca2 = PCA(n_components=p).fit(X_std_sklearn)
loadings_mismatched = pca2.components_.T * np.sqrt(pca2.explained_variance_)
corr_mismatched = np.corrcoef(
    np.hstack([X_std_sklearn, pca2.transform(X_std_sklearn)]).T
)[:p, p:]

# Correlation itself is scale-invariant -- ddof choice never touches it.
print(f"\nCorr(X_j, Z_k) is identical either way: "
      f"{np.allclose(corr_Xj_Zk, corr_mismatched)}")

ratio = loadings_mismatched[:, 0] / corr_mismatched[:, 0]
print(f"StandardScaler(ddof=0) + PCA(ddof=1) loading/correlation ratio: "
      f"{np.round(ratio, 6)}")
print(f"sqrt(n / (n - 1)) = {np.sqrt(n / (n - 1)):.6f}  <- exactly the mismatch")
```

On Iris ($n=150$), that mismatch is a 0.335% overstatement of the scaled loadings when `StandardScaler` feeds `PCA` — small here, but it grows as $n$ shrinks, and it's the kind of silent discrepancy that turns into a confusing "why don't my numbers match the paper" debugging session.

### Common Misconceptions

- **"'Loadings' means the same thing everywhere."** No. ISLR and `pca.components_` mean the unit-norm eigenvector; classical/factor-analysis usage means that eigenvector scaled by $\sqrt{\lambda_k}$. The two can differ by a large factor for the leading component. Check which one you have: sum the squares down a *column* (should be 1 for the unit-norm convention) versus across a *row* (should be 1 for the scaled convention on standardized data, since a feature's variance is fully distributed across all $p$ components).
- **"sklearn exposes a `.loadings_` attribute."** It doesn't. `PCA` exposes `components_` (unit-norm), `explained_variance_`, `explained_variance_ratio_`, and `singular_values_`. Anything called "loadings" beyond the raw eigenvector has to be computed by hand from these, as above.
- **"Standardizing with sklearn's `StandardScaler` before `PCA` is perfectly consistent with PCA's own variance accounting."** Almost, not exactly: `StandardScaler` divides by the population standard deviation (`ddof=0`), while `PCA.explained_variance_` uses the sample convention (`ddof=1`) internally. Chaining them makes scaled loadings differ from the true $\operatorname{Corr}(X_j, Z_k)$ by exactly $\sqrt{n/(n-1)}$ — negligible for large $n$, but a real, quantifiable gap rather than floating-point noise, and worth knowing about before assuming exact equality.

---

## sklearn's `score` Is Not the ISLR Score

### Intuition

Naming collision, and a bad one given everything above. ISLR's *scores* $z_{i1}$ are what `pca.transform(X)` returns — one row per observation. `pca.score(X)` is something else entirely: a single float, the **average log-likelihood** of the rows of `X` under a Gaussian whose covariance is rebuilt from the fitted components. It exists so PCA has a scalar to maximize and can be dropped into `cross_val_score` / `GridSearchCV` like a supervised estimator — the standard way to choose `n_components` by held-out likelihood. `score_samples(X)` is the per-row version (one log-likelihood per observation); `score` is just its mean.

The "extremely complicated paper" is Tipping & Bishop (1999), *probabilistic PCA*. Everything sklearn actually uses from it fits in three lines:

1. Treat the data as Gaussian, $x \sim \mathcal{N}(\mu, C)$, with $\mu$ = `mean_`.
2. Build $C$ from the fit: the $k$ kept directions get their true variances $\lambda_1,\dots,\lambda_k$; every direction you *dropped* is treated as isotropic noise with variance $\sigma^2$ = `noise_variance_` = the **mean of the discarded eigenvalues**.
3. `score_samples` = ordinary multivariate-normal log-density of each row under that $\mathcal{N}(\mu, C)$. `score` = its mean.

So a row scores high if it sits close to the kept $k$-dimensional subspace (small residual relative to $\sigma^2$) and within the usual spread along it; it scores low if it sticks far out of that subspace.

### Formal Definition

With $V_k$ the $p\times k$ matrix of kept loadings (`components_.T`), $\lambda_i$ = `explained_variance_`, and $\sigma^2 = \frac{1}{p-k}\sum_{i>k}\lambda_i$ (`noise_variance_`; 0 if nothing was dropped):

$$
C = V_k\operatorname{diag}(\lambda_1-\sigma^2,\dots,\lambda_k-\sigma^2)V_k^T + \sigma^2 I_p
$$

This is `get_covariance()`. Along a kept eigenvector $v_i$ the model variance is $v_i^TCv_i = \lambda_i$; along a discarded one it is $\sigma^2$. Then, with $x_c = x-\mu$,

$$
\texttt{score\_samples}(x) = -\tfrac{1}{2}\left[p\log 2\pi + \log|C| + x_c^TC^{-1}x_c\right],
\qquad
\texttt{score}(X) = \frac{1}{n}\sum_i \texttt{score\_samples}(x_i)
$$

That's the full content of the method — read directly from `sklearn/decomposition/_pca.py` (`score_samples`, `score`, the `noise_variance_` line in `_fit_full`) and `_base.py` (`get_covariance`).

### In Code

```python
"""
PCA.score is NOT the ISLR 'score'. ISLR's scores (z_i1) come from transform.
PCA.score(X) is the AVERAGE LOG-LIKELIHOOD of the rows of X under a Gaussian
whose covariance is rebuilt from the fitted components -- the probabilistic
PCA model of Tipping & Bishop (1999). Reproduced here from scratch in numpy.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape
k = 2  # keep two components, discard two

pca = PCA(n_components=k).fit(X)

# --- The two different things called "score" ---------------------------
z = pca.transform(X)                  # ISLR scores: shape (150, 2), one row per flower
ll = pca.score(X)                     # sklearn score: ONE float
print(f"transform(X) -> ISLR scores, shape {z.shape}, first row {np.round(z[0], 4)}")
print(f"score(X)     -> average log-likelihood, a single float: {ll:.4f}")

# --- Step 1: the noise variance is the mean of the DISCARDED eigenvalues --
X_c = X - X.mean(axis=0)
eigvals = np.sort(np.linalg.eigvalsh((X_c.T @ X_c) / (n - 1)))[::-1]
sigma2 = eigvals[k:].mean()
print(f"\nall eigenvalues:        {np.round(eigvals, 4)}")
print(f"kept (k={k}):             {np.round(eigvals[:k], 4)}")
print(f"mean of discarded:      {sigma2:.4f}  == pca.noise_variance_ {pca.noise_variance_:.4f}")

# --- Step 2: model covariance = kept directions at (lambda - sigma2),
#             plus sigma2 of isotropic noise in EVERY direction ------------
W = pca.components_.T                                 # (p, k) loading vectors as columns
C = (W * (pca.explained_variance_ - sigma2)) @ W.T + sigma2 * np.eye(p)
print(f"\nC == pca.get_covariance(): {np.allclose(C, pca.get_covariance())}")

# Sanity: along a kept direction the model variance is exactly lambda_i;
# along a discarded direction it is exactly sigma2.
eigvecs_all = np.linalg.eigh((X_c.T @ X_c) / (n - 1))[1][:, ::-1]
var_along = np.einsum("ij,jk,ki->i", eigvecs_all.T, C, eigvecs_all)  # v_i^T C v_i for every i
print(f"model variance along each PC direction: {np.round(var_along, 4)}")
print(f"  -> kept directions keep their eigenvalue, discarded ones are flattened to sigma2")

# --- Step 3: ordinary multivariate-normal log-density, N(mean_, C) --------
P = np.linalg.inv(C)
_, logdet_C = np.linalg.slogdet(C)
mahal = np.sum((X_c @ P) * X_c, axis=1)               # (x-mu)^T C^-1 (x-mu), vectorized
log_pdf = -0.5 * (p * np.log(2 * np.pi) + logdet_C + mahal)

print(f"\nmy log_pdf == pca.score_samples(X): {np.allclose(log_pdf, pca.score_samples(X))}")
print(f"my log_pdf.mean() == pca.score(X):  {np.isclose(log_pdf.mean(), pca.score(X))}")
```

### In Code — What It's For

```python
"""
What score is FOR: it gives PCA a scalar to maximize, which lets it plug into
model selection like any supervised estimator. Higher = held-out flowers are
more probable under the fitted Gaussian. Compare k = 1..4 on a train/test
split and watch the held-out likelihood -- and the reason it moves.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape

rng = np.random.default_rng(0)
idx = rng.permutation(n)
train, test = X[idx[:100]], X[idx[100:]]

print(f"{'k':>2}  {'noise_var':>9}  {'train score':>11}  {'test score':>10}")
for k in range(1, p + 1):
    pca = PCA(n_components=k).fit(train)
    print(f"{k:>2}  {pca.noise_variance_:>9.4f}  {pca.score(train):>11.4f}  {pca.score(test):>10.4f}")

# k = p-1 and k = p are the SAME Gaussian: with one eigenvalue discarded,
# sigma2 equals that eigenvalue, so C collapses back to the full sample
# covariance. score cannot tell them apart -- by construction, not by accident.

# --- Why an outlier scores low: residual off the kept subspace --------------
pca = PCA(n_components=1).fit(X)
z = pca.transform(X)
reconstruction = pca.inverse_transform(z)                # back to 4-D, lying on the PC1 line
residual = np.linalg.norm(X - reconstruction, axis=1)   # distance to the PC1 line
ll = pca.score_samples(X)

worst = np.argmin(ll)
best = np.argmax(ll)
print(f"\nlowest  log-lik flower #{worst}: residual off PC1 = {residual[worst]:.3f}, ll = {ll[worst]:.3f}")
print(f"highest log-lik flower #{best}: residual off PC1 = {residual[best]:.3f}, ll = {ll[best]:.3f}")
print(f"corr(residual, log-lik) across all flowers: {np.corrcoef(residual, ll)[0, 1]:.3f}")
```

### Common Misconceptions

- **"`score` returns the principal component scores."** No — `transform` does. `score` returns one float (mean log-likelihood); `score_samples` returns one log-likelihood per row. Same word, unrelated quantity.
- **"Higher `score` means more variance explained."** It means the held-out rows are more probable under the fitted Gaussian. Those usually move together, but `score` also penalizes a model whose $\sigma^2$ is too small for the residuals it sees, which is why held-out `score` can peak below $k=p$ and is usable for choosing $k$ — `explained_variance_ratio_` alone never can (it only ever increases with $k$).
- **"`score` can pick between $k=p-1$ and $k=p$."** It can't: with one eigenvalue discarded, $\sigma^2$ *is* that eigenvalue and $C$ collapses back to the full sample covariance, identical to the $k=p$ model. The two log-likelihoods are equal to machine precision (visible in the table above), by construction.
- **"`noise_variance_` is a fitted noise estimate."** Only in the PPCA sense: it is exactly the mean of the eigenvalues you threw away. If you keep everything it is 0 and the model degenerates to a plain Gaussian with the sample covariance — `score` then just measures Gaussian fit and says nothing about dimensionality.

---

## PCA With Missing Values (ISLR §12.3)

### Intuition

PCA needs a complete matrix, but the rank-$M$ reconstruction it produces can fill holes: if the data are close to an $M$-dimensional plane, a row with one hidden value is pinned down by where its other coordinates put it on that plane. ISLR's Algorithm 12.1 alternates the two steps: fill holes (starting from column means), compute the first $M$ PCs of the filled matrix, overwrite only the holes with the rank-$M$ reconstruction, repeat until the error on the observed cells stops decreasing. ISLR's footnote calls this "Hard-Impute"; the convex, shrinkage-based relative is soft-impute. Full treatment, including choosing $M$ by hiding observed cells: [[Matrix Completion and Soft-Impute]].

### Formal Definition

ISLR eq. 12.12, the PCA objective restricted to the observed pairs $\mathcal O$:

$$
\min_{A\in\mathbb R^{n\times M},\,B\in\mathbb R^{p\times M}}\sum_{(i,j)\in\mathcal O}\Big(x_{ij}-\sum_{m=1}^M a_{im}b_{jm}\Big)^2,
\qquad \hat x_{ij}=\sum_{m=1}^M\hat a_{im}\hat b_{jm}\ \text{ for }(i,j)\notin\mathcal O
$$

With complete data $\hat a_{im}=z_{im}$ (scores) and $\hat b_{jm}=\phi_{jm}$ (loadings), the usual PCA solution. With missing cells there is no closed form; Algorithm 12.1 decreases the objective each iteration but can stop at a local optimum.

### In Code

```python
"""
PCA with missing values: ISLR section 12.3, Algorithm 12.1, on Iris.
ISLR's protocol on USArrests (p = 4, like Iris): standardize, pick 20 random
observations, hide ONE random variable in each (so ~10% of 50 x 4 cells),
impute with M = 1 principal component, report the correlation between true
and imputed values over 100 runs. Here: Iris has 150 rows, so hide one
variable in 30 random flowers (5% of cells). Gold standard (unattainable in
practice): reconstruct the hidden cells from PC1 of the COMPLETE data.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

Xs_full = StandardScaler().fit_transform(load_iris().data)
n, p = Xs_full.shape


def hard_impute(Xs: np.ndarray, M: int, max_iter: int = 200, tol: float = 1e-7) -> np.ndarray:
    """ISLR Algorithm 12.1: column-mean start, rank-M PCA reconstruction, refill holes."""
    observed = ~np.isnan(Xs)
    filled = np.where(observed, Xs, np.nanmean(Xs, axis=0))
    prev_obj = np.inf
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(filled, full_matrices=False)
        approx = (U[:, :M] * s[:M]) @ Vt[:M]                   # step 2(a): first M PCs
        filled = np.where(observed, Xs, approx)                  # step 2(b): refill holes
        obj = np.sum((Xs[observed] - approx[observed]) ** 2)     # step 2(c): eq. 12.14
        if prev_obj - obj < tol * prev_obj:                      # stop when it fails to decrease
            break
        prev_obj = obj
    return filled


U, s, Vt = np.linalg.svd(Xs_full, full_matrices=False)
gold = s[0] * np.outer(U[:, 0], Vt[0])                           # z_i1 * phi_j1 on complete data

rng = np.random.default_rng(0)
r_impute, r_gold = [], []
for _ in range(100):
    rows = rng.choice(n, size=30, replace=False)
    cols = rng.integers(0, p, size=30)
    Xs = Xs_full.copy()
    Xs[rows, cols] = np.nan
    imputed = hard_impute(Xs, M=1)[rows, cols]
    truth = Xs_full[rows, cols]
    r_impute.append(np.corrcoef(imputed, truth)[0, 1])
    r_gold.append(np.corrcoef(gold[rows, cols], truth)[0, 1])

print(f"Algorithm 12.1, M = 1:         r = {np.mean(r_impute):.2f} +/- {np.std(r_impute):.2f}")
print(f"complete-data PC1 (gold std):  r = {np.mean(r_gold):.2f} +/- {np.std(r_gold):.2f}")
print("ISLR, USArrests:               r = 0.63 +/- 0.11  vs gold 0.79 +/- 0.08")
```

```text
Algorithm 12.1, M = 1:         r = 0.80 +/- 0.08
complete-data PC1 (gold std):  r = 0.86 +/- 0.06
ISLR, USArrests:               r = 0.63 +/- 0.11  vs gold 0.79 +/- 0.08
```

Iris imputes better than USArrests because PC1 of standardized Iris carries about 73% of the variance (vs about 62% for USArrests), so a single component pins a hidden value down more tightly.

### Common Misconceptions

- **"Impute with column means, then run PCA."** Mean imputation shrinks the variance of each incomplete column and weakens its correlations with the others, which biases the loadings. Iterating the fill with the PCA reconstruction is what fixes that.
- **"$M$ is whatever explains 90% of the variance."** For imputation, $M$ should be chosen by how well hidden *observed* cells are recovered (ISLR suggests exactly this); too large an $M$ fits noise in the observed cells and imputes worse.
