---
type: concept
title: "Eigenvalues and Eigenvectors"
libs:
  - numpy
  - scikit-learn
tags:
  - concept/math
related:
  - "[[Sklearn Datasets]]"
  - "[[Principal Component Analysis]]"
  - "[[Singular Value Decomposition]]"
created: 2026-09-12
updated: 2026-09-12
---

# Eigenvalues and Eigenvectors

> The eigenvalue equation $A\mathbf{v} = \lambda\mathbf{v}$ looks innocent, but the moment you ask "for which $\lambda$ does a *nonzero* $\mathbf{v}$ exist," it turns into a question about singular matrices — and that's the entire reason $\det(A - \lambda I) = 0$ shows up. This note covers why the determinant condition arises, how to solve the resulting characteristic polynomial (by hand for small cases, numerically for anything bigger), and why the same machinery is what powers PCA on real datasets.

---

## Why We Set the Determinant to Zero

### Intuition

An eigenvector of a square matrix $A$ is a **nonzero** vector $\mathbf{v}$ whose direction $A$ doesn't change — $A$ only rescales it: $A\mathbf{v} = \lambda\mathbf{v}$. The word "nonzero" is doing all the work. If you allowed $\mathbf{v} = \mathbf{0}$, then $A\mathbf{0} = \lambda\mathbf{0} = \mathbf{0}$ holds for *literally any* $\lambda$ and *any* matrix $A$. That statement is true but empty — it tells you nothing about $A$. So the definition of an eigenvector explicitly excludes $\mathbf{v} = \mathbf{0}$, and the real question becomes: **for which scalars $\lambda$ does $(A - \lambda I)\mathbf{v} = \mathbf{0}$ have a solution other than $\mathbf{v} = \mathbf{0}$?**

That reframes an eigenvalue problem as a question about homogeneous linear systems. A homogeneous system $M\mathbf{v} = \mathbf{0}$ always has the trivial solution $\mathbf{v} = \mathbf{0}$. It has *additional*, nontrivial solutions if and only if $M$ is singular (non-invertible). Why: if $M$ were invertible, you could left-multiply both sides by $M^{-1}$ and get $\mathbf{v} = M^{-1}\mathbf{0} = \mathbf{0}$, which rules out anything but the trivial answer. So a nontrivial null space requires $M$ to be singular — and "singular" is defined algebraically as $\det(M) = 0$.

Setting $\det(A - \lambda I) = 0$ is therefore not an arbitrary trick to memorize. It is the exact, necessary-and-sufficient algebraic condition for $(A - \lambda I)$ to have a nontrivial null space, which is exactly what "an eigenvector exists for this $\lambda$" means.

### Formal Definition

For a square matrix $M \in \mathbb{R}^{n \times n}$, the following are equivalent:

$$
\det(M) = 0 \iff M \text{ is singular} \iff \operatorname{rank}(M) < n \iff \exists\, \mathbf{v} \neq \mathbf{0} : M\mathbf{v} = \mathbf{0}
$$

Applying this to the eigenvalue equation:

$$
A\mathbf{v} = \lambda\mathbf{v},\ \mathbf{v} \neq \mathbf{0} \iff (A - \lambda I)\mathbf{v} = \mathbf{0},\ \mathbf{v} \neq \mathbf{0}
$$

By the equivalence above, such a nonzero $\mathbf{v}$ exists **if and only if** $\det(A - \lambda I) = 0$. Expanded, this is called the **characteristic equation**; its roots are the eigenvalues of $A$, and for each root $\lambda_i$ the corresponding eigenvector is any nonzero vector in $\operatorname{null}(A - \lambda_i I)$.

### In Code

```python
"""
Why det(A - lambda*I) = 0: a lambda that is NOT an eigenvalue leaves
(A - lambda*I) invertible, forcing the trivial solution v=0. A lambda
that IS an eigenvalue makes (A - lambda*I) singular, so a nontrivial
null-space vector exists. Uses the practice-problem matrix [[1,5],[2,4]].
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

A = np.array([[1.0, 5.0], [2.0, 4.0]])

# ── The characteristic equation for a 2x2: lambda^2 - trace*lambda + det = 0
trace = np.trace(A)
det = np.linalg.det(A)
eigenvalues = np.roots([1.0, -trace, det])
print(f"trace(A) = {trace}, det(A) = {det}")
print(f"Eigenvalues (roots of lambda^2 - {trace:.0f}*lambda + ({det:.0f}) = 0): "
      f"{sorted(eigenvalues)}")

# ── Case 1: lambda is NOT an eigenvalue -> (A - lambda*I) is invertible
lam_not_eigenvalue = 6.0001
M = A - lam_not_eigenvalue * np.eye(2)
print(f"\nlambda = {lam_not_eigenvalue} (not an eigenvalue):")
print(f"  det(A - lambda*I)  = {np.linalg.det(M):.6f}   (nonzero -> invertible)")
print(f"  rank(A - lambda*I) = {np.linalg.matrix_rank(M)}  (full rank)")
print("  -> Mv=0 forces v = M^-1 * 0 = 0. No nontrivial eigenvector.")

# ── Case 2: lambda IS an eigenvalue -> (A - lambda*I) is singular
lam_eigenvalue = 6.0
M = A - lam_eigenvalue * np.eye(2)
print(f"\nlambda = {lam_eigenvalue} (an eigenvalue):")
print(f"  det(A - lambda*I)  = {np.linalg.det(M):.6f}   (zero -> singular)")
print(f"  rank(A - lambda*I) = {np.linalg.matrix_rank(M)}  (rank-deficient)")

eigenvector = null_space(M).ravel()
print(f"  Nontrivial null-space vector: {np.round(eigenvector, 4)}")

# ── Sanity check against np.linalg.eig ──────────────────────────────
eigvals_np, eigvecs_np = np.linalg.eig(A)
print(f"\nnp.linalg.eig eigenvalues:  {eigvals_np}")
print(f"np.linalg.eig eigenvectors (columns):\n{eigvecs_np}")
```

Running this: for $\lambda = 6.0001$, $\det(A-\lambda I) \approx -0.0007$ and rank is 2 (full) — only $\mathbf{v}=\mathbf{0}$ solves the system. For $\lambda = 6$ (a true eigenvalue), $\det(A-\lambda I) = 0$, rank drops to 1, and a whole line of nonzero eigenvectors appears (direction $(1, 1)$).

---

## Solving the Characteristic Polynomial

### Intuition

For a $2\times2$ matrix, expanding $\det(A - \lambda I)$ gives a shortcut you can reuse for every $2\times2$ matrix without re-deriving it: the characteristic polynomial is always $\lambda^2 - \operatorname{trace}(A)\,\lambda + \det(A) = 0$. Only two numbers — trace and determinant — feed the quadratic formula. Beyond $2\times2$, hand expansion gets combinatorially messy fast (the general formula involves sums of all $k \times k$ principal minors), which is precisely why `np.linalg.eig` exists — you use the shortcut when it's convenient and defer to the numerical solver otherwise.

### Formal Definition

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$
\det(A - \lambda I) = (a-\lambda)(d-\lambda) - bc = \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \operatorname{trace}(A)\,\lambda + \det(A)
$$

$$
\lambda = \frac{\operatorname{trace}(A) \pm \sqrt{\operatorname{trace}(A)^2 - 4\det(A)}}{2}
$$

For the practice problem $A = \begin{pmatrix}1 & 5\\ 2 & 4\end{pmatrix}$: $\operatorname{trace}(A) = 5$, $\det(A) = 1\cdot4 - 5\cdot2 = -6$. So $\lambda^2 - 5\lambda - 6 = 0$, discriminant $= 25 + 24 = 49$, and $\lambda = \frac{5 \pm 7}{2} \Rightarrow \lambda = 6$ or $\lambda = -1$.

More generally, for an $n \times n$ matrix, $\det(A - \lambda I) = \lambda^n - c_1\lambda^{n-1} + c_2\lambda^{n-2} - \cdots \pm c_n$, where $c_k$ is the sum of **all** $k \times k$ principal minors of $A$ (a principal minor deletes the same set of rows and columns). $c_1 = \operatorname{trace}(A)$ and $c_n = \det(A)$ are just the two easy endpoints of that pattern.

### In Code

```python
"""
The 2x2 shortcut (trace, det) generalizes to sums of principal minors
for any n x n matrix. Built directly from the matrix entries -- not
from already knowing the eigenvalues -- then checked against
np.linalg.eigvals.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def characteristic_poly_coeffs(A: np.ndarray) -> np.ndarray:
    """Characteristic polynomial coefficients via principal minors.

    Returns [1, -c1, c2, -c3, ...] (highest degree first) for
    det(lambda*I - A) = lambda^n - c1*lambda^(n-1) + c2*lambda^(n-2) - ...
    where c_k is the sum of all k x k principal minors of A.
    This matches np.roots' expected coefficient order.
    """
    n = A.shape[0]
    coeffs = [1.0]
    for k in range(1, n + 1):
        minor_sum = sum(
            np.linalg.det(A[np.ix_(idx, idx)])
            for idx in combinations(range(n), k)
        )
        coeffs.append(((-1.0) ** k) * minor_sum)
    return np.array(coeffs)


# ── The practice problem, via the 2x2 shortcut ──────────────────────
A = np.array([[1.0, 5.0], [2.0, 4.0]])
coeffs_2x2 = characteristic_poly_coeffs(A)
print(f"2x2 characteristic polynomial coeffs: {coeffs_2x2}")
print(f"Roots: {sorted(np.roots(coeffs_2x2))}")
print(f"np.linalg.eigvals:  {sorted(np.linalg.eigvals(A))}")

# ── A 3x3 case with a repeated eigenvalue -- hand expansion is no
# longer a two-line shortcut, so lean on the principal-minor formula
B = np.array([
    [2.0, 0.0, 1.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 2.0],
])

coeffs_3x3 = characteristic_poly_coeffs(B)
print(f"\n3x3 characteristic polynomial coeffs: {np.round(coeffs_3x3, 4)}")
print(f"  -> lambda^3 {coeffs_3x3[1]:+.0f}*lambda^2 "
      f"{coeffs_3x3[2]:+.0f}*lambda {coeffs_3x3[3]:+.0f} = 0")

roots_3x3 = np.roots(coeffs_3x3)
print(f"Roots (eigenvalues):  {sorted(np.round(roots_3x3.real, 4))}")
print(f"np.linalg.eigvals(B): {sorted(np.round(np.linalg.eigvals(B).real, 4))}")
```

`B` is built so that row/column 2 is fully decoupled (zeros everywhere except the diagonal) while rows/columns 1 and 3 form a $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ block. That block alone has eigenvalues $2+1=3$ and $2-1=1$; the decoupled entry contributes a second eigenvalue of $3$. So $B$'s eigenvalues are $\{1, 3, 3\}$ — a repeated eigenvalue, which becomes the running example in the misconceptions below.

---

## Eigen-decomposition of a Real Covariance Matrix (PCA)

### Intuition

The same $\det(A - \lambda I) = 0$ condition that finds eigenvalues of a $2\times2$ toy example is exactly what powers PCA on real data. A covariance matrix is always symmetric, and the spectral theorem guarantees that every real symmetric matrix has $n$ *real* eigenvalues and a full set of *orthogonal* eigenvectors — no complex roots, no missing eigenvectors, unlike the general case. `sklearn.decomposition.PCA` isn't doing anything exotic under the hood: it centers the data, and its principal components and explained variances are exactly the eigenvectors and eigenvalues of the covariance matrix, sorted by eigenvalue in descending order.

### Formal Definition

Given centered data $X_c \in \mathbb{R}^{n \times p}$ (column means zero), the covariance matrix is

$$
\Sigma = \frac{1}{n-1} X_c^T X_c \in \mathbb{R}^{p \times p}, \quad \Sigma = \Sigma^T
$$

By the spectral theorem, $\Sigma = V\Lambda V^T$ where $V$ is orthogonal ($V^TV = I$) and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_p)$ with all $\lambda_i \geq 0$. The columns of $V$ (sorted by $\lambda_i$ descending) are the principal components; $\lambda_i$ is the variance captured along the $i$-th component; $\sum_i \lambda_i = \operatorname{trace}(\Sigma)$ is the total variance.

### In Code

```python
"""
The covariance matrix of any dataset is symmetric, so det(Sigma - lambda*I) = 0
always has p REAL roots (spectral theorem). Those roots are exactly what
sklearn's PCA reports as explained_variance_. Uses the Iris dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data  # (150, 4): sepal/petal length & width, see [[Sklearn Datasets]]

# ── Manual eigen-decomposition of the covariance matrix ─────────────
X_centered = X - X.mean(axis=0)
cov = (X_centered.T @ X_centered) / (X_centered.shape[0] - 1)

# eigh (not eig): exploits symmetry, guarantees real eigenvalues, and
# returns them in ASCENDING order.
eigvals, eigvecs = np.linalg.eigh(cov)

order = np.argsort(eigvals)[::-1]          # descending, PCA convention
eigvals_sorted = eigvals[order]
eigvecs_sorted = eigvecs[:, order]

# ── Sklearn's PCA on the same data ──────────────────────────────────
pca = PCA(n_components=4).fit(X)

print("Manual eigenvalues:          ", np.round(eigvals_sorted, 4))
print("PCA explained_variance_:     ", np.round(pca.explained_variance_, 4))
print("Total variance (trace(cov)): ", round(np.trace(cov), 4))
print("Sum of eigenvalues:          ", round(eigvals_sorted.sum(), 4))

# Eigenvectors and PCA components can each flip sign (+-v both solve
# Sigma*v = lambda*v), so compare magnitudes.
max_diff = np.max(np.abs(np.abs(eigvecs_sorted.T) - np.abs(pca.components_)))
print(f"Max |eigenvector| vs |PCA component| difference: {max_diff:.2e}")

explained_ratio = eigvals_sorted / eigvals_sorted.sum()
print("Explained variance ratio (manual):", np.round(explained_ratio, 4))
print("PCA explained_variance_ratio_:    ", np.round(pca.explained_variance_ratio_, 4))
```

This is the bridge from the toy $2\times2$ problem to [[Principal Component Analysis]]: the "find $\lambda$ such that $\det(A-\lambda I)=0$" question and "find the directions of maximum variance in my data" question are the same linear-algebra question, just applied to a covariance matrix instead of an arbitrary $2\times2$.

---

## Common Misconceptions

- **"Setting $\det(A-\lambda I)=0$ is just an algebra trick to memorize."** No — it is the precise necessary-and-sufficient condition for $(A-\lambda I)$ to be singular, which is required for $(A-\lambda I)\mathbf{v}=\mathbf{0}$ to have a nonzero solution. See the first section above.
- **"$\mathbf{v}=\mathbf{0}$ counts as an eigenvector."** No. $\mathbf{v}=\mathbf{0}$ trivially satisfies $A\mathbf{v}=\lambda\mathbf{v}$ for *every* $\lambda$ and *every* $A$, so it carries zero information about the matrix. The definition of an eigenvector explicitly requires $\mathbf{v}\neq\mathbf{0}$.
- **"$\det(A-\lambda I)=0$ means $A$ itself is singular."** No — it means $(A-\lambda I)$ is singular. $A$ itself is only singular when $0$ is one of its eigenvalues, since $\det(A) = \det(A - 0\cdot I)$.
- **"Every square matrix is diagonalizable."** No. Diagonalizability needs $n$ linearly independent eigenvectors. A repeated eigenvalue doesn't automatically break this: the $B$ matrix in the "Solving the Characteristic Polynomial" section has eigenvalue $3$ with multiplicity 2, yet $B$ is still diagonalizable because it's symmetric (spectral theorem guarantees a full orthogonal eigenbasis regardless of multiplicity). A general, non-symmetric matrix with a repeated eigenvalue can instead be "defective" — it lacks enough independent eigenvectors to diagonalize.
- **"Eigenvalues must be real numbers."** Only guaranteed for symmetric (more generally, normal) matrices. A general real matrix can have complex-conjugate eigenvalue pairs — a 2D rotation matrix (by anything other than $0°$/$180°$) has no real eigenvector at all, since no direction survives a rotation unchanged.
- **"Every matrix has eigenvalues, so every matrix has an eigen-decomposition."** Eigenvalues are only defined for square matrices. Rectangular matrices don't have eigenvalues — the corresponding tool there is the [[Singular Value Decomposition]], which factors *any* matrix shape using singular values instead.
