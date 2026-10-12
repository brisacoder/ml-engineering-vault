---
type: concept
title: "Singular Value Decomposition"
libs:
  - numpy
  - scikit-learn
tags:
  - concept/math
related:
  - "[[Eigenvalues and Eigenvectors]]"
  - "[[Principal Component Analysis]]"
  - "[[Ridge Regression]]"
  - "[[Sklearn Datasets]]"
  - "[[Matrix Completion and Soft-Impute]]"
created: 2026-09-25
updated: 2026-10-12
---

# Singular Value Decomposition

> Eigendecomposition asks "which directions does this matrix leave pointing the same way?" — a question that only makes sense when a matrix maps a space to itself. SVD asks a question that works for any matrix of any shape: "which input directions does it send to which output directions, and with what gain?" For a symmetric matrix the two answers coincide; in general they don't, and knowing exactly how they're related is what makes the PCA-via-SVD connection non-magical. Every example below runs on the centered Iris matrix.

---

## How Singular Vectors Relate to Eigenvectors

### Intuition

An SVD of $A$ is two eigendecompositions happening at once, on two *different* symmetric matrices, glued together by $A$ itself. $A^TA$ lives in the input space and its eigenvectors are the right singular vectors $V$. $AA^T$ lives in the output space and its eigenvectors are the left singular vectors $U$. Both matrices share the same nonzero eigenvalues, $s_i^2$. And $A$ is precisely the map that carries each $v_i$ onto the matching $u_i$, stretched by $s_i$.

The crucial difference from an eigenvector: an eigenvector is mapped onto *itself* ($Av = \lambda v$, same vector, same space). A right singular vector is mapped onto a *different* vector in a *different* space ($Av_i = s_i u_i$). That's why eigenvectors need a square matrix and singular vectors don't.

### Formal Definition

For $A \in \mathbb{R}^{n\times p}$ with economy SVD $A = USV^T$ ($U \in \mathbb{R}^{n\times p}$, $V\in\mathbb{R}^{p\times p}$, both with orthonormal columns; $S=\operatorname{diag}(s_1\geq\cdots\geq s_p\geq 0)$):

$$
A^TA = VS^2V^T \qquad\qquad AA^T = US^2U^T
$$

so $V$ = eigenvectors of $A^TA$, $U$ = eigenvectors of $AA^T$ (for the $p$ nonzero eigenvalues; the remaining $n-p$ eigenvalues of $AA^T$ are zero), and $\lambda_i = s_i^2$ for both. The two sets are linked by

$$
Av_i = s_i u_i \qquad\qquad A^Tu_i = s_i v_i
$$

For the centered data matrix $X_c$ specifically, $X_c^TX_c/(n-1)$ is the covariance matrix, so $V$ is exactly the PCA loading matrix and $s_i^2/(n-1)$ the PC variances — the identity used throughout [[Principal Component Analysis]]. $X_cX_c^T$ is the $n\times n$ Gram matrix of similarities *between observations*; its eigenvectors $U$ are the normalized PC scores (the "dual" PCA).

### In Code

```python
"""
SVD of a matrix A is two eigendecompositions at once, glued together:
  right singular vectors V = eigenvectors of A.T @ A   (feature space, p x p)
  left  singular vectors U = eigenvectors of A @ A.T   (observation space, n x n)
  both share the same nonzero eigenvalues, s_i**2,
  and A maps each v_i onto the matching u_i:  A @ v_i = s_i * u_i.
Shown on the centered Iris matrix (150 x 4).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data
A = X - X.mean(axis=0)  # centered data matrix, n x p
n, p = A.shape

U, s, Vt = np.linalg.svd(A, full_matrices=False)  # U: n x p, s: (p,), Vt: p x p
V = Vt.T

# --- Right singular vectors = eigenvectors of A.T @ A (p x p) -----------
eigvals_AtA, eigvecs_AtA = np.linalg.eigh(A.T @ A)
order = np.argsort(eigvals_AtA)[::-1]
eigvals_AtA, eigvecs_AtA = eigvals_AtA[order], eigvecs_AtA[:, order]

print("s**2:                 ", np.round(s ** 2, 3))
print("eigenvalues of A.T@A: ", np.round(eigvals_AtA, 3))
print("V == eigvecs(A.T@A) up to sign:",
      np.allclose(np.abs(V), np.abs(eigvecs_AtA)))

# --- Left singular vectors = eigenvectors of A @ A.T (n x n) -------------
# A @ A.T is 150 x 150 but has rank p = 4, so only 4 nonzero eigenvalues;
# the other 146 are numerically zero.
eigvals_AAt, eigvecs_AAt = np.linalg.eigh(A @ A.T)
order = np.argsort(eigvals_AAt)[::-1]
eigvals_AAt, eigvecs_AAt = eigvals_AAt[order], eigvecs_AAt[:, order]

print("\ntop-4 eigenvalues of A@A.T:", np.round(eigvals_AAt[:p], 3))
print("5th eigenvalue (rank deficiency):", f"{eigvals_AAt[p]:.2e}")
print("U == top-4 eigvecs(A@A.T) up to sign:",
      np.allclose(np.abs(U), np.abs(eigvecs_AAt[:, :p])))

# --- The glue: A sends each right vector onto the matching left vector ---
# A @ v_i = s_i * u_i  <=>  A @ V = U @ diag(s)
print("\nA @ V == U * s:", np.allclose(A @ V, U * s))
# ...and A.T sends each left vector back:  A.T @ u_i = s_i * v_i
print("A.T @ U == V * s:", np.allclose(A.T @ U, V * s))
```

### Common Misconceptions

- **"Singular vectors are just eigenvectors of $A$."** Only when $A$ is symmetric positive semi-definite. In general they are eigenvectors of $A^TA$ and $AA^T$ — two other matrices — and $A$ itself may have no real eigenvectors at all (see the rotation example below).
- **"$U$ and $V$ are interchangeable."** They live in different spaces: $V$ in $\mathbb{R}^p$ (one entry per feature), $U$ in $\mathbb{R}^n$ (one entry per observation). For a data matrix, $V$ is the loadings, $US$ is the scores. This is the same row-vs-column distinction as $x_i$ vs $X_j$ in [[Principal Component Analysis#Quick Reference: ISLR Terms ↔ Linear Algebra]].
- **"Sign is determined."** As with eigenvectors, $(u_i, v_i)$ and $(-u_i, -v_i)$ are equally valid — the pair must flip together to keep $Av_i = s_iu_i$. Compare magnitudes when checking two implementations against each other.

---

## Physical Interpretation: Input Modes, Output Modes, Gains

### Intuition

The physics picture for eigenvectors is a system mapping a space to itself — a stiffness matrix $K$ acting on displacements, say. The eigenvectors are the *mode shapes* (the patterns that vibrate without changing shape), and the eigenvalues give the natural frequencies ($\omega_i^2 = \lambda_i$ for $Kx = \omega^2 M x$ with $M=I$). Worth being precise here: the *eigenvector* is the shape, the *eigenvalue* is the frequency; they are two different objects.

That picture requires input and output to be the same kind of thing (displacement in, force out, both living in the same $\mathbb{R}^n$). SVD is the generalization to a system where they're not — an operator from an input space to a genuinely different output space. Then:

- $v_i$ is a principal **input direction**: an input pattern the system responds to cleanly, with no cross-talk into the other patterns.
- $u_i$ is the corresponding principal **output direction**: the shape of the response that input produces.
- $s_i$ is the **gain**: how much that input pattern is amplified on its way through.

$A = USV^T$ then reads as three physical steps: rotate the input to express it in the $v$-basis ($V^T$), amplify each component by its gain ($S$), rotate the result into the output space along the $u$-basis ($U$). $s_1$ is the largest gain the system can deliver to any unit input, and $v_1$ is the input that achieves it — this is exactly the "principal gain" of a MIMO system in control theory, and why $s_1$ is the operator 2-norm. When the operator is symmetric the input and output spaces coincide, $U=V$, and the gain picture collapses back into the resonant-mode picture: they are one and the same in that special case.

For a data matrix the physical reading becomes: $A = \sum_i s_i\, u_i v_i^T$, a sum of rank-1 layers. Each layer is one *mode of variation*: $v_i$ says which combination of features moves together, $u_i$ says how much each observation exhibits that pattern, $s_i$ says how much of the data's total energy that mode carries.

### In Code

```python
"""
What U, s, V mean physically for a linear map A: R^p -> R^n.
  v_i : the i-th principal INPUT direction  (orthonormal basis of the domain)
  u_i : the OUTPUT direction that input produces (orthonormal basis of the range)
  s_i : the GAIN -- how much that input pattern is amplified on its way out.
Unlike an eigenvector, v_i is NOT mapped onto itself; it lands on a
different vector in a different space. s_1 is the largest gain any unit
input can achieve, and v_1 is the input that achieves it.
Centered Iris again: A is 150 x 4, so inputs live in R^4, outputs in R^150.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data
A = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
V = Vt.T

# --- Gain of the principal input direction vs. random unit inputs ---------
rng = np.random.default_rng(0)
random_inputs = rng.standard_normal((4, 10_000))
random_inputs /= np.linalg.norm(random_inputs, axis=0)  # unit-norm columns

gains = np.linalg.norm(A @ random_inputs, axis=0)  # ||A x|| for each unit x
print(f"s_1 (largest singular value):              {s[0]:.4f}")
print(f"max ||A x|| over 10,000 random unit x:      {gains.max():.4f}  (never exceeds s_1)")
print(f"||A v_1||:                                  {np.linalg.norm(A @ V[:, 0]):.4f}  (equals s_1)")
print(f"s_p (smallest singular value):              {s[-1]:.4f}")
print(f"min ||A x|| over 10,000 random unit x:      {gains.min():.4f}  (never below s_p)")

# --- A as rotate -> stretch -> rotate ------------------------------------
# V.T rotates an input into the v-basis, diag(s) stretches each axis,
# U rotates the result into the output space. The three factors multiply
# back to A exactly.
print("\nU @ diag(s) @ V.T == A:", np.allclose(U @ np.diag(s) @ Vt, A))

# --- A as a sum of rank-1 layers: pattern x who-has-it x strength ---------
# Each layer s_i * u_i v_i.T is one "mode": v_i says WHICH feature
# combination, u_i says HOW MUCH each of the 150 flowers has of it,
# s_i says how strong the mode is overall.
layers = s[:, None, None] * (U.T[:, :, None] * Vt[:, None, :])  # (p, n, p)
print("sum of all p rank-1 layers == A:", np.allclose(layers.sum(axis=0), A))

energy = s ** 2 / np.sum(s ** 2)
print("fraction of ||A||_F^2 carried by each layer:", np.round(energy, 4))
```

### Common Misconceptions

- **"Singular values are frequencies."** No — the frequency analogy belongs to eigenvalues of a self-map. Singular values are *gains*: ratios of output magnitude to input magnitude along the matched direction pair. The largest one is the operator norm, the smallest measures how close the matrix is to being rank-deficient.
- **"$v_1$ is the direction the data points in."** $v_1$ is a direction in *feature space* ($\mathbb{R}^p$): the combination of features along which the observations are most spread. Which *observations* sit far out along it is a different question, answered by $u_1$ in $\mathbb{R}^n$. "Which features co-vary?" and "which samples are extreme?" are two questions, two vectors.

---

## When SVD and Eigendecomposition Coincide — and When They Don't

### Intuition

For a symmetric positive semi-definite matrix (every covariance matrix is one) the input and output spaces are the same, $U=V$, and $s_i=\lambda_i$: the SVD *is* the eigendecomposition. Move away from symmetric and the two split: a rotation matrix has no real eigenvectors at all yet has a perfectly good SVD (all singular values 1, since rotation preserves lengths); a non-normal matrix has eigenvectors that aren't orthogonal and singular values that differ from its eigenvalues. Singular vectors are the robust object — always real, always orthonormal, always present.

### Formal Definition

If $A=A^T$ and $A\succeq 0$: $A = Q\Lambda Q^T$ is simultaneously an eigendecomposition and an SVD with $U=V=Q$, $S=\Lambda$. If $A$ is merely symmetric with some $\lambda_i<0$: $s_i=|\lambda_i|$ and $u_i = \operatorname{sign}(\lambda_i)\,v_i$. If $A$ is *normal* ($AA^T=A^TA$): $s_i = |\lambda_i|$ still holds, but eigenvectors may be complex. If $A$ is not normal: no simple relation between $\{s_i\}$ and $\{\lambda_i\}$ beyond $\prod s_i = |\prod\lambda_i| = |\det A|$.

### In Code

```python
"""
When SVD and eigendecomposition coincide, and when they don't.
  symmetric PSD matrix (e.g. a covariance matrix): U == V == eigenvectors,
      s_i == lambda_i. The 'resonant mode' picture and the 'input/output
      mode' picture are the same picture.
  general square matrix: eigenvectors may be non-orthogonal, complex, or
      missing entirely, and |eigenvalue| != singular value. Singular vectors
      are always real, orthonormal, and exist for every matrix of any shape.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data
A = X - X.mean(axis=0)
n, p = A.shape
cov = (A.T @ A) / (n - 1)  # symmetric, positive semi-definite

# --- Case 1: symmetric PSD -> SVD IS the eigendecomposition --------------
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

U_c, s_c, Vt_c = np.linalg.svd(cov)
print("covariance: singular values == eigenvalues:", np.allclose(s_c, eigvals))
print("covariance: U == V (same orthogonal matrix):", np.allclose(np.abs(U_c), np.abs(Vt_c.T)))
print("covariance: U == eigenvectors:              ", np.allclose(np.abs(U_c), np.abs(eigvecs)))

# --- Case 2: a rotation -- no real eigenvectors at all -------------------
# Rotate the (sepal length, sepal width) plane by 90 degrees: no direction
# survives unchanged, so the eigenvalues are complex. Singular values are
# all 1 (a rotation changes no lengths) and U, V are real.
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

print("\nrotation eigenvalues:     ", np.round(np.linalg.eigvals(R), 4), " (complex: no real modes)")
print("rotation singular values: ", np.round(np.linalg.svd(R, compute_uv=False), 4), "(real, all 1)")

# --- Case 3: non-normal square matrix -> the two decompositions diverge --
# Scale the covariance matrix's COLUMNS by 1/sd (one-sided standardizing).
# Still a perfectly ordinary real 4x4 matrix, but no longer symmetric or
# normal (B @ B.T != B.T @ B), so eigen- and singular structure split apart.
B = cov @ np.diag(1 / X.std(axis=0, ddof=1))
eigvals_B, eigvecs_B = np.linalg.eig(B)
U_B, s_B, Vt_B = np.linalg.svd(B)

print("\nB normal? (B B^T == B^T B):        ", np.allclose(B @ B.T, B.T @ B))
print("B eigenvalues (real here):          ", np.round(np.sort(eigvals_B.real)[::-1], 4))
print("B singular values:                  ", np.round(s_B, 4), " <- not the eigenvalues")
print("B eigenvectors orthonormal?         ", np.allclose(eigvecs_B.T @ eigvecs_B, np.eye(p)))
print("B singular vectors orthonormal?     ", np.allclose(Vt_B @ Vt_B.T, np.eye(p)))
print("largest |v_i . v_j| between distinct eigenvectors:",
      f"{np.max(np.abs(eigvecs_B.T @ eigvecs_B - np.eye(p))):.3f}", "(0 would mean orthogonal)")
```

### Common Misconceptions

- **"Every square matrix has an eigenbasis, so eigendecomposition is always available."** Defective matrices don't; rotation matrices have only complex eigenvectors. SVD never has either problem — which is why it is the numerically safe default in libraries (the `svd` solver in [[Ridge Regression]], `sklearn.decomposition.PCA` for wide data). For a symmetric PSD product like a covariance matrix, though, `eigh` is perfectly safe and much cheaper when $n \gg p$, and sklearn's PCA does exactly that on tall-skinny data (`covariance_eigh`, see [[Principal Component Analysis]]).
- **"If the singular values match the eigenvalues, the matrix is symmetric."** Matching happens for all *normal* matrices (orthogonal, unitary, symmetric, skew-symmetric, ...). Symmetric is sufficient, not necessary.

---

## Truncation vs Soft-Thresholding: Two Ways to Shrink a Spectrum

### Intuition

Once a matrix is written as $USV^T$, "simplify it" means "edit the diagonal" and leave $U$, $V$ alone. There are two standard edits. **Hard truncation** keeps the top $k$ gains untouched and silences the rest; it is the best rank-$k$ approximation (Eckart-Young), the operation behind PCA reconstruction and ISLR's hard-impute. **Soft-thresholding** subtracts the same $\lambda$ from every gain and floors at zero; weak modes vanish and strong modes are dampened. It is the operation behind soft-impute ([[Matrix Completion and Soft-Impute]]), and it is Lasso's soft-threshold ([[Lasso Regression]]) applied to the vector of singular values.

### Formal Definition

$$
\text{hard: } A_k=\sum_{i\le k}s_iu_iv_i^T=\arg\min_{\operatorname{rank}(Z)\le k}\lVert A-Z\rVert_F,\qquad \lVert A-A_k\rVert_F=\sqrt{\textstyle\sum_{i>k}s_i^2}
$$

$$
\text{soft: } S_\lambda(A)=\sum_i\max(s_i-\lambda,0)\,u_iv_i^T=\arg\min_Z\tfrac12\lVert A-Z\rVert_F^2+\lambda\lVert Z\rVert_*,\qquad \lVert Z\rVert_*=\textstyle\sum_i s_i(Z)
$$

The first problem is non-convex (the rank constraint), the second is convex (the nuclear norm is a norm). Both answers use the same singular vectors.

### In Code

```python
"""
Two ways to shrink a spectrum, on the centered Iris matrix.
  hard (truncated SVD): keep s_1..s_k unchanged, drop the rest.
      Eckart-Young: the best rank-k approximation in Frobenius norm.
  soft (singular value thresholding): s_i -> max(s_i - lam, 0).
      The exact minimizer of 0.5*||Z - A||_F^2 + lam*||Z||_*  (nuclear norm).
Same singular vectors in both; only the diagonal changes.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X = load_iris().data
A = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("singular values:", np.round(s, 3))

# --- Hard: Eckart-Young. No rank-2 matrix gets closer to A ----------------
k = 2
A_hard = (U[:, :k] * s[:k]) @ Vt[:k]
err_hard = np.linalg.norm(A - A_hard)
print(f"\nrank-{k} truncation error = {err_hard:.4f} = sqrt(s_3^2 + s_4^2) = {np.sqrt(np.sum(s[k:] ** 2)):.4f}")

rng = np.random.default_rng(0)
best_random = min(
    np.linalg.norm(A - (A @ Q) @ Q.T)                       # project A onto a random 2-D subspace
    for Q in (np.linalg.qr(rng.standard_normal((4, k)))[0] for _ in range(5000))
)
print(f"best of 5000 random rank-{k} projections = {best_random:.4f}  (never below truncation)")

# --- Soft: same vectors, every kept value reduced by lam --------------------
lam = 5.0
s_soft = np.maximum(s - lam, 0.0)
A_soft = (U * s_soft) @ Vt
print(f"\nlam = {lam}: s -> {np.round(s_soft, 3)}  (rank {np.count_nonzero(s_soft)})")
print("singular values of A_soft:", np.round(np.linalg.svd(A_soft, compute_uv=False), 3))
print(f"nuclear norm: A = {s.sum():.3f}, A_soft = {s_soft.sum():.3f}  (survivors lose lam each, the rest vanish)")

# Both live in the same subspace: the leading right singular vectors agree
V_soft = np.linalg.svd(A_soft, full_matrices=False)[2]
print("same leading directions:", np.allclose(np.abs(V_soft[:2]), np.abs(Vt[:2])))
```

```text
singular values: [25.1    6.013  3.414  1.885]

rank-2 truncation error = 3.8993 = sqrt(s_3^2 + s_4^2) = 3.8993
best of 5000 random rank-2 projections = 4.4054  (never below truncation)

lam = 5.0: s -> [20.1    1.013  0.     0.   ]  (rank 2)
singular values of A_soft: [20.1    1.013  0.     0.   ]
nuclear norm: A = 36.411, A_soft = 21.113  (survivors lose lam each, the rest vanish)
same leading directions: True
```

### Common Misconceptions

- **"Truncation is the regularized version."** Truncation restricts rank but applies no shrinkage to what it keeps; soft-thresholding is the one with a penalty term. They behave differently when fitted to noisy or partially observed data: see the hard-impute vs soft-impute comparison in [[Matrix Completion and Soft-Impute#Soft-Impute vs Hard-Impute (ISLR Algorithm 12.1)]].
- **"Eckart-Young needs a square matrix."** It holds for any shape, in both the Frobenius and the spectral (2-) norm.
