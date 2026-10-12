---
type: concept
title: "Matrix Product Transpose and Inverse Rules"
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
created: 2026-09-21
updated: 2026-09-21
---

# Matrix Product Transpose and Inverse Rules

> Formulas built from $(W^TW)^{-1}W^T$ — the normal-equations operator behind OLS, and the shape you'll meet again in whitening transforms and generalized eigenvalue problems — trip up two different "does the order flip?" rules at once. Transposing a product always reverses factor order, no matter the matrix's shape or rank. Inverting a product also reverses order, but *only* when every factor is individually square and invertible — and that precondition is exactly what fails for a rectangular design matrix, which is the whole reason $(W^TW)^{-1}W^T$ gets written the long way instead of just $W^{-1}$. This note derives both rules, shows where the inverse rule's precondition silently breaks, and works through the combined $A^{-T}$ notation where the two reversals cancel each other out.

---

## Transpose of a Product Reverses Order — Unconditionally

### Intuition

$(AB)^T = B^TA^T$ holds no matter what $A$ and $B$ look like — rectangular, singular, whatever — as long as $AB$ itself is defined. That's a stronger guarantee than it sounds, because it means you never have to check invertibility or squareness before flipping a transpose across a product. The reason it's unconditional: the proof only ever uses that scalar multiplication commutes, which is true for every real (or complex) number, full stop.

### Formal Definition

For $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$:

$$
((AB)^T)_{ij} = (AB)_{ji} = \sum_{k=1}^{n} A_{jk}B_{ki} = \sum_{k=1}^{n} B_{ki}A_{jk} = \sum_{k=1}^{n} (B^T)_{ik}(A^T)_{kj} = (B^TA^T)_{ij}
$$

The middle step just swaps the order of the scalars $A_{jk}$ and $B_{ki}$ inside the sum — always legal — then relabels them as entries of $B^T$ and $A^T$. Nothing here required $A$ or $B$ to be square, full rank, or invertible. It generalizes directly to any chain:

$$
(A_1A_2\cdots A_n)^T = A_n^T\cdots A_2^TA_1^T
$$

### In Code

```python
"""
Transpose of a product reverses factor order -- (AB)^T = B^T A^T --
and this holds unconditionally, even when neither factor is square or
invertible. Demonstrated on the Iris PCA projection: the centered data
matrix X_c (150x4, rectangular) times the loading matrix W (4x2, also
rectangular). See [[Principal Component Analysis]] for what W means.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X: np.ndarray = load_iris().data                 # (150, 4)
X_c: np.ndarray = X - X.mean(axis=0)               # centered, still (150, 4)

pca = PCA(n_components=2).fit(X)
W: np.ndarray = pca.components_.T                  # (4, 2): loadings as columns

Z: np.ndarray = X_c @ W                             # (150, 2): PCA scores

lhs: np.ndarray = Z.T                               # (X_c @ W)^T
rhs: np.ndarray = W.T @ X_c.T                       # W^T @ X_c^T

print(f"X_c shape: {X_c.shape}  (not square -> no ordinary inverse exists)")
print(f"W shape:   {W.shape}   (not square -> no ordinary inverse exists either)")
print(f"(X_c @ W)^T shape: {lhs.shape}, W^T @ X_c^T shape: {rhs.shape}")
print(f"Max difference: {np.max(np.abs(lhs - rhs)):.2e}")
```

### Common Misconceptions

- **"The transpose-reversal rule only applies to square matrices."** No — the proof above never assumes $m=n$. It's demonstrated here on a genuinely rectangular $150\times4$ matrix times a rectangular $4\times2$ matrix.
- **"If neither factor is invertible, you can't say anything about the transpose of their product."** Invertibility never entered the derivation. Transpose reversal is a statement about index bookkeeping, not about rank or singularity — it's a fundamentally different kind of fact than the inverse rule below.

---

## Inverse of a Product Reverses Order — Only If Every Factor Is Invertible

### Intuition

$(AB)^{-1} = B^{-1}A^{-1}$ looks structurally identical to the transpose rule, but it carries a real precondition: $A^{-1}$ and $B^{-1}$ have to actually exist, which means $A$ and $B$ must each be square and full rank. Drop that condition and the rule doesn't just fail to simplify — it becomes a statement about objects that don't exist.

### Formal Definition

If $A, B \in \mathbb{R}^{n\times n}$ are each invertible:

$$
(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AIA^{-1} = AA^{-1} = I
$$

so $B^{-1}A^{-1}$ satisfies the defining property of $(AB)^{-1}$, and by uniqueness of inverses, $(AB)^{-1}=B^{-1}A^{-1}$. This generalizes to a chain: $(A_1A_2\cdots A_n)^{-1} = A_n^{-1}\cdots A_2^{-1}A_1^{-1}$, provided every $A_i$ is individually invertible.

The subtlety worth sitting with: **a product $AB$ can be square and perfectly invertible even when $A$ and $B$ are not even square themselves.** The classic case is a Gram matrix $X^TX$, where $X$ is $n\times p$ with $n>p$ (more rows than columns). $X$ has no inverse — it isn't square — yet $X^TX$, a $p\times p$ matrix, generically is invertible. There is no way to "split" $(X^TX)^{-1}$ into $(X^T)^{-1}(X)^{-1}$, because neither of those exists on its own.

### In Code

```python
"""
Inverse of a product reverses order -- (AB)^-1 = B^-1 A^-1 -- but only
when A and B are each individually square and invertible. Demonstrated
on two real 4x4 matrices built from Iris (covariance and correlation),
then contrasted with the Gram-matrix case where the whole product is
invertible even though NEITHER individual factor is even square.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X: np.ndarray = load_iris().data
X_c: np.ndarray = X - X.mean(axis=0)
n_samples: int = X_c.shape[0]

A: np.ndarray = (X_c.T @ X_c) / (n_samples - 1)     # covariance matrix, (4, 4)
stds: np.ndarray = np.sqrt(np.diag(A))
B: np.ndarray = A / np.outer(stds, stds)             # correlation matrix, (4, 4)

# ── Rule holds: both A and B are square and invertible ──────────────
lhs: np.ndarray = np.linalg.inv(A @ B)
rhs: np.ndarray = np.linalg.inv(B) @ np.linalg.inv(A)
print(f"(AB)^-1 vs B^-1 @ A^-1 max diff: {np.max(np.abs(lhs - rhs)):.2e}")

# ── Contrast: X_c itself (150x4) has no inverse at all ───────────────
try:
    np.linalg.inv(X_c)
except np.linalg.LinAlgError as e:
    print(f"\nX_c^-1 does not exist: {e}")

# Yet the Gram matrix X_c^T @ X_c, built from two non-square,
# non-invertible factors, is perfectly invertible:
gram: np.ndarray = X_c.T @ X_c
print(f"\nX_c^T @ X_c is square and invertible: shape={gram.shape}, "
      f"rank={np.linalg.matrix_rank(gram)}")
print("But (X_c^T @ X_c)^-1 CANNOT be split into (X_c^T)^-1 @ (X_c)^-1 "
      "-- neither exists.")
```

### Common Misconceptions

- **"If $AB$ is invertible, then $A$ and $B$ must individually be invertible."** True only when $A$ and $B$ are both square of the same size — and even then it needs its own short proof ($\det(AB)=\det(A)\det(B)\neq0$ forces both determinants nonzero). For rectangular factors the statement doesn't even parse: determinants aren't defined for non-square matrices, and the Gram-matrix example above shows the product can be invertible while neither factor is square at all.
- **"You can always split $(AB)^{-1}$ into $B^{-1}A^{-1}$ as long as $AB$ is invertible."** No — the split requires each factor's *own* inverse to exist, which is a strictly stronger condition than the product being invertible.

---

## The Pseudoinverse Trap: $(X^TX)^{-1}X^T$ Is Not $X^{-1}X^{-T}$

### Intuition

$(X^TX)^{-1}X^T$ is the ordinary least squares normal-equations operator — it's exactly what you'd get by naively applying the inverse-reversal rule from the section above to split $(X^TX)^{-1}$, then stopping one step short of also splitting off the leftover $X^T$. It's tempting to keep going and rewrite it as $X^{-1}X^{-T}X^T$, but that requires $X^{-1}$ to exist, and $X$ — a genuine data matrix with more rows (samples) than columns (features) — is rectangular by construction. The whole reason anyone writes $(X^TX)^{-1}X^T$ instead of just $X^{-1}$ is *because* $X^{-1}$ doesn't exist. This expression has a name: it's the Moore–Penrose **left pseudoinverse**, $X^+ = (X^TX)^{-1}X^T$, and $\alpha=0$ in [[Ridge Regression]]'s $(X^TX+\alpha I)^{-1}X^T$ collapses to exactly this.

### Formal Definition

For $X\in\mathbb{R}^{n\times p}$ with $n>p$ and full column rank $p$, the left pseudoinverse satisfies

$$
X^+X = I_p \qquad \text{but in general} \qquad XX^+ \neq I_n
$$

Only the left identity holds. $XX^+$ is instead the orthogonal projector onto $X$'s column space — a $n\times n$ matrix of rank $p<n$, which cannot equal $I_n$ (rank $n$). The OLS solution $\hat\beta=(X^TX)^{-1}X^Ty=X^+y$ is the projection of $y$ onto that column space, expressed in the basis given by $X$'s columns.

### In Code

```python
"""
The pseudoinverse trap: (X^T X)^-1 X^T is the OLS normal-equations
operator (the Moore-Penrose LEFT pseudoinverse), and it is NOT
decomposable into X^-1 X^-T -- X is rectangular, so X^-1 does not
exist. Worked end-to-end on Iris: predict petal width from the other
three measurements. Cross-checked against sklearn, lstsq, and pinv.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

iris = load_iris()
X_raw: np.ndarray = iris.data[:, :3]     # sepal length, sepal width, petal length
y: np.ndarray = iris.data[:, 3]           # petal width -- the regression target
n_samples: int = X_raw.shape[0]

X: np.ndarray = np.column_stack([np.ones(n_samples), X_raw])   # add intercept, (150, 4)

# ── Normal equations: the (X^T X)^-1 X^T construction ────────────────
beta_normal_eq: np.ndarray = np.linalg.inv(X.T @ X) @ X.T @ y

# ── Cross-checks against independent implementations ─────────────────
beta_lstsq: np.ndarray = np.linalg.lstsq(X, y, rcond=None)[0]
beta_pinv: np.ndarray = np.linalg.pinv(X) @ y
beta_sklearn: np.ndarray = LinearRegression(fit_intercept=False).fit(X, y).coef_

print("beta via normal equations: ", np.round(beta_normal_eq, 4))
print("beta via lstsq:            ", np.round(beta_lstsq, 4))
print("beta via pinv:             ", np.round(beta_pinv, 4))
print("beta via sklearn:          ", np.round(beta_sklearn, 4))
print(f"Max pairwise difference:   "
      f"{np.max(np.abs(beta_normal_eq - beta_sklearn)):.2e}")

# ── X itself has no ordinary inverse -- it's 150x4, not square ───────
try:
    np.linalg.inv(X)
except np.linalg.LinAlgError as e:
    print(f"\nX^-1 does not exist: {e}")

# ── The pseudoinverse is only a LEFT inverse, not two-sided ──────────
X_pinv: np.ndarray = np.linalg.pinv(X)
left_identity: np.ndarray = X_pinv @ X        # (4, 4): should be ~ I_4
right_product: np.ndarray = X @ X_pinv         # (150, 150): should NOT be I_150

print(f"\nX^+ @ X   (4x4) equals I_4:      {np.allclose(left_identity, np.eye(4))}")
print(f"X @ X^+   (150x150) equals I_150: {np.allclose(right_product, np.eye(n_samples))}")
print(f"rank(X @ X^+) = {np.linalg.matrix_rank(right_product)}  "
      f"(<= 4, so it can't be I_150)")
```

### Common Misconceptions

- **"$(X^TX)^{-1}X^T$ can always be algebraically split into $X^{-1}X^{-T}$."** Only if $X$ is square and invertible — see the section above. For any real design matrix with more observations than features (the normal case), it can't be, and the construction exists precisely *because* $X^{-1}$ doesn't exist.
- **"The Moore–Penrose pseudoinverse is a true two-sided inverse."** No — $X^+X=I_p$ holds (left inverse), but $XX^+\neq I_n$ in general, confirmed numerically above. $XX^+$ is a projector, not an identity.

---

## Combining Both: $A^{-T}$ and the Double-Reversal Cancellation

### Intuition

Once transpose and inverse are both in play, a natural question is whether "$A^{-T}$" is ambiguous — transpose then invert, or invert then transpose? It isn't: both orders always agree for an invertible $A$. And when you apply the combined operation to a product $AB$, the transpose-reversal and inverse-reversal each flip the order once, and the two flips cancel — landing back on the *original* factor order, just with each factor turned into its own $A^{-T}$.

### Formal Definition

$A^{-T}$ is unambiguous because

$$
A^T(A^{-1})^T = (A^{-1}A)^T = I^T = I
$$

(using the transpose-reversal rule on $A^{-1}$ and $A$), which shows $(A^{-1})^T$ satisfies the defining property of $(A^T)^{-1}$. So $(A^T)^{-1}=(A^{-1})^T=:A^{-T}$, always, for any invertible $A$.

Combining the two reversal rules on a product $AB$ (both invertible):

$$
(AB)^{-T} = ((AB)^{-1})^T = (B^{-1}A^{-1})^T = (A^{-1})^T(B^{-1})^T = A^{-T}B^{-T}
$$

The inverse rule reverses $A,B \to B^{-1},A^{-1}$; the transpose rule reverses that again, back to $A^{-1},B^{-1}$ order — i.e. $A^{-T}, B^{-T}$ in the *original* $A,B$ order. This is exactly what licenses a step like $(W^TW)^{-1}=W^{-1}W^{-T}$ in the first place: that's a single inverse-reversal on the two-factor product $W^TW$, taking $(W^T,W)\to(W^{-1},W^{-T})$ — order flipped once, not twice, because only `.inv()` was applied to the whole product, not `.inv().T`.

### In Code

```python
"""
Combining inverse and transpose: A^-T := (A^-1)^T = (A^T)^-1 always
agree, and on a product the two order-reversals cancel:
(AB)^-T = A^-T @ B^-T -- the SAME order as the original A, B, not
reversed. Reuses the real covariance/correlation matrices from Iris.
The final check shows what goes wrong if you use the single-reversed
inverse product where the double-reversed -T product was needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

X: np.ndarray = load_iris().data
X_c: np.ndarray = X - X.mean(axis=0)
n_samples: int = X_c.shape[0]

A: np.ndarray = (X_c.T @ X_c) / (n_samples - 1)
stds: np.ndarray = np.sqrt(np.diag(A))
B: np.ndarray = A / np.outer(stds, stds)

# ── A^-T is unambiguous: transpose-then-invert == invert-then-transpose
A_invT_route1: np.ndarray = np.linalg.inv(A).T
A_invT_route2: np.ndarray = np.linalg.inv(A.T)
print(f"(A^-1)^T vs (A^T)^-1 max diff: "
      f"{np.max(np.abs(A_invT_route1 - A_invT_route2)):.2e}")

# ── Double reversal cancels: (AB)^-T == A^-T @ B^-T, order unchanged ─
lhs: np.ndarray = np.linalg.inv(A @ B).T
rhs: np.ndarray = np.linalg.inv(A).T @ np.linalg.inv(B).T
print(f"(AB)^-T vs A^-T @ B^-T max diff: {np.max(np.abs(lhs - rhs)):.2e}")

# ── Contrast: naively reversing order AGAIN (as if two flips don't
# cancel) gives a genuinely different, wrong matrix -- B^-T @ A^-T
# instead of the correct A^-T @ B^-T:
naive_reversed: np.ndarray = np.linalg.inv(B).T @ np.linalg.inv(A).T
print(f"(AB)^-T vs B^-T @ A^-T (naively re-reversed, WRONG) max diff: "
      f"{np.max(np.abs(lhs - naive_reversed)):.2e}  <- large: order genuinely matters")
```

Note that a lazier contrast — comparing against plain `A^-1 @ B^-1` — would misleadingly *pass* here, only because $A$ and $B$ (covariance and correlation matrices) are each symmetric, so $A^{-1}=A^{-T}$ and $B^{-1}=B^{-T}$ individually. That coincidence would hide the real point. The genuine test is whether $A^{-1}$ and $B^{-1}$ *commute* — they generally don't, so $B^{-T}A^{-T} \neq A^{-T}B^{-T}$, and that's the actual reason order matters here.

### Common Misconceptions

- **"$A^{-T}$ is ambiguous depending on whether you transpose first or invert first."** No — proven equal above for any invertible $A$; the notation is safe precisely because the two operations commute.
- **"Since inverting a product reverses order, inverting-and-transposing a product must double-reverse to something new."** It does apply two reversals, but they cancel, landing back on the original $A,B$ order (each factor replaced by its own $A^{-T}$) — not on some further-scrambled arrangement. The final print above shows the actual "wrong rule" result (order flipped again, as if the two reversals didn't cancel) is numerically distinct — using the single-reversal instinct where the double-reversal rule applies computes the wrong matrix.
- **"If $A$ and $B$ are each symmetric, order never matters for their inverses."** Individually, $A^{-1}=A^{-T}$ and $B^{-1}=B^{-T}$ hold — but $A^{-1}B^{-1}$ and $B^{-1}A^{-1}$ are still generally different matrices, because $A^{-1}$ and $B^{-1}$ don't commute just from each being symmetric. Symmetry of the factors is a distinct property from commutativity between them.
