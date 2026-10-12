---
type: concept
title: "K-Means Clustering"
libs:
  - scikit-learn
  - numpy
tags:
  - concept/stats
  - concept/other
related:
  - "[[Principal Component Analysis]]"
  - "[[Eigenvalues and Eigenvectors]]"
  - "[[Sklearn Datasets]]"
  - "[[Sklearn Feature Scaling]]"
created: 2026-09-27
updated: 2026-09-27
---

# K-Means Clustering

> K-means partitions $n$ points into $k$ groups by minimizing the within-cluster sum of squares. That single objective is also what ties it to PCA: both are variance decompositions of the same total, one discrete, one continuous. This note starts with the question of what "explained variance" even means for a partition — and why running PCA on `KMeans.transform` output does not answer it.

---

## Explained Variance for a Partition

### Intuition

`KMeans.transform(X)` does not project anything. It returns an $n \times k$ matrix of *distances* from each point to each centroid. Running PCA on that matrix tells you how the $k$ distance columns co-vary with one another — a fact about the geometry of the distance features, not about how well the partition explains the original data. It barely changes when you swap the fitted centroids for three random points, which settles whether it measures clustering quality.

K-means *does* have a perfectly good explained variance; sklearn just doesn't wrap it in an attribute. The total sum of squares around the grand mean splits exactly into a within-cluster part and a between-cluster part — the same identity as one-way ANOVA. sklearn gives you the within part as `inertia_`, so the fraction of variance a partition explains is $1 - \texttt{inertia\_}/\text{TSS}$. That is the $R^2$ of the clustering, and it is exactly what the elbow plot draws, just unnormalized.

### Formal Definition

With $\bar{x}$ the grand mean, $\mu_j$ the centroid of cluster $C_j$ and $n_j = |C_j|$:

$$
\underbrace{\sum_{i=1}^n \|x_i-\bar{x}\|^2}_{\text{TSS}}
\;=\;
\underbrace{\sum_{j=1}^k\sum_{i\in C_j}\|x_i-\mu_j\|^2}_{\text{WSS}\;=\;\texttt{inertia\_}}
\;+\;
\underbrace{\sum_{j=1}^k n_j\|\mu_j-\bar{x}\|^2}_{\text{BSS}}
$$

$$
R^2_{\text{k-means}} = \frac{\text{BSS}}{\text{TSS}} = 1 - \frac{\texttt{inertia\_}}{\text{TSS}}
$$

Note $\text{TSS} = (n-1)\operatorname{tr}(\Sigma) = (n-1)\sum_i\lambda_i$ — the same total that PCA's `explained_variance_ratio_` divides by. Both methods are splitting one pie.

### In Code — Why PCA on `transform` Output Is Not It

```python
"""
Why "PCA on KMeans.transform(X)" does not give k-means an explained variance.
KMeans.transform returns DISTANCES from each sample to each of the k
centroids -- a (n, k) matrix of derived features. PCA on it reports how
those k distance columns co-vary with each other, which says nothing about
how well the partition explains the original data. Iris, k = 3.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
n, p = X.shape

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
D = km.transform(X)  # (150, 3): distance of every flower to each centroid
print(f"KMeans.transform(X) shape: {D.shape}  -- distances, not a projection")
print(f"first flower's distances to the 3 centroids: {np.round(D[0], 3)}")

# PCA on the distance matrix: a statement about the 3 distance columns only.
pca_on_D = PCA().fit(D)
print(f"\nPCA on distances, cumulative explained variance: "
      f"{np.round(np.cumsum(pca_on_D.explained_variance_ratio_), 4)}")

# The distance columns are strongly (anti)correlated with each other by
# construction: all three are smooth functions of WHERE the flower sits along
# Iris's dominant axis, so PC1 mostly re-encodes that position. This happens
# no matter how good the centroids are.
print(f"correlation between distance columns:\n{np.round(np.corrcoef(D.T), 3)}")

# Same PCA on a deliberately terrible partition -> nearly the same curve.
rng = np.random.default_rng(0)
bad_centers = X[rng.choice(n, 3, replace=False)]
D_bad = np.sqrt(((X[:, None, :] - bad_centers[None]) ** 2).sum(axis=-1))
pca_on_D_bad = PCA().fit(D_bad)
print(f"\nsame curve for 3 RANDOM points as 'centroids': "
      f"{np.round(np.cumsum(pca_on_D_bad.explained_variance_ratio_), 4)}")
print("-> the number does not react to clustering quality; it is not a k-means metric.")
```

### In Code — The Identity, and the Elbow Plot as $R^2$

```python
"""
k-means DOES have an explained variance -- it just isn't an attribute.
Total sum of squares splits exactly into within-cluster + between-cluster
(the ANOVA / Huygens identity), and sklearn hands you the within part as
inertia_. So:
    explained variance of a k-means partition = 1 - inertia_ / TSS
This is the R^2 of the clustering and exactly what the elbow plot shows.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X = load_iris().data
n, p = X.shape

grand_mean = X.mean(axis=0)
tss = np.sum((X - grand_mean) ** 2)  # total SS around the grand mean

km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)

# within-cluster SS: squared distance of every point to ITS centroid
wss = np.sum((X - km.cluster_centers_[km.labels_]) ** 2)
# between-cluster SS: each centroid's squared distance to the grand mean,
# weighted by how many points it owns
sizes = np.bincount(km.labels_, minlength=3)
bss = np.sum(sizes * np.sum((km.cluster_centers_ - grand_mean) ** 2, axis=1))

print(f"TSS              = {tss:.4f}")
print(f"WSS (by hand)    = {wss:.4f}")
print(f"km.inertia_      = {km.inertia_:.4f}   <- sklearn's within-cluster SS")
print(f"BSS              = {bss:.4f}")
print(f"WSS + BSS == TSS : {np.isclose(wss + bss, tss)}")
print(f"\nexplained variance (R^2) of k=3 partition = BSS/TSS = 1 - inertia/TSS = "
      f"{1 - km.inertia_ / tss:.4f}")

# The elbow curve is this same quantity, one k at a time.
print(f"\n{'k':>2}  {'inertia':>9}  {'explained':>9}")
for k in range(1, 8):
    inertia = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_
    print(f"{k:>2}  {inertia:>9.3f}  {1 - inertia / tss:>9.4f}")
```

### Common Misconceptions

- **"sklearn can't give me k-means explained variance."** It can, in one line: `1 - km.inertia_ / ((X - X.mean(0))**2).sum()`. `inertia_` *is* the within-cluster sum of squares; the total is trivial to compute.
- **"`KMeans.transform` is a dimensionality reduction like `PCA.transform`."** It's a feature map to centroid distances. Useful as an input to a downstream model, useless as a lens on clustering quality.
- **"Explained variance goes up with $k$, so more clusters is better."** It goes up monotonically all the way to $R^2=1$ at $k=n$ — same trap as `explained_variance_ratio_` with more components. Use the elbow, silhouette, or a held-out criterion, not the raw number.

---

## How K-Means and PCA Explained Variance Relate

### Intuition

Ding & He (2004) showed that if you relax k-means' hard cluster assignments to continuous values, the relaxed problem is solved exactly by the top $k-1$ principal components. The discrete optimum can never beat the continuous one, so PCA's cumulative explained variance for $k-1$ components is an **upper bound** on what any $k$-cluster partition can explain. On Iris the gap is large: 3 clusters explain 88% while 2 components keep 98%, because the flowers don't sit in three tight balls along those axes — the continuous projection keeps the spread inside each species that a partition must write off as within-cluster error.

The pipeline that actually makes sense is the reverse of the one in the question: PCA first (reduce $p$), then k-means on the scores. Then two separate numbers exist and neither pretends to be the other — PCA's ratio says how much of the original variance the reduced space kept, and $1-\texttt{inertia\_}/\text{TSS}$ *in the reduced space* says how much of that the partition explains.

### Formal Definition

For $k$ clusters with eigenvalues $\lambda_1\ge\cdots\ge\lambda_p$ of the covariance matrix:

$$
\text{BSS}_k \;\le\; (n-1)\sum_{i=1}^{k-1}\lambda_i
\qquad\Longleftrightarrow\qquad
R^2_{\text{k-means}}(k) \;\le\; \sum_{i=1}^{k-1}\frac{\lambda_i}{\sum_j\lambda_j}
$$

The right-hand side is `np.cumsum(pca.explained_variance_ratio_)[k-2]`.

### In Code

```python
"""
Where PCA's cumulative explained variance and k-means' explained variance
actually meet (Ding & He, 2004): the relaxed k-means problem is solved by
the top k-1 principal components, so
    k-means explained variance with k clusters  <=  PCA cumulative variance of k-1 components.
PCA is the best CONTINUOUS variance split, k-means the best DISCRETE one;
the discrete optimum can never beat the continuous one. Verified on Iris.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris().data
tss = np.sum((X - X.mean(axis=0)) ** 2)

pca = PCA().fit(X)
cum_pca = np.cumsum(pca.explained_variance_ratio_)

print(f"{'k':>2}  {'k-means R^2':>11}  {'PCA cum. (k-1 comps)':>21}  bound holds")
for k in range(2, 6):
    r2 = 1 - KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ / tss
    bound = cum_pca[k - 2]  # k-1 components -> index k-2
    print(f"{k:>2}  {r2:>11.4f}  {bound:>21.4f}  {r2 <= bound + 1e-12}")

# The sensible pipeline is the other way round: PCA FIRST to reduce, then
# k-means on the scores. Then "explained variance" refers to the PCA step,
# and inertia is measured in the reduced space -- two separate numbers.
Z = PCA(n_components=2).fit_transform(X)
km_2d = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
tss_2d = np.sum((Z - Z.mean(axis=0)) ** 2)
print(f"\nPCA(2) keeps {cum_pca[1]:.4f} of the variance; "
      f"k-means(3) on those scores explains {1 - km_2d.inertia_ / tss_2d:.4f} of THAT.")
```

### Common Misconceptions

- **"k-means explained variance should match PCA's cumulative variance."** It never can exceed it and usually falls well short; equality would need the data to sit in exactly $k$ points along the leading $k-1$ axes. The bound is the relationship, not equality.
- **"K-means after PCA changes the answer to 'how much variance is explained'."** It changes what the question is about. Report both numbers with their denominators: PCA's against the original TSS, k-means' against the TSS of the scores it was run on.
- **"Standardization doesn't matter for k-means."** It matters exactly as much as for PCA ([[Sklearn Feature Scaling]]) and for the same reason: both minimize squared Euclidean distances, so the feature with the largest raw variance dominates the objective. On raw Iris that is petal length, same as in [[Principal Component Analysis]].
