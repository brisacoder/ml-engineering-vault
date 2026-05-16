---
type: concept
title: "NumPy Array Thinking Patterns"
libs:
  - "numpy"
tags:
  - "concept/optimization"
  - "task/data-wrangling"
related:
  - "[[Array-Oriented Programming]]"
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Vectorization Expressiveness Limits]]"
created: 2026-05-14
updated: 2026-05-14
---

# NumPy Array Thinking Patterns

> Translating an imperative algorithm into [[Array-Oriented Programming|array-oriented]] form requires a specific toolkit of mental patterns. This note collects the key techniques: shifted arrays for consecutive differences, reshape-for-grouping to operate on chunks, pairwise operations with index arrays, and unbuffered scatter with `np.add.at`. Each pattern replaces a class of Python loops with a single vectorised expression.

---

## Pattern 1: Shifted Arrays — Consecutive Differences

### Intuition

When you need to compare or combine *adjacent* elements (element `i` with element `i+1`), the array-oriented trick is to create two views of the same data, shifted by one position:

```
original: [a, b, c, d, e, f, g, h, i]
arr[:-1]:  [a, b, c, d, e, f, g, h]     ← "all but last"
arr[1:]:      [b, c, d, e, f, g, h, i]  ← "all but first"
```

Now `arr[1:] - arr[:-1]` gives you the difference between consecutive elements — no loop needed. This is the array-oriented equivalent of `for i in range(n-1): diff[i] = arr[i+1] - arr[i]`.

### In Code — Consecutive Differences

```python
import numpy as np

# Compute spaces between consecutive elements
array = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

# Array-oriented: shifted subtraction
diffs = array[1:] - array[:-1]
print(f"Differences: {diffs}")
# [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

# NumPy has a built-in for this common pattern:
diffs_builtin = np.diff(array)
print(f"np.diff:     {diffs_builtin}")

# Verify they match
assert np.allclose(diffs, diffs_builtin)
```

### In Code — Path Length via Shifted Arrays

```python
import numpy as np

# Compute the length of a parametric curve: (sin(3t), sin(4t))
t = np.linspace(0, 2 * np.pi, 10_000)
x = np.sin(3 * t)
y = np.sin(4 * t)

# Each tiny segment has length sqrt(dx² + dy²)
# dx[i] = x[i+1] - x[i], dy[i] = y[i+1] - y[i]

# ❌ Imperative:
# total = 0.0
# for i in range(len(x) - 1):
#     dx = x[i+1] - x[i]
#     dy = y[i+1] - y[i]
#     total += np.sqrt(dx**2 + dy**2)

# ✅ Array-oriented: shifted arrays + vectorised ops + reduction
path_length = np.sum(np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2))
print(f"Path length: {path_length:.4f}")

# This is numerically computing a line integral — a one-liner!
```

### In Code — Other Shifted Array Applications

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data[:, 0]  # sepal length

# 1. Moving difference (returns/changes)
returns = data[1:] / data[:-1] - 1
print(f"First 5 relative changes: {np.round(returns[:5], 4)}")

# 2. Detect where values change (useful for run-length encoding)
values = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 1])
change_points = np.where(values[1:] != values[:-1])[0] + 1
print(f"Values change at indices: {change_points}")

# 3. Check if array is sorted
arr = np.array([1, 3, 5, 7, 9])
is_sorted = np.all(arr[1:] >= arr[:-1])
print(f"Is sorted: {is_sorted}")

# 4. Second derivative (diff of diff)
signal = np.sin(np.linspace(0, 4*np.pi, 1000))
first_deriv = np.diff(signal)
second_deriv = np.diff(signal, n=2)  # np.diff does this natively
print(f"Signal shape: {signal.shape}, "
      f"1st deriv: {first_deriv.shape}, "
      f"2nd deriv: {second_deriv.shape}")
```

### Common Misconceptions

- **"Shifted slices create copies"** — No. `arr[1:]` and `arr[:-1]` are **views** (they share memory with the original). The only new memory allocation is for the result of the subtraction.
- **"`np.diff` is always the right tool"** — `np.diff` handles the simple case, but the shifted-slice pattern is more general. You can do `arr[2:] - arr[:-2]` for stride-2 differences, or `arr[1:] * arr[:-1]` for consecutive products.

---

## Pattern 2: Reshape for Grouping

### Intuition

When you need to operate on **groups of consecutive elements** (blocks of 4 bytes, 64×64 pixel tiles, batches of 32 samples), the trick is to reshape the array so that the grouping becomes a new axis. Then you can reduce (mean, sum, max) or transform along that axis.

The key insight: reshaping an array is **free** — it just changes how NumPy interprets the memory layout. No data is copied.

### In Code — Image Downscaling

```python
import numpy as np

# Downscale a 1920×2560×3 image by factor 64 using ONLY reshape + mean

# Simulate an image (in practice, use matplotlib.image.imread)
rng = np.random.default_rng(42)
image = rng.integers(0, 256, (1920, 2560, 3), dtype=np.uint8)
print(f"Original shape: {image.shape}")

# Step 1: reshape to split each spatial dimension into (n_blocks, block_size)
# (1920, 2560, 3) → (30, 64, 40, 64, 3)
factor = 64
reshaped = image.reshape(
    image.shape[0] // factor, factor,
    image.shape[1] // factor, factor,
    3
)
print(f"Reshaped: {reshaped.shape}")

# Step 2: average over the block dimensions (axes 1 and 3)
# IMPORTANT: reduce axis 3 before axis 1 (because reducing shifts later axes)
resampled = np.mean(np.mean(reshaped, axis=3), axis=1)
print(f"Resampled: {resampled.shape}")

# Step 3: convert back to uint8
resampled = resampled.astype(np.uint8)
print(f"Final shape: {resampled.shape}, dtype: {resampled.dtype}")

# Alternative: use keepdims=True to avoid axis-numbering headaches
resampled_alt = np.mean(
    np.mean(reshaped, axis=1, keepdims=True),
    axis=3, keepdims=True,
).reshape(image.shape[0] // factor, image.shape[1] // factor, 3).astype(np.uint8)

assert np.array_equal(resampled, resampled_alt)
print("Both methods match!")
```

### In Code — Byte Endianness Swap

```python
import numpy as np

# Problem: big-endian float32 data needs to be made little-endian
# Solution: reshape into groups of 4 bytes, reverse each group

# Simulate big-endian data (4 float32s = 16 bytes)
values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
big_endian_bytes = values.tobytes()  # native order
raw = np.frombuffer(big_endian_bytes, dtype=np.uint8)
print(f"Raw bytes: {raw}")

# Reshape into groups of 4, reverse each group, flatten back
swapped = raw.reshape(-1, 4)[:, ::-1].reshape(-1)
print(f"Swapped:   {swapped}")

# The pattern: reshape(-1, group_size)[:, ::-1].reshape(-1)
# Works for any byte-swap, bit-reversal, or group-level transform
```

### In Code — Batch Processing with Reshape

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# Take first 140 samples (divisible by batch_size)
X = iris.data[:140]

# Process in batches of 20 without a loop
batch_size = 20
n_samples = X.shape[0]
n_features = X.shape[1]

# Reshape: (140, 4) → (7, 20, 4)
batches = X.reshape(n_samples // batch_size, batch_size, n_features)
print(f"Batched shape: {batches.shape}")

# Compute per-batch statistics (all batches at once!)
batch_means = batches.mean(axis=1)    # shape (7, 4)
batch_stds = batches.std(axis=1)      # shape (7, 4)
batch_maxes = batches.max(axis=1)     # shape (7, 4)

print(f"Per-batch means shape: {batch_means.shape}")
print(f"Batch 0 mean: {np.round(batch_means[0], 2)}")
print(f"Batch 6 mean: {np.round(batch_means[6], 2)}")
```

### Common Misconceptions

- **"Reshape copies data"** — No. `np.reshape` returns a view when possible (it just reinterprets the same memory). The only case where it copies is when the new shape requires non-contiguous memory access that can't be expressed as strides.
- **"I need to reduce the higher axis first"** — You need to be careful about axis numbering: reducing axis 3 doesn't change axes 0, 1, 2. But reducing axis 1 shifts axis 3 → axis 2. Using `keepdims=True` or negative axis indices avoids this confusion.

---

## Pattern 3: Pairwise Operations with Index Arrays

### Intuition

Many problems require computing something for every pair of elements (distances, forces, similarities). The imperative approach is an O(n²) nested loop. The array-oriented approach uses **index arrays** to select all pairs at once, then computes everything in one vectorised pass.

`np.triu_indices(n, k=1)` generates the indices of the upper triangle (all unique pairs `(i, j)` where `i < j`), avoiding both the diagonal and double-counting.

### In Code — Pairwise Distances

```python
import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist, squareform

iris = load_iris()
X = iris.data[:50]  # first 50 samples, 4 features each

# ❌ Imperative: nested loop
# n = len(X)
# dist = np.zeros((n, n))
# for i in range(n):
#     for j in range(i+1, n):
#         d = np.sqrt(np.sum((X[i] - X[j])**2))
#         dist[i, j] = d
#         dist[j, i] = d

# ✅ Array-oriented: index arrays + broadcasting
i_idx, j_idx = np.triu_indices(len(X), k=1)
# i_idx and j_idx are arrays of all unique pairs: (0,1), (0,2), ..., (49,50)
print(f"Number of pairs: {len(i_idx)}")  # 50*49/2 = 1225

# Compute all pairwise displacements at once
pw_displacement = X[j_idx] - X[i_idx]  # shape (1225, 4)
pw_distance = np.sqrt(np.sum(pw_displacement**2, axis=1))  # shape (1225,)

# Build the full distance matrix
dist_matrix = np.zeros((len(X), len(X)))
dist_matrix[i_idx, j_idx] = pw_distance
dist_matrix[j_idx, i_idx] = pw_distance  # symmetric

# Verify against scipy
expected = squareform(pdist(X))
print(f"Max error vs scipy: {np.abs(dist_matrix - expected).max():.2e}")
```

### In Code — N-Body Gravitational Forces (Complete)

```python
import numpy as np

G = 1.0  # gravitational constant

def array_forces(m, x):
    """Compute gravitational forces between all pairs of bodies.

    m: masses, shape (n,)
    x: positions, shape (n, 3)
    returns: forces, shape (n, 3)

    Key techniques used:
    1. np.triu_indices for pair generation
    2. Broadcasting for vectorised physics
    3. np.add.at for unbuffered scatter (see Pattern 4 below)
    """
    # Generate all unique pairs (i < j)
    i, j = np.triu_indices(len(x), k=1)

    # Pairwise displacement vectors: shape (n_pairs, 3)
    pw_displacement = x[j] - x[i]

    # Pairwise distances: shape (n_pairs,)
    pw_distance = np.sqrt(np.sum(pw_displacement**2, axis=-1))

    # Unit direction vectors: shape (n_pairs, 3)
    pw_direction = pw_displacement / pw_distance[:, np.newaxis]

    # Force magnitudes × directions: F = G * m_i * m_j * dir / r²
    pw_force = (
        G * m[i, np.newaxis] * m[j, np.newaxis]
        * pw_direction / pw_distance[:, np.newaxis]**2
    )

    # Accumulate forces (Newton's third law: F_ij = -F_ji)
    total_force = np.zeros_like(x)
    np.add.at(total_force, i, pw_force)    # body i pulled toward j
    np.add.at(total_force, j, -pw_force)   # body j pulled toward i

    return total_force


# Simulate a sun + two planets
m = np.array([100.0, 1.0, 1.0])
x = np.array([[0, 0, 0], [0, 0.9, 0], [0, 1.1, 0]], dtype=np.float64)

forces = array_forces(m, x)
print("Forces on each body:")
for body_idx in range(len(m)):
    print(f"  Body {body_idx} (m={m[body_idx]:5.1f}): "
          f"F = [{forces[body_idx, 0]:10.4f}, "
          f"{forces[body_idx, 1]:10.4f}, "
          f"{forces[body_idx, 2]:10.4f}]")

# The sun feels a net force toward the planets
# Each planet feels a strong pull toward the sun
```

### Common Misconceptions

- **"This uses more memory than the loop"** — Yes. The array-oriented version creates arrays of shape `(n_pairs, 3)`, which for large `n` can be substantial (n=10,000 → ~100M pairs → ~2.4 GB). For very large problems, you need chunked approaches or [[Numba JIT Deep Dive|Numba]].
- **"np.triu_indices is the only way"** — It's the cleanest for pairwise problems, but `np.meshgrid`, `np.ix_`, and broadcasting with `X[:, np.newaxis] - X[np.newaxis, :]` also work (the broadcast version creates an n×n×d tensor, which uses more memory but avoids the scatter step).

---

## Pattern 4: Unbuffered Scatter with np.add.at

### Intuition

When multiple array-oriented computations need to be accumulated into the same output locations (like forces on body `i` from multiple partners), normal `+=` doesn't work correctly because NumPy **buffers** the assignment — if index `i` appears twice, only the last write survives.

`np.add.at(array, indices, values)` performs **unbuffered** accumulation: every write is applied, even if the same index appears multiple times.

### In Code — The Problem with Normal +=

```python
import numpy as np

# Problem: accumulate values into specific indices
result = np.zeros(3)
indices = np.array([0, 1, 0, 2, 0])  # index 0 appears 3 times
values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# ❌ Normal fancy indexing += : WRONG for repeated indices
result_wrong = np.zeros(3)
result_wrong[indices] += values
print(f"Wrong (buffered +=): {result_wrong}")
# Expected [9, 2, 4] but got [5, 2, 4]
# Only the LAST write to index 0 (value 5.0) is kept!

# ✅ np.add.at: unbuffered, all writes applied
result_correct = np.zeros(3)
np.add.at(result_correct, indices, values)
print(f"Correct (np.add.at): {result_correct}")
# [9, 2, 4] ← 1+3+5=9 for index 0, 2 for index 1, 4 for index 2

# ✅ Alternative: np.bincount for integer indices
result_bincount = np.bincount(indices, weights=values, minlength=3)
print(f"bincount:            {result_bincount}")
```

### In Code — np.add.at in the N-Body Problem

```python
import numpy as np

# In the N-body problem, each pair (i, j) produces a force.
# Body i may participate in many pairs, so we need to accumulate
# all forces on body i from all its partners.

# Simplified example: 4 bodies, 6 pairs
n = 4
i_idx, j_idx = np.triu_indices(n, k=1)
print(f"Pairs: {list(zip(i_idx, j_idx))}")
# (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

# Fake forces for each pair (1D for simplicity)
pair_forces = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)

# Accumulate: body i gets +force from each pair it's in (as first element)
#             body j gets -force from each pair it's in (as second element)
total_force = np.zeros(n)
np.add.at(total_force, i_idx, pair_forces)   # body i attracted to j
np.add.at(total_force, j_idx, -pair_forces)  # body j attracted to i (Newton's 3rd)

print(f"Total force on each body: {total_force}")
# Body 0: +10+20+30 = +60 (pulled by bodies 1,2,3)
# Body 1: -10+40+50 = +80
# Body 2: -20-40+60 = 0
# Body 3: -30-50-60 = -140
# Net force sums to zero (conservation of momentum) ✓
print(f"Sum of all forces: {total_force.sum():.1f} (should be 0)")
```

### Common Misconceptions

- **"`np.add.at` is as fast as `+=`"** — It's slower because it can't be vectorised as aggressively (it must handle repeated indices). For performance-critical code, consider `np.bincount` (for 1D integer indices) or `scipy.sparse` matrix operations.
- **"I can use `np.add.at` with any ufunc"** — Yes! `np.multiply.at`, `np.maximum.at`, `np.minimum.at` all exist. Any NumPy ufunc supports `.at()` for unbuffered operations.

---

## Pattern 5: 3D+ Array Slicing

### Intuition

NumPy arrays can have arbitrary dimensions. Slicing in higher dimensions follows the same rules as 1D, with each dimension getting its own slice separated by commas. The key insight: slicing with a **range** (e.g., `0:1`) preserves the dimension; slicing with a **scalar** (e.g., `0`) removes it.

### In Code — 3D Slicing

```python
import numpy as np

# 3D array: 2 "layers" × 3 "rows" × 5 "columns"
arr = np.arange(2 * 3 * 5).reshape(2, 3, 5)
print("Full array:")
print(arr)

# Select specific regions:

# All layers, last 2 rows, last 4 columns
print(f"\narr[:, 1:, 1:] shape: {arr[:, 1:, 1:].shape}")  # (2, 2, 4)

# First layer only, all rows, last 3 columns
# Using range 0:1 → keeps dimension (shape 1,3,3)
print(f"arr[0:1, :, 2:] shape: {arr[0:1, :, 2:].shape}")  # (1, 3, 3)

# Using scalar 0 → removes dimension (shape 3,3)
print(f"arr[0, :, 2:]   shape: {arr[0, :, 2:].shape}")     # (3, 3)

# Both give the same data, different number of dimensions.
# Use range when you need to maintain dimensionality (e.g., for broadcasting).
# Use scalar when you want to "peel off" a dimension.

# Step slicing: every other element along axis 1
print(f"\narr[:, ::2, :] shape: {arr[:, ::2, :].shape}")  # (2, 2, 5)
print(arr[:, ::2, :])
```

### In Code — Practical Example: RGB Channel Manipulation

```python
import numpy as np

# Simulate a small RGB image: 4 rows × 6 columns × 3 channels
rng = np.random.default_rng(42)
image = rng.integers(0, 256, (4, 6, 3), dtype=np.uint8)

# Extract individual channels
red   = image[:, :, 0]  # shape (4, 6) — scalar index removes channel dim
green = image[:, :, 1]
blue  = image[:, :, 2]

# Make the image grayscale (weighted average of channels)
# Standard weights: R=0.2989, G=0.5870, B=0.1140
gray = (0.2989 * image[:, :, 0]
      + 0.5870 * image[:, :, 1]
      + 0.1140 * image[:, :, 2]).astype(np.uint8)
print(f"Grayscale shape: {gray.shape}")

# Swap red and blue channels
swapped = image[:, :, ::-1]  # reverses the channel axis
print(f"Original RGB[0,0]: {image[0, 0]}")
print(f"Swapped BGR[0,0]:  {swapped[0, 0]}")

# Crop: top-left 2×3 region
crop = image[:2, :3, :]  # first 2 rows, first 3 columns, all channels
print(f"Crop shape: {crop.shape}")
```

### Common Misconceptions

- **"I need to use `np.take` or `np.index_select` for slicing"** — For contiguous ranges, regular Python slice syntax (`arr[a:b, c:d]`) is simpler and returns a view (no copy). `np.take` is useful for non-contiguous or repeated indices.
- **"Slicing always creates a copy"** — Slicing creates a **view** (shared memory). Only fancy indexing with an array of indices creates a copy.

---

For the full loop→vectorised replacement table, see [[Accelerated Python - CPU Vectorization]].
For the Pandas equivalent patterns, see [[NumPy Pandas Vectorization]].
For the paradigm that motivates these patterns, see [[Array-Oriented Programming]].
