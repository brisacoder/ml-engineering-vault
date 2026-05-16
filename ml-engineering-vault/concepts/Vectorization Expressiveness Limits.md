---
type: concept
title: "Vectorization Expressiveness Limits"
libs:
  - "numpy"
  - "pytorch"
tags:
  - "concept/optimization"
related:
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[JIT Compilation Mechanics]]"
  - "[[Numba JIT Deep Dive]]"
  - "[[JAX Tracing and Compilation]]"
created: 2026-05-14
updated: 2026-05-14
---

# Vectorization Expressiveness Limits

> Not everything can be vectorised. The [[Accelerated Python - CPU Vectorization|CPU Vectorization]] note shows patterns where NumPy replaces Python loops beautifully. This note covers the **opposite** — algorithms where the loop body depends on intermediate results, data-dependent branching prevents array operations, or the algorithm says "iterate until converged." These are the limits of array-oriented programming.

---

## Closed-Form vs Iterative — The Key Distinction

### Intuition

A function is "vectorisation-friendly" when every operation is a **closed-form expression** — a fixed sequence of math operations with no data-dependent branching. You can replace every scalar with an array and the function just works.

A function is "vectorisation-hostile" when it has a loop that **depends on the data**: "keep going until this particular element converges." Different array elements may converge after different numbers of iterations. NumPy has no way to express "stop computing element 42 but keep going for element 7,391."

The telltale sign: **"iterate until converged"** in the algorithm description.

### Formal Definition

A **closed-form** function $f(x)$ can be computed by a fixed, finite sequence of elementary operations (addition, multiplication, exponentiation, logarithms, etc.). The same sequence applies to every input — no branching.

An **iterative** function requires a loop where the number of iterations depends on the input value:

$$x_{n+1} = g(x_n) \quad \text{until} \quad |x_{n+1} - x_n| < \epsilon$$

### In Code — A "Good" Function (Vectorises Naturally)

The log-Gamma function from Numerical Recipes is a classic example. It looks like it has a loop, but the loop is over a **fixed set of coefficients** — it's really just 6 additions written compactly:

```python
import numpy as np
import scipy.special

# Fixed coefficients — this loop is just syntactic sugar for 6 lines
COEFFICIENTS = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
]


def log_of_gamma(x: np.ndarray) -> np.ndarray:
    """Log-Gamma via Lanczos approximation.

    The loop over coefficients is FIXED — it always runs 6 times,
    regardless of the input. This is a closed-form expression
    disguised as a loop.
    """
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    series = np.full_like(x, 1.000000000190015, dtype=np.float64)
    for i, coeff in enumerate(COEFFICIENTS):
        series += coeff / (x + i + 1)
    return -tmp + np.log(2.5066282746310005 * series / x)


# Works perfectly with arrays — no modification needed
xs = np.linspace(0.01, 10, 100_000)
result = log_of_gamma(xs)

# Verify against SciPy
error = np.abs(result - scipy.special.loggamma(xs)).max()
print(f"Max error vs SciPy: {error:.2e}")  # ~1e-14
```

**Why this works:** the `for` loop iterates over coefficients, not over data. It runs exactly 6 times for every input element. NumPy broadcasts each `coeff / (x + i + 1)` across the entire array simultaneously.

### In Code — A "Bad" Function (Resists Vectorisation)

The incomplete Gamma function $P(a, x) = \frac{1}{\Gamma(a)} \int_0^x t^{a-1} e^{-t} dt$ uses a series expansion where different inputs converge at different rates:

```python
import numpy as np
import scipy.special

SMALL_NUMBER = 3e-7


def incomplete_gamma_P(a: float, x: float) -> float:
    """Scalar version — iterates until THIS specific (a, x) converges.

    The loop has a data-dependent exit: it returns as soon as the
    delta term is small relative to the running sum. Different
    values of x converge after different numbers of iterations.
    """
    gln = float(scipy.special.loggamma(a))

    delta = summation = 1.0 / a

    for i in range(100):
        delta *= x / (a + i + 1)
        summation += delta
        # DATA-DEPENDENT EXIT: "has THIS input converged yet?"
        if abs(delta) < abs(summation) * SMALL_NUMBER:
            return summation * np.exp(-x + a * np.log(x) - gln)

    raise RuntimeError("did not converge")


# Works fine with scalars
xs = np.linspace(0.01, 15, 10_000)
result_scalar = np.array([incomplete_gamma_P(2.0, x) for x in xs])

# Verify
error = np.abs(result_scalar - scipy.special.gammainc(2.0, xs)).max()
print(f"Max error vs SciPy: {error:.2e}")  # ~1e-7
```

Now try passing an array:

```python
# ❌ This CRASHES
try:
    incomplete_gamma_P(2.0, xs)
except ValueError as e:
    print(f"Error: {e}")
    # "The truth value of an array with more than one element is ambiguous"
```

The error comes from `if abs(delta) < abs(summation) * SMALL_NUMBER`. When `delta` and `summation` are arrays, Python can't decide whether to enter the `if` — some elements satisfy the condition, others don't. Python's `if` needs a single True/False, not an array of booleans.

### Common Misconceptions

- **"Any loop can be vectorised if you try hard enough"** — Only loops with fixed iteration count and no data-dependent branching. If the algorithm says "iterate until converged," you're fighting the paradigm.
- **"The loop over coefficients in log_of_gamma is a problem"** — It's not. It runs exactly 6 times regardless of input. It's just shorthand for 6 lines of `series += coeff / (x + i + 1)`. The key question is: does the loop count depend on the **data**?
- **"SciPy solves this, so it's academic"** — SciPy's implementations are written in Fortran/C, not vectorised NumPy. Understanding *why* they can't be pure NumPy helps you recognise the pattern in your own code.

---

## Workaround 1: Just Keep Computing (Ignore Convergence)

### Intuition

The simplest fix: remove the convergence check entirely and always iterate the maximum number of times. Elements that converged early get "wasted" iterations (computing deltas that are essentially zero), but you avoid all branching.

This is the **standard approach in deep learning**. A PyTorch training loop doesn't check whether each weight has converged — it runs for a fixed number of epochs:

```python
for epoch in range(1000):
    optimizer.zero_grad()
    predictions = model(features)
    loss = loss_function(predictions, targets)
    loss.backward()
    optimizer.step()
```

The number of epochs is a hyperparameter, not a convergence criterion. This is philosophically the same as "keep computing" — and it works because the extra computation is cheap compared to the bookkeeping overhead of tracking convergence per-element.

### In Code — Keep Computing

```python
import numpy as np
import scipy.special


def incomplete_gamma_P_keep_going(a: float, x: np.ndarray) -> np.ndarray:
    """Always iterate 100 times. Some elements converge in 5
    iterations, some in 50 — we don't care, we do 100 for all.

    This works because once an element converges, its delta is
    essentially zero, so the extra additions don't change the sum.
    """
    gln = float(scipy.special.loggamma(a))

    delta = np.full_like(x, 1.0 / a)
    summation = delta.copy()

    for i in range(100):
        delta *= x / (a + i + 1)
        summation += delta
        # No convergence check — just keep going

    return summation * np.exp(-x + a * np.log(x) - gln)


xs = np.linspace(0.01, 15, 100_000)

# Verify correctness for different values of a
for a_val in [0.5, 2.0, 10.0]:
    result = incomplete_gamma_P_keep_going(a_val, xs)
    expected = scipy.special.gammainc(a_val, xs)
    error = np.abs(result - expected).max()
    print(f"a={a_val:4.1f}  max error: {error:.2e}")
```

### Common Misconceptions

- **"Wasted iterations = wasted accuracy"** — No. Once delta is near zero, adding it to summation is a no-op to machine precision. The result is identical.
- **"100 iterations is arbitrary"** — It is. You pick a number large enough that the slowest-converging element will have converged. If you're unsure, add an assertion: `assert np.all(np.abs(delta) < np.abs(summation) * 1e-7)` after the loop.

---

## Workaround 2: Track Convergence Per Element (Masking)

### Intuition

Instead of computing everything for 100 iterations, maintain a boolean mask `not_converged` and only update the elements that haven't converged yet. Elements that converge early are skipped in subsequent iterations.

This sounds smarter but is often **slower** than the brute-force approach because:
1. The boolean mask costs memory and computation to maintain.
2. Fancy indexing (`array[mask]`) creates copies, not views — that's extra memory allocation at every iteration.
3. As elements converge, the remaining work is on a non-contiguous subset of the array, which defeats CPU cache prefetching.

### In Code — Masked Updates

```python
import numpy as np
import scipy.special

SMALL_NUMBER = 3e-7


def incomplete_gamma_P_masked(a: float, x: np.ndarray) -> np.ndarray:
    """Track which elements have converged and skip them.

    This is the 'clever' approach but often slower due to
    fancy indexing overhead.
    """
    gln = float(scipy.special.loggamma(a))

    delta = np.full_like(x, 1.0 / a)
    summation = delta.copy()

    # Boolean mask: True where element hasn't converged yet
    not_converged = np.ones(x.shape, dtype=np.bool_)

    for i in range(100):
        # Only update elements that haven't converged
        delta[not_converged] *= x[not_converged] / (a + i + 1)
        summation[not_converged] += delta[not_converged]

        # Update the mask
        not_converged &= (
            np.abs(delta) >= np.abs(summation) * SMALL_NUMBER
        )

        # Early exit if ALL elements converged
        if not not_converged.any():
            break

    return summation * np.exp(-x + a * np.log(x) - gln)


xs = np.linspace(0.01, 15, 100_000)

for a_val in [0.5, 2.0, 10.0]:
    result = incomplete_gamma_P_masked(a_val, xs)
    expected = scipy.special.gammainc(a_val, xs)
    error = np.abs(result - expected).max()
    print(f"a={a_val:4.1f}  max error: {error:.2e}")
```

### In Code — Benchmarking Both Workarounds

```python
import numpy as np
import scipy.special
import time


def benchmark(func, *args, n_runs: int = 20, label: str = "") -> float:
    _ = func(*args)  # warmup
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func(*args)
        times.append(time.perf_counter() - start)
    median_ms = np.median(times) * 1000
    print(f"{label:<30s}  {median_ms:8.2f} ms")
    return median_ms


xs = np.linspace(0.01, 15, 100_000)

# Assuming both functions from above are defined
t_keep = benchmark(incomplete_gamma_P_keep_going, 2.0, xs,
                   label="Keep going (brute force)")
t_mask = benchmark(incomplete_gamma_P_masked, 2.0, xs,
                   label="Masked (bookkeeping)")

print(f"\nKeep-going is {t_mask/t_keep:.1f}× faster than masked")
# Typically: keep-going is 2-5× faster!
```

### Common Misconceptions

- **"Skipping converged elements must be faster — it does less work"** — The work it *avoids* (a few multiplications on near-zero values) is cheaper than the work it *adds* (boolean mask evaluation, fancy indexing copies, cache-unfriendly access patterns).
- **"This would be different on GPU"** — On GPU it's even worse. GPU kernels operate on contiguous warps of 32 threads. If 20 out of 32 elements have converged, all 32 threads still run — the 20 are just masked out. This is called **warp divergence** and it's a major GPU performance killer.

---

## The Right Tool for Iterative Algorithms

### Intuition

When you genuinely need per-element convergence tracking without the overhead of NumPy masking, the answer is [[Beyond NumPy - NumExpr Numba JAX|Numba or JAX]]:

- **Numba** `@jit`: write the scalar loop with the `if` convergence check, compile it to machine code. Each element gets its own iteration count. No temporary arrays, no masking overhead.
- **JAX** `jax.lax.while_loop`: JAX's functional equivalent of "iterate until converged," compiled by XLA.

### In Code — Numba Solves It Cleanly

```python
import numpy as np
import numba as nb
import scipy.special

SMALL_NUMBER = 3e-7


@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def incomplete_gamma_P_numba(a, x):
    """Numba version: write the natural scalar algorithm with
    data-dependent early exit. Numba compiles it and broadcasts
    over arrays automatically.

    This is the BEST of both worlds:
    - Clean code (looks like the original scalar version)
    - Fast (compiled to machine code, no Python overhead)
    - Correct (each element converges independently)
    """
    gln = np.lgamma(a)  # Numba knows math.lgamma
    delta = 1.0 / a
    summation = delta

    for i in range(100):
        delta *= x / (a + i + 1)
        summation += delta
        if abs(delta) < abs(summation) * SMALL_NUMBER:
            break

    return summation * np.exp(-x + a * np.log(x) - gln)


xs = np.linspace(0.01, 15, 100_000)

# First call compiles
result = incomplete_gamma_P_numba(2.0, xs)

# Verify
expected = scipy.special.gammainc(2.0, xs)
error = np.abs(result - expected).max()
print(f"Max error: {error:.2e}")
```

### In Code — The PyTorch Training Loop Connection

The "iterate a fixed number of times" pattern is so deeply embedded in ML that it has become invisible. Here's the full picture:

```python
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load real dataset
iris = load_iris()
X = StandardScaler().fit_transform(iris.data).astype(np.float32)
y = iris.target.astype(np.int64)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Simple classifier
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Fixed iteration count — no convergence check!
# This is philosophically identical to incomplete_gamma_P_keep_going:
# some weights "converge" early, but we keep updating them anyway.
losses = []
for epoch in range(200):
    optimizer.zero_grad()
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss at epoch 50: {losses[49]:.4f}  (probably already 'converged')")
print(f"Loss at epoch 200: {losses[-1]:.4f}  (barely changed from epoch 50)")

# The model was "done" by epoch 50, but we kept going for 200.
# The extra iterations cost compute but the code is simpler and
# it's the standard practice in ML.
```

### Common Misconceptions

- **"ML should use convergence criteria"** — Some do (early stopping, tolerance-based), but fixed epochs is the default because: (a) the loss landscape is stochastic, so convergence oscillates; (b) the compute cost of checking convergence across millions of parameters is nontrivial; (c) it works well enough in practice.
- **"This only matters for numerical algorithms"** — Any algorithm with "iterate until condition" has this problem: Newton's method, EM algorithm, power iteration, PageRank, k-means, etc.

---

## Summary — Pattern Recognition Guide

| Pattern in Algorithm | Vectorisable? | Strategy |
|---------------------|:-------------:|----------|
| Fixed formula (no loops) | ✅ Yes | Direct NumPy |
| Loop over fixed coefficients | ✅ Yes | NumPy (loop runs same count for all elements) |
| "Iterate until converged" | ❌ No | Keep-going (brute force) or Numba `@jit` |
| Data-dependent branching (`if x[i] > threshold`) | ❌ No | `np.where` for simple cases, Numba for complex |
| Recurrence (`x[i] = f(x[i-1])`) | ❌ No | Numba `@jit` or `np.ufunc.accumulate` |
| Graph traversal / tree recursion | ❌ No | Numba `@jit` or pure Python |

For the performance cost of sticking with NumPy's multi-pass model, see [[NumPy Multi-Pass Bottleneck]].
For tools that solve both problems, see [[Beyond NumPy - NumExpr Numba JAX]].
