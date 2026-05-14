---
type: concept
title: "NumPy Multi-Pass Bottleneck"
libs:
  - "numpy"
tags:
  - "concept/optimization"
related:
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[GPU Library Decision Guide]]"
created: 2026-05-14
updated: 2026-05-14
---

# NumPy Multi-Pass Bottleneck

> NumPy is fast compared to Python loops, but it leaves a lot of performance on the table. The root cause is that NumPy evaluates one operation at a time across the **entire** array, creating temporary arrays at every step. When those arrays are larger than your CPU caches, the bottleneck shifts from computation to **memory bandwidth** — you spend more time waiting for RAM than doing math.

---

## The Core Problem: Operation-at-a-Time Execution

### Intuition

Imagine you have three arrays `a`, `b`, `c`, each with 5 million floats (40 MB each). You want to compute the quadratic formula:

$$x = \frac{-b + \sqrt{b^2 - 4ac}}{2a}$$

When you write `(-b + np.sqrt(b**2 - 4*a*c)) / (2*a)`, Python sees this as a chain of separate operations. NumPy has no way to "see" the whole expression — it just gets called one operation at a time by the Python interpreter. So what actually happens is:

1. Compute `b**2` → write 40 MB temp array to RAM
2. Compute `4*a` → write another 40 MB temp array to RAM
3. Compute `(4*a) * c` → read two temps, write 40 MB result to RAM
4. Compute `b**2 - (4*a*c)` → read two temps, write 40 MB result to RAM
5. Compute `np.sqrt(...)` → read and write 40 MB
6. Compute `-b` → read and write 40 MB
7. Compute `-b + sqrt(...)` → read two, write 40 MB
8. Compute `2*a` → read and write 40 MB
9. Compute `... / (2*a)` → read two, write 40 MB

That's **9 full passes** over 40 MB each. You're reading/writing ~360 MB of data through the CPU-to-RAM bus just to compute a simple formula.

A compiled solution (NumExpr, Numba, JAX) does it in **1 pass**: for each index `i`, compute the entire formula on `a[i], b[i], c[i]` before moving to `i+1`. Those three values fit in CPU registers — no RAM round-trips at all.

### Formal Definition

The problem is the gap between **arithmetic intensity** (FLOPs per byte transferred) and **memory bandwidth**:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs performed}}{\text{Bytes moved to/from RAM}}$$

NumPy's multi-pass approach has low arithmetic intensity because each operation reads/writes the full array. Single-pass approaches have high arithmetic intensity because data stays in registers or L1 cache.

### In Code — Seeing the Temporaries

This is essentially what NumPy does internally when you write `quadratic_formula(a, b, c)`:

```python
import numpy as np

# Generate 5 million random coefficients
rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)


def quadratic_formula(a, b, c):
    """What you write — clean and readable."""
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def pedantic_quadratic_formula(a, b, c):
    """What NumPy actually does — one temp array per operation."""
    tmp1 = np.negative(b)          # -b              → 40 MB written
    tmp2 = np.square(b)            # b**2            → 40 MB written
    tmp3 = np.multiply(4, a)       # 4*a             → 40 MB written
    tmp4 = np.multiply(tmp3, c)    # (4*a)*c         → 40 MB written
    del tmp3
    tmp5 = np.subtract(tmp2, tmp4) # b**2 - 4*a*c    → 40 MB written
    del tmp2, tmp4
    tmp6 = np.sqrt(tmp5)           # sqrt(...)       → 40 MB written
    del tmp5
    tmp7 = np.add(tmp1, tmp6)      # -b + sqrt(...)  → 40 MB written
    del tmp1, tmp6
    tmp8 = np.multiply(2, a)       # 2*a             → 40 MB written
    return np.divide(tmp7, tmp8)   # final divide    → 40 MB written


# Both produce the same result
result_clean = quadratic_formula(a, b, c)
result_pedantic = pedantic_quadratic_formula(a, b, c)

print(f"Max difference: {np.abs(result_clean - result_pedantic).max()}")
# Max difference: 0.0
```

### Common Misconceptions

- **"NumPy is slow"** — No. NumPy is ~50× faster than Python loops. But it's ~3–10× slower than compiled single-pass approaches. The bottleneck isn't computation speed, it's memory bandwidth.
- **"This only matters for huge arrays"** — If your arrays fit in L1/L2 cache (a few hundred KB to a few MB), the multi-pass overhead is small. It kicks in hard when arrays exceed L3 cache (typically 8–32 MB).
- **"NumPy fuses operations automatically"** — NumPy can sometimes avoid one temporary (`a + b + c` may reuse a buffer), but it cannot fuse arbitrary expressions. It has no expression-level optimizer.

---

## CPU Cache Hierarchy — Why Array Size Matters

### Intuition

Your CPU has a hierarchy of memory, each level larger but slower:

| Level | Typical Size | Latency | Bandwidth |
|-------|-------------|---------|-----------|
| L1 cache | 32–64 KB per core | ~1 ns | ~1 TB/s |
| L2 cache | 256 KB–1 MB per core | ~4 ns | ~500 GB/s |
| L3 cache | 8–32 MB shared | ~10 ns | ~200 GB/s |
| RAM (DDR4/5) | 16–256 GB | ~50–100 ns | ~40–80 GB/s |

When NumPy processes an array of 5 million float64s (40 MB), it blows through L1, L2, and L3 caches. Every intermediate result has to be written all the way back to RAM and then read back for the next operation. The CPU is mostly idle, waiting for data.

### In Code — Measuring the Cache Effect

```python
import numpy as np
import time


def benchmark_quadratic(n: int, repeats: int = 10) -> float:
    """Benchmark the quadratic formula at different array sizes."""
    rng = np.random.default_rng(42)
    a = rng.uniform(5, 10, n)
    b = rng.uniform(10, 20, n)
    c = rng.uniform(-0.1, 0.1, n)

    # Warmup
    _ = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


# Test at different array sizes to see the cache boundary effect
sizes = [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000]
for n in sizes:
    array_mb = n * 8 * 3 / 1e6  # 3 arrays of float64
    t = benchmark_quadratic(n)
    throughput = n / t / 1e6  # millions of elements per second
    print(f"n={n:>10,}  arrays={array_mb:6.1f} MB  time={t*1e3:8.2f} ms  "
          f"throughput={throughput:6.1f} M elem/s")

# You'll typically see throughput DROP when arrays exceed L3 cache
```

### Common Misconceptions

- **"More RAM bandwidth = problem solved"** — DDR5 helps, but the fundamental issue is that NumPy forces multiple passes. Even with infinite bandwidth, single-pass is better because it also benefits from instruction-level parallelism and register reuse.
- **"GPU solves this"** — GPUs have higher memory bandwidth (~1 TB/s), but the multi-pass problem still exists. Libraries like CuPy have the same limitation as NumPy — they just hit it at a higher threshold. See [[Beyond NumPy - NumExpr Numba JAX]] for real solutions.

---

## Quantifying the Gap — Benchmark Ladder

### Intuition

The notebook benchmarks five approaches, from slowest to fastest. The performance ladder looks like this:

| Approach | Time (5M elements) | Relative Speed |
|----------|:------------------:|:--------------:|
| Pure Python loop | ~2400 ms | 1× |
| NumPy (multi-pass) | ~30 ms | ~80× |
| NumExpr (virtual machine) | ~18 ms | ~130× |
| JAX (XLA compiler) | ~23 ms | ~100× |
| Numba (LLVM compiler) | ~4 ms | ~600× |

The exact numbers depend on hardware, but the pattern is consistent: **NumPy is orders of magnitude faster than Python, but compiled single-pass solutions are several times faster than NumPy.**

### In Code — Full Benchmark

```python
import numpy as np
import time

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)


def quadratic_formula(a, b, c):
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


def benchmark(func, *args, n_runs: int = 5, label: str = "") -> float:
    """Simple benchmark returning median time in ms."""
    # Warmup
    _ = func(*args)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_ms = np.median(times) * 1000
    print(f"{label:<25s} {median_ms:8.2f} ms")
    return median_ms


# 1. Pure Python loop (only test on smaller array for sanity)
a_small, b_small, c_small = a[:50_000], b[:50_000], c[:50_000]

def python_loop(a, b, c):
    result = np.empty_like(c)
    for i in range(len(a)):
        result[i] = (-b[i] + np.sqrt(b[i]**2 - 4*a[i]*c[i])) / (2*a[i])
    return result

t_loop = benchmark(python_loop, a_small, b_small, c_small, n_runs=2,
                   label="Pure Python (50K)")

# 2. NumPy vectorised
benchmark(quadratic_formula, a, b, c, label="NumPy (5M)")

# 3. NumExpr
import numexpr as ne

def numexpr_quadratic(a, b, c):
    return ne.evaluate("(-b + sqrt(b**2 - 4*a*c)) / (2*a)")

benchmark(numexpr_quadratic, a, b, c, label="NumExpr (5M)")
```

### Common Misconceptions

- **"The Python loop is slow because of function call overhead"** — Partly, but the main cost is that each scalar `np.sqrt(...)` call has Python-to-C bridge overhead. Using `math.sqrt` instead of `np.sqrt` inside the loop would be ~2× faster (still 100× slower than vectorised).
- **"NumPy and NumExpr give the same result"** — They do to machine precision, but the order of floating-point operations may differ slightly. Don't expect bit-identical results.

---

For single-pass alternatives that solve this bottleneck, see [[Beyond NumPy - NumExpr Numba JAX]].
For patterns that vectorise well in NumPy, see [[Accelerated Python - CPU Vectorization]].
For GPU alternatives, see [[GPU Library Decision Guide]].
