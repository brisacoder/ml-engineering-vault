---
type: concept
title: "Numba JIT Deep Dive"
libs:
  - "numpy"
tags:
  - "concept/optimization"
related:
  - "[[JIT Compilation Mechanics]]"
  - "[[JAX Tracing and Compilation]]"
  - "[[JIT Pitfalls - Impure Functions and Side Effects]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[Accelerated Python - GPU Acceleration]]"
created: 2026-05-14
updated: 2026-05-14
---

# Numba JIT Deep Dive

> Numba compiles Python functions to machine code using LLVM, the same compiler backend that powers Clang/C++. It forces an **imperative** style: you write explicit scalar loops with index arithmetic, and Numba turns them into native code. This note covers the nuts and bolts — decorators, parallelism, type restrictions, and GPU compilation. For the conceptual overview, see [[JIT Compilation Mechanics]]. For basic usage examples, see [[Beyond NumPy - NumExpr Numba JAX]].

---

## @nb.njit — The Core Decorator

### Intuition

`@nb.njit` (short for `@nb.jit(nopython=True)`) tells Numba: "compile this function to machine code with **zero** Python interpreter involvement." If Numba can't compile something (e.g., you used a `dict`), it raises an error immediately rather than silently falling back to slow Python.

The older `@nb.jit` (without `nopython=True`) would silently fall back to Python interpretation for unsupported operations, hiding performance bugs. Always use `@nb.njit` — if it doesn't compile, you want to know.

### In Code — Basic @nb.njit

```python
import numpy as np
import numba as nb


@nb.njit
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Manual dot product — Numba compiles this to SIMD instructions."""
    total = 0.0
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


rng = np.random.default_rng(42)
x = rng.standard_normal(1_000_000)
y = rng.standard_normal(1_000_000)

# First call triggers compilation (~200-500ms)
result = dot_product(x, y)

# Verify against NumPy
expected = np.dot(x, y)
print(f"Numba:  {result:.6f}")
print(f"NumPy:  {expected:.6f}")
print(f"Match:  {abs(result - expected) < 1e-6}")
```

### In Code — @nb.njit vs @nb.jit (Why nopython Matters)

```python
import numpy as np
import numba as nb


# ✅ @nb.njit: fails loudly if it can't compile
@nb.njit
def good_function(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i] ** 2
    return total


# ❌ Old @nb.jit without nopython=True: silently falls back to Python
# @nb.jit  # DON'T DO THIS — hides performance problems
# def sneaky_slow(arr):
#     result = {}  # dict isn't supported in nopython mode
#     for i in range(arr.shape[0]):
#         result[i] = arr[i] ** 2
#     return result
# This would "work" but run at Python speed, defeating the purpose.

arr = np.arange(1_000_000, dtype=np.float64)
print(f"Sum of squares: {good_function(arr):.0f}")
```

### Common Misconceptions

- **"@nb.jit and @nb.njit are the same"** — No. `@nb.jit` allows fallback to Python object mode (slow). `@nb.njit` enforces nopython mode (fast). Always use `@nb.njit`.
- **"Numba compiles at decoration time"** — No. `@nb.njit` just wraps the function. Compilation happens at the first *call*, when Numba knows the input types.
- **"I need to add type annotations for Numba"** — Optional. Numba infers types from the actual arguments. You *can* provide signatures like `@nb.njit(nb.float64(nb.float64[:], nb.float64[:]))` to force specific types and compile eagerly, but it's rarely needed.

---

## parallel=True and nb.prange — Multi-Threaded Numba

### Intuition

By default, `@nb.njit` generates single-threaded code. Adding `parallel=True` tells Numba to look for opportunities to parallelise, and `nb.prange` (parallel range) explicitly marks a loop as safe to distribute across CPU cores.

The key requirement: each iteration must be **independent**. If iteration `i` depends on the result of iteration `i-1`, you can't parallelise it.

### In Code — Single-Threaded vs Parallel

```python
import numpy as np
import numba as nb
import time


@nb.njit
def quadratic_serial(a, b, c):
    """Single-threaded: one core does all the work."""
    n = a.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = (-b[i] + np.sqrt(b[i]**2 - 4*a[i]*c[i])) / (2*a[i])
    return out


@nb.njit(parallel=True)
def quadratic_parallel(a, b, c):
    """Multi-threaded: work is split across all CPU cores.
    Only change: range → nb.prange
    """
    n = a.shape[0]
    out = np.empty(n)
    for i in nb.prange(n):  # ← the only difference!
        out[i] = (-b[i] + np.sqrt(b[i]**2 - 4*a[i]*c[i])) / (2*a[i])
    return out


rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)

# Warmup (triggers compilation)
_ = quadratic_serial(a, b, c)
_ = quadratic_parallel(a, b, c)

# Benchmark
for label, func in [("Serial", quadratic_serial),
                     ("Parallel", quadratic_parallel)]:
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = func(a, b, c)
        times.append(time.perf_counter() - start)
    print(f"{label:10s}: {np.median(times)*1000:.2f} ms")
```

### In Code — When nb.prange is NOT Safe

```python
import numpy as np
import numba as nb


# ❌ UNSAFE: each iteration depends on the previous one (prefix sum)
@nb.njit(parallel=True)
def broken_prefix_sum(arr):
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in nb.prange(1, len(arr)):  # ← WRONG! i depends on i-1
        out[i] = out[i-1] + arr[i]
    return out

# This will either give wrong results or crash.
# The correct version uses a serial loop:

@nb.njit
def correct_prefix_sum(arr):
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):  # ← serial range, correct
        out[i] = out[i-1] + arr[i]
    return out


arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Correct prefix sum: {correct_prefix_sum(arr)}")
print(f"NumPy equivalent:   {np.cumsum(arr)}")
```

### In Code — Parallel Reduction Pattern

```python
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def parallel_variance(arr):
    """Compute variance using parallel reduction.
    Each thread accumulates a local sum, then they're combined.

    Numba automatically handles the reduction for +=, *=, etc.
    inside prange loops.
    """
    n = arr.shape[0]
    mean = 0.0
    for i in nb.prange(n):
        mean += arr[i]
    mean /= n

    var = 0.0
    for i in nb.prange(n):
        diff = arr[i] - mean
        var += diff * diff
    var /= n

    return mean, var


rng = np.random.default_rng(42)
data = rng.standard_normal(10_000_000)

mean, var = parallel_variance(data)
print(f"Numba mean={mean:.6f}, var={var:.6f}")
print(f"NumPy mean={data.mean():.6f}, var={data.var():.6f}")
```

### Common Misconceptions

- **"parallel=True automatically parallelises everything"** — It only parallelises loops marked with `nb.prange`. Regular `range` loops inside the same function stay serial.
- **"More threads = always faster"** — For small arrays, the overhead of thread creation and synchronisation can make parallel code *slower* than serial. Parallel shines for arrays > 100K elements.
- **"nb.prange is just like Python's multiprocessing"** — No. `nb.prange` uses threads (shared memory), not processes. There's no GIL issue because Numba code runs without the Python interpreter.

---

## What Numba Can and Cannot Compile

### Intuition

Numba in `nopython` mode only understands a specific set of types and operations. Think of it as: **Numba knows NumPy, scalars, tuples, and basic math — nothing else.**

### In Code — What Works

```python
import numpy as np
import numba as nb


# ✅ Scalar arithmetic
@nb.njit
def scalar_math(x):
    return np.sin(x) ** 2 + np.cos(x) ** 2  # always 1.0


# ✅ NumPy array creation and manipulation
@nb.njit
def array_ops(n):
    arr = np.zeros(n)
    for i in range(n):
        arr[i] = i * 2.5
    return arr


# ✅ Tuples (fixed size, heterogeneous)
@nb.njit
def return_tuple(arr):
    return arr.mean(), arr.std(), arr.min(), arr.max()


# ✅ Nested functions
@nb.njit
def helper(x):
    return x ** 2 + 1

@nb.njit
def uses_helper(arr):
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = helper(arr[i])
    return out


# ✅ 2D array indexing
@nb.njit
def matrix_trace(mat):
    total = 0.0
    for i in range(min(mat.shape[0], mat.shape[1])):
        total += mat[i, i]
    return total


# Test them all
print(f"scalar_math(3.14): {scalar_math(3.14):.10f}")
print(f"array_ops(5): {array_ops(5)}")
print(f"return_tuple: {return_tuple(np.arange(10.0))}")
print(f"uses_helper: {uses_helper(np.array([1.0, 2.0, 3.0]))}")
print(f"matrix_trace: {matrix_trace(np.eye(5))}")
```

### In Code — What Breaks

```python
import numpy as np
import numba as nb


# ❌ Dictionaries — not a known type
# @nb.njit
# def sum_dict_values(d):
#     out = 0.0
#     for v in d.values():
#         out += v
#     return out
# sum_dict_values({"a": 1.0, "b": 2.0})
# → TypingError: Failed in nopython mode

# ❌ Lists of mixed types
# @nb.njit
# def mixed_list():
#     return [1, "hello", 3.14]
# → TypingError

# ❌ Classes / custom objects
# @nb.njit
# def use_class():
#     from collections import Counter
#     return Counter([1, 2, 2, 3])
# → TypingError

# ❌ Pandas DataFrames
# @nb.njit
# def use_pandas(df):
#     return df["column"].sum()
# → TypingError

# ❌ String formatting
# @nb.njit
# def format_string(x):
#     return f"value is {x}"
# → TypingError

# ❌ try/except
# @nb.njit
# def handle_error(x):
#     try:
#         return 1.0 / x
#     except ZeroDivisionError:
#         return 0.0
# → TypingError

# WORKAROUND: Keep non-compilable code OUTSIDE the @nb.njit function.
# Use Numba for the hot inner loop, regular Python for everything else.

@nb.njit
def compute_kernel(arr):
    """Just the math — Numba-friendly."""
    total = 0.0
    for i in range(arr.shape[0]):
        total += np.sqrt(arr[i])
    return total

def full_pipeline(data_dict: dict) -> dict:
    """Outer function: regular Python handles dicts, strings, etc."""
    results = {}
    for key, arr in data_dict.items():
        results[key] = compute_kernel(arr)
    return results

data = {
    "sensor_a": np.random.rand(100_000),
    "sensor_b": np.random.rand(200_000),
}
print(full_pipeline(data))
```

### Common Misconceptions

- **"Numba supports all of NumPy"** — Almost, but not quite. Most `np.*` functions work, but some (like `np.einsum`, `np.linalg.svd`, some fancy indexing patterns) are not yet supported. Check the [Numba docs](https://numba.readthedocs.io/en/stable/reference/numpysupported.html).
- **"I should rewrite my whole program in Numba"** — No. Use Numba *surgically* on the hot loop. Keep data loading, I/O, and orchestration in regular Python.

---

## numba.cuda — GPU Programming

### Intuition

Numba can also compile functions for NVIDIA GPUs via `numba.cuda`. The programming model closely mirrors CUDA C: you define a **kernel** that runs on thousands of GPU threads simultaneously, and each thread computes one element (or a small tile) of the output.

This is a fundamentally different model from NumPy or even `@nb.njit`:
- You manually manage data transfer between CPU and GPU.
- You think in terms of a **grid** of thread blocks.
- Each thread knows its position via `nb.cuda.grid()`.

### In Code — CUDA Kernel Structure

```python
# NOTE: This code requires an NVIDIA GPU. It won't run on CPU-only machines.
# Shown here for reference — understand the pattern even if you can't run it.

# import numba as nb
# import numba.cuda
# import numpy as np
#
#
# @nb.cuda.jit
# def vector_add_gpu(a, b, out):
#     """Each GPU thread computes ONE element."""
#     i = nb.cuda.grid(1)  # get this thread's global index
#     if i < out.shape[0]:  # bounds check (grid may be larger than array)
#         out[i] = a[i] + b[i]
#
#
# # Allocate arrays
# n = 1_000_000
# a = np.random.rand(n).astype(np.float32)
# b = np.random.rand(n).astype(np.float32)
# out = np.empty_like(a)
#
# # Transfer data to GPU
# d_a = nb.cuda.to_device(a)
# d_b = nb.cuda.to_device(b)
# d_out = nb.cuda.device_array_like(out)
#
# # Launch kernel: 256 threads per block, enough blocks to cover n elements
# threads_per_block = 256
# blocks = (n + threads_per_block - 1) // threads_per_block
# vector_add_gpu[blocks, threads_per_block](d_a, d_b, d_out)
#
# # Transfer result back to CPU
# result = d_out.copy_to_host()
# print(f"Max error: {np.abs(result - (a + b)).max():.2e}")
```

### In Code — Matrix Multiplication Kernel

```python
# CUDA matrix multiplication — the "hello world" of GPU programming
# Requires NVIDIA GPU.

# import numba as nb
# import numba.cuda
# import numpy as np
#
#
# @nb.cuda.jit
# def matmul_gpu(A, B, C):
#     """Matrix multiplication C = A @ B.
#     Each thread computes ONE element of the output matrix.
#     """
#     i, j = nb.cuda.grid(2)  # 2D grid: (row, col)
#     if i < C.shape[0] and j < C.shape[1]:
#         tmp = 0.0
#         for k in range(A.shape[1]):
#             tmp += A[i, k] * B[k, j]
#         C[i, j] = tmp
#
#
# # Setup
# n = 512
# A = np.random.rand(n, n).astype(np.float32)
# B = np.random.rand(n, n).astype(np.float32)
# C = np.zeros((n, n), dtype=np.float32)
#
# # Launch 2D grid
# threads = (16, 16)  # 16×16 = 256 threads per block
# blocks = ((n + 15) // 16, (n + 15) // 16)
# matmul_gpu[blocks, threads](A, B, C)
#
# print(f"Max error vs NumPy: {np.abs(C - A @ B).max():.2e}")
```

### Common Misconceptions

- **"numba.cuda is the only way to use GPUs from Python"** — CuPy gives you NumPy-like GPU arrays. JAX gives you transparent GPU execution. PyTorch and TensorFlow handle GPUs internally. `numba.cuda` is for when you need to write *custom* GPU kernels — which is rare in ML.
- **"GPU is always faster"** — For small arrays, the CPU-to-GPU data transfer time dominates. GPUs only shine for large, parallelisable workloads (millions of elements, matrix operations).

---

## Numba vs JAX for GPU — Different Worlds

### Intuition

| Aspect | Numba CUDA | JAX on GPU |
|--------|-----------|------------|
| Programming model | Imperative (write CUDA kernels) | Array-oriented (write NumPy code) |
| Data transfer | Manual (`to_device`, `copy_to_host`) | Automatic (`jax.device_put`) |
| Code portability | GPU-specific code, won't run on CPU | Same code runs on CPU *and* GPU |
| Custom kernels | Full control over thread layout | Not directly (use `jax.custom_vjp` or Pallas) |
| Use case | Custom GPU algorithms, max performance | ML training, general array computation |

**JAX's killer advantage for GPU:** you write `@jax.jit` once and it runs on CPU, GPU, or TPU — zero code changes. Numba requires you to rewrite your code entirely for GPU.

### In Code — JAX GPU Portability

```python
# JAX: same code, different device
# (Shown for reference — requires GPU-enabled JAX)

# import jax
# import jax.numpy as jnp
#
# @jax.jit
# def quadratic_formula_jax(a, b, c):
#     return (-b + jnp.sqrt(b**2 - 4*a*c)) / (2*a)
#
# # Check what devices are available
# print("Available devices:", jax.devices())
#
# # Move data to GPU (if available)
# a_gpu = jax.device_put(a, jax.devices()[0])
# b_gpu = jax.device_put(b, jax.devices()[0])
# c_gpu = jax.device_put(c, jax.devices()[0])
#
# # Same function, now running on GPU!
# result = quadratic_formula_jax(a_gpu, b_gpu, c_gpu)
# print(f"Result lives on: {result.device}")
```

### Common Misconceptions

- **"I need numba.cuda to use my GPU for ML"** — Almost certainly not. PyTorch, TensorFlow, and JAX handle GPU execution automatically. `numba.cuda` is for writing custom low-level GPU code outside of ML frameworks.

---

For JAX's compilation pipeline, see [[JAX Tracing and Compilation]].
For pitfalls shared by both, see [[JIT Pitfalls - Impure Functions and Side Effects]].
For the conceptual overview, see [[JIT Compilation Mechanics]].
