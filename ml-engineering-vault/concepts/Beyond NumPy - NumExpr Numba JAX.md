---
type: concept
title: "Beyond NumPy - NumExpr Numba JAX"
libs:
  - "numpy"
  - "general"
tags:
  - "concept/optimization"
related:
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[Accelerated Python - GPU Acceleration]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[GPU Library Decision Guide]]"
  - "[[JIT Compilation Mechanics]]"
  - "[[Numba JIT Deep Dive]]"
  - "[[JAX Tracing and Compilation]]"
  - "[[JIT Pitfalls - Impure Functions and Side Effects]]"
created: 2026-05-14
updated: 2026-05-14
---

# Beyond NumPy — NumExpr, Numba, JAX

> NumPy's [[NumPy Multi-Pass Bottleneck|multi-pass bottleneck]] means it's 3–10× slower than it needs to be for pure math on large arrays. Three libraries solve this by compiling your expression into a **single-pass** loop: NumExpr (string-based virtual machine), Numba (LLVM JIT compiler), and JAX (XLA compiler). Each trades off simplicity against power.

---

## NumExpr — Fastest Way to Speed Up a NumPy Formula

### Intuition

NumExpr takes a math expression as a **string**, parses it, and executes it in a custom virtual machine that processes the expression element-by-element. Instead of NumPy's "compute `b**2` for all 5M elements, store to RAM, then compute `4*a` for all 5M elements, store to RAM...", NumExpr does "for element 0: compute the whole formula; for element 1: compute the whole formula; ...".

This means:
- **No temporary arrays** — intermediate values live in CPU registers
- **One pass through RAM** — reads `a[i], b[i], c[i]` once, writes `result[i]` once
- **Automatic multithreading** — uses all CPU cores by default

The tradeoff: you write your expression as a string, not Python code. NumExpr only supports a fixed set of math operations (arithmetic, trig, comparisons) — no custom functions, no conditionals.

### In Code — Basic Usage

```python
import numpy as np
import numexpr as ne

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)

# NumPy version — multi-pass, creates temporaries
result_numpy = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

# NumExpr version — single-pass, no temporaries
# Variables are looked up by name in the calling scope
result_numexpr = ne.evaluate("(-b + sqrt(b**2 - 4*a*c)) / (2*a)")

# Verify correctness
print(f"Max difference: {np.abs(result_numpy - result_numexpr).max():.2e}")
# Max difference: 0.00e+00
```

### In Code — What NumExpr Supports

```python
import numpy as np
import numexpr as ne

x = np.linspace(0.1, 10, 1_000_000)
y = np.linspace(1, 5, 1_000_000)

# Arithmetic
ne.evaluate("x + y")
ne.evaluate("x * y - 3.14")
ne.evaluate("x ** 2 + y ** 0.5")

# Math functions (same names as C, not NumPy)
ne.evaluate("sqrt(x)")
ne.evaluate("log(x)")        # natural log (not log10!)
ne.evaluate("log10(x)")
ne.evaluate("exp(x)")
ne.evaluate("sin(x) + cos(y)")
ne.evaluate("arctan2(y, x)")
ne.evaluate("abs(x - y)")

# Comparisons → boolean array
mask = ne.evaluate("(x > 2) & (y < 4)")
print(f"Elements matching: {mask.sum()}")

# Complex expressions
ne.evaluate("where(x > 5, x**2, sqrt(x))")  # ternary if/else

# Thread control
ne.set_num_threads(4)  # limit to 4 cores
result = ne.evaluate("x**2 + y**2")
print(f"Using {ne.detect_number_of_threads()} threads")
```

### In Code — What NumExpr Cannot Do

```python
import numpy as np
import numexpr as ne

# ❌ Custom Python functions
# ne.evaluate("my_func(x)")  → NameError

# ❌ Indexing or slicing
# ne.evaluate("x[y > 3]")  → SyntaxError

# ❌ Reductions (sum, mean, max)
# ne.evaluate("sum(x)")  → not supported
# Workaround: ne.evaluate("x**2 + y**2").sum()

# ❌ Matrix operations
# ne.evaluate("x @ y")  → not supported

# ❌ String operations
# ne.evaluate("x + 'hello'")  → not supported
```

### Common Misconceptions

- **"NumExpr is always faster than NumPy"** — For small arrays (< 10K elements), NumExpr's parsing overhead makes it slower. It shines for arrays > 100K elements.
- **"The string syntax is a hack"** — It's the key feature. Because NumExpr sees the entire expression at once (unlike NumPy, which sees one operation at a time), it can plan a single-pass execution.
- **"NumExpr compiles to machine code"** — No. It runs on a custom bytecode interpreter (virtual machine). This is faster than NumPy's multi-pass but slower than true compiled code (Numba, JAX).

---

## Numba — JIT-Compile Python to Machine Code

### Intuition

Numba takes a Python function and compiles it to native machine code using LLVM (the same compiler backend that powers Clang/C++). Unlike NumExpr, you write **normal Python** — no strings. Numba understands NumPy types and operations natively.

The key decorator for array math is `@nb.vectorize`: it takes a function that works on scalars and automatically generates a function that works on arrays, processing element-by-element in compiled code.

There's a first-call cost: Numba compiles the function the first time you call it (typically 0.1–2s). After that, it runs at near-C speed.

### In Code — @vectorize for Element-wise Operations

```python
import numpy as np
import numba as nb

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)])
def quadratic_numba(a, b, c):
    """Write scalar logic — Numba broadcasts it over arrays."""
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)


# First call triggers compilation (slow)
result = quadratic_numba(a, b, c)

# Subsequent calls are fast (~4ms for 5M elements)
result = quadratic_numba(a, b, c)

# Verify
result_numpy = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
print(f"Max difference: {np.abs(result - result_numpy).max():.2e}")
```

### In Code — @jit for General Loops

```python
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances — O(n²) loop that NumPy
    can do with broadcasting, but Numba does it without the huge
    temporary matrix.

    points: shape (n, d)
    returns: shape (n, n)
    """
    n = points.shape[0]
    d = points.shape[1]
    result = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):
            dist = 0.0
            for k in range(d):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            result[i, j] = dist
            result[j, i] = dist

    return result


# Generate 2000 points in 3D
rng = np.random.default_rng(42)
points = rng.standard_normal((2000, 3))

# First call compiles
distances = pairwise_distances(points)

# Compare with scipy
from scipy.spatial.distance import cdist

expected = cdist(points, points)
print(f"Max difference: {np.abs(distances - expected).max():.2e}")
```

### In Code — @jit with Parallel Loops

```python
import numpy as np
import numba as nb


@nb.jit(nopython=True, parallel=True)
def row_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row to sum to 1, using parallel threads."""
    result = np.empty_like(matrix)
    n_rows = matrix.shape[0]

    for i in nb.prange(n_rows):  # prange = parallel range
        row_sum = 0.0
        for j in range(matrix.shape[1]):
            row_sum += matrix[i, j]
        for j in range(matrix.shape[1]):
            result[i, j] = matrix[i, j] / row_sum

    return result


rng = np.random.default_rng(42)
mat = rng.uniform(0, 1, (10_000, 500))

result = row_normalize(mat)
print(f"Row sums (should be all 1.0): {result.sum(axis=1)[:5]}")
```

### Common Misconceptions

- **"Numba can compile any Python"** — `nopython=True` mode supports a subset: NumPy arrays, scalar math, loops, tuples, basic string ops. No dictionaries, no classes, no pandas, no custom objects.
- **"@vectorize and @jit are the same"** — `@vectorize` is for element-wise scalar functions broadcast over arrays. `@jit` is for arbitrary functions with loops, conditionals, and array indexing.
- **"I should Numba everything"** — If NumPy already vectorises your operation, Numba won't be much faster (NumPy's inner loops are already C). Numba shines when you have loops that **can't** be vectorised (see [[Vectorization Expressiveness Limits]]).

---

## JAX — XLA Compilation + Autodiff + GPU

### Intuition

JAX is a NumPy replacement from Google that compiles expressions using XLA (the same compiler that powers TensorFlow). `jax.jit` traces your function, builds a computation graph, and compiles it to optimised machine code.

Unlike Numba, JAX operates at the **array** level, not the scalar level. You write NumPy-like code using `jax.numpy` instead of `numpy`, and JAX fuses operations across the whole expression. The compilation is more aggressive than NumExpr's virtual machine.

JAX's killer features beyond speed: automatic differentiation (`jax.grad`) and seamless GPU/TPU execution (same code, different backend).

### In Code — Basic jax.jit

```python
import jax
import jax.numpy as jnp

# Force CPU to compare fairly with NumPy
jax.config.update("jax_platform_name", "cpu")
# Enable float64 (JAX defaults to float32)
jax.config.update("jax_enable_x64", True)

import numpy as np

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)


@jax.jit
def quadratic_jax(a, b, c):
    """Same formula, using jax.numpy instead of numpy."""
    return (-b + jnp.sqrt(b**2 - 4 * a * c)) / (2 * a)


# First call triggers XLA compilation (slow, ~1-3s)
result_jax = quadratic_jax(a, b, c)

# IMPORTANT: JAX operations are asynchronous on GPU.
# .block_until_ready() forces synchronization for accurate timing.
result_jax = quadratic_jax(a, b, c).block_until_ready()

# Convert back to numpy for comparison
result_numpy = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
print(f"Max difference: {np.abs(np.asarray(result_jax) - result_numpy).max():.2e}")
```

### In Code — JAX's Unique Power: Autodiff

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def f(x):
    """A function we want to differentiate: f(x) = x³ + 2x² - 5x + 3"""
    return x**3 + 2*x**2 - 5*x + 3


# Automatic derivatives — no manual calculus needed
df = jax.grad(f)       # f'(x) = 3x² + 4x - 5
ddf = jax.grad(df)     # f''(x) = 6x + 4

x = 2.0
print(f"f({x})  = {f(x)}")       # 2³ + 2·4 - 10 + 3 = 9
print(f"f'({x}) = {df(x)}")      # 3·4 + 8 - 5 = 15
print(f"f''({x}) = {ddf(x)}")    # 12 + 4 = 16
```

### In Code — JAX Gotchas

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ❌ GOTCHA 1: JAX arrays are IMMUTABLE (no in-place updates)
arr = jnp.array([1, 2, 3, 4, 5])
# arr[0] = 99  → TypeError!
# Instead:
arr = arr.at[0].set(99)
print(arr)  # [99, 2, 3, 4, 5]


# ❌ GOTCHA 2: jax.jit does NOT support data-dependent control flow
@jax.jit
def bad_jit(x):
    if x > 0:       # ← this branch depends on the VALUE of x
        return x**2  #   which isn't known at compile time
    return -x

# bad_jit(1.0)  → ConcretizationTypeError!

# ✅ Use jax.lax.cond instead:
@jax.jit
def good_jit(x):
    return jax.lax.cond(x > 0, lambda x: x**2, lambda x: -x, x)

print(good_jit(3.0))   # 9.0
print(good_jit(-3.0))  # 3.0


# ❌ GOTCHA 3: Python-level side effects are captured once at trace time
@jax.jit
def sneaky(x):
    print("I only print during tracing, not on every call!")
    return x + 1

sneaky(1.0)  # prints
sneaky(2.0)  # does NOT print — uses cached compiled code


# ❌ GOTCHA 4: Random numbers work differently
# NumPy: np.random.normal(0, 1, 5)  ← implicit global state
# JAX: explicit key management
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
samples1 = jax.random.normal(key1, shape=(5,))
samples2 = jax.random.normal(key2, shape=(5,))
print(f"samples1: {samples1}")
print(f"samples2: {samples2}")
```

### Common Misconceptions

- **"JAX is just a faster NumPy"** — JAX is a *different programming model*. Its functional purity constraints (no mutation, no side effects, explicit randomness) are the price you pay for autodiff and compilation. Think of it as a DSL that happens to look like NumPy.
- **"jax.jit compiles Python"** — No. It *traces* Python to build an XLA computation graph, then compiles that graph. Your Python code only runs once (during tracing). This is why data-dependent `if` statements break.
- **"I need a GPU to benefit from JAX"** — On CPU, JAX's XLA compiler still fuses operations and does loop optimisations. It's a meaningful speedup over NumPy even without a GPU.

---

## Decision Guide — Which Tool When?

| Situation | Tool | Why |
|-----------|------|-----|
| Simple math formula on large arrays | **NumExpr** | Zero learning curve, just write a string |
| Scalar loop that can't be vectorised | **Numba @jit** | Compiles arbitrary loops to machine code |
| Element-wise function on arrays | **Numba @vectorize** | Write scalar logic, get array broadcast free |
| Need gradients (ML training) | **JAX** | `jax.grad` is automatic differentiation |
| Targeting GPU/TPU | **JAX** or **CuPy** | Same code runs on CPU and GPU |
| Prototyping, quick fix | **NumPy** | Already works, readable, debuggable |

For the performance ceiling that motivates all of this, see [[NumPy Multi-Pass Bottleneck]].
For algorithms that resist vectorization entirely, see [[Vectorization Expressiveness Limits]].
For how JIT compilation works under the hood, see [[JIT Compilation Mechanics]].
For deep dives: [[Numba JIT Deep Dive]] and [[JAX Tracing and Compilation]].
For pitfalls with globals and side effects, see [[JIT Pitfalls - Impure Functions and Side Effects]].
