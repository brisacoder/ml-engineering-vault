---
type: concept
title: "JIT Compilation Mechanics"
libs:
  - "numpy"
  - "general"
tags:
  - "concept/optimization"
related:
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Numba JIT Deep Dive]]"
  - "[[JAX Tracing and Compilation]]"
  - "[[JIT Pitfalls - Impure Functions and Side Effects]]"
  - "[[Vectorization Expressiveness Limits]]"
created: 2026-05-14
updated: 2026-05-14
---

# JIT Compilation Mechanics

> JIT (Just-In-Time) compilation means translating Python code into optimised machine code **at runtime**, the first time a function is called. After that first call, the compiled version is cached and reused. This is how Numba and JAX solve the [[NumPy Multi-Pass Bottleneck]] — by fusing multiple array operations into a single compiled pass. This note covers how JIT works conceptually; for library-specific details, see [[Numba JIT Deep Dive]] and [[JAX Tracing and Compilation]].

---

## What "Just-In-Time" Actually Means

### Intuition

There are three ways to run code:

1. **Interpreted** (Python): the interpreter reads each line, figures out what to do, and does it. Flexible but slow — every operation pays the overhead of type-checking, dispatch, and the interpreter loop.

2. **Ahead-of-Time compiled** (C, Fortran, Rust): a compiler translates the entire program to machine code *before* it runs. Fast but inflexible — you need to know all types upfront and recompile when anything changes.

3. **Just-In-Time compiled** (Numba, JAX, PyTorch 2.0): the first time a function is called, the runtime sees the actual input types, compiles an optimised version for those specific types, and caches it. Subsequent calls skip compilation and run at compiled speed.

The tradeoff is clear: JIT gives you the speed of compiled code with the flexibility of Python, at the cost of a one-time compilation delay on the first call.

### In Code — Observing the Compilation Cost

```python
import numpy as np
import time

# === Simulating JIT behavior with a manual cache ===
# (This illustrates the concept; real JIT uses Numba/JAX)

class FakeJIT:
    """Demonstrates the JIT pattern: compile on first call, cache for later."""

    def __init__(self, func):
        self.func = func
        self._compiled = None
        self._compile_time = 0

    def __call__(self, *args):
        if self._compiled is None:
            # "Compilation" step — happens once
            start = time.perf_counter()
            self._compiled = self.func  # pretend we compiled it
            self._compile_time = time.perf_counter() - start
            print(f"  [JIT] Compiled '{self.func.__name__}' "
                  f"in {self._compile_time*1e6:.0f} µs (first call)")
        return self._compiled(*args)


@FakeJIT
def my_function(x):
    return np.sqrt(x**2 + 1)

arr = np.random.randn(1_000_000)

# First call: includes "compilation"
t0 = time.perf_counter()
result1 = my_function(arr)
t1 = time.perf_counter()
print(f"  First call:  {(t1-t0)*1e3:.2f} ms")

# Second call: uses cached compiled version
t0 = time.perf_counter()
result2 = my_function(arr)
t1 = time.perf_counter()
print(f"  Second call: {(t1-t0)*1e3:.2f} ms")
```

### Common Misconceptions

- **"JIT always makes code faster"** — Only if the function is called more than once. If you call a JIT-compiled function exactly once, you pay the compilation cost *and* get no benefit from caching.
- **"JIT compilation happens at import time"** — No. It happens at the first *call* with specific input types. The decorator itself (`@jax.jit`, `@nb.njit`) just marks the function for future compilation.
- **"The compiled version is always the same"** — Not necessarily. If you call a Numba function with `float64` arrays and later with `float32` arrays, it compiles *two* separate versions. JAX recompiles when input **shapes** change.

---

## Operation Fusion — Why JIT Makes NumPy Faster

### Intuition

The whole point of JIT compilation in the NumPy context is **operation fusion**. Instead of executing each array operation separately (creating temporary arrays that bust CPU caches — see [[NumPy Multi-Pass Bottleneck]]), the JIT compiler sees the *entire* expression and fuses it into a single loop.

Here's the mental model with the quadratic formula $x = \frac{-b + \sqrt{b^2 - 4ac}}{2a}$:

**NumPy (unfused):** 9 separate loops, 9 temporary arrays, 9 RAM round-trips:
```
Loop 1: for all i: tmp1[i] = b[i]²
Loop 2: for all i: tmp2[i] = 4 * a[i]
Loop 3: for all i: tmp3[i] = tmp2[i] * c[i]
...9 more loops...
```

**JIT-compiled (fused):** 1 loop, 0 temporary arrays, 1 RAM round-trip:
```
for i in range(n):
    result[i] = (-b[i] + sqrt(b[i]² - 4*a[i]*c[i])) / (2*a[i])
```

The fused version reads each element once, computes the entire formula in CPU registers, and writes the result once. This is why JIT-compiled code is 3–10× faster than NumPy on large arrays.

### In Code — The Performance Ladder

```python
import numpy as np
import time

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)


def benchmark(func, *args, n_runs: int = 10, label: str = "") -> float:
    """Benchmark returning median time in ms. Warmup included."""
    _ = func(*args)  # warmup / trigger compilation
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    median_ms = np.median(times) * 1000
    print(f"{label:<35s} {median_ms:8.2f} ms")
    return median_ms


# NumPy — unfused, multi-pass
def quadratic_numpy(a, b, c):
    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

t_numpy = benchmark(quadratic_numpy, a, b, c, label="NumPy (unfused)")

# The goal of JIT compilation is to match or beat this time
# by fusing all 9 operations into 1 loop.
# See [[Numba JIT Deep Dive]] and [[JAX Tracing and Compilation]]
# for the actual compiled versions.
print(f"\nNumPy baseline: {t_numpy:.1f} ms")
print(f"JIT target:     ~{t_numpy/5:.1f}–{t_numpy/3:.1f} ms (3-5× faster)")
```

### Common Misconceptions

- **"Fusion just means running the loop in C"** — It's more than that. Fusion eliminates temporary arrays entirely, which means the data stays in CPU registers or L1 cache. The speedup comes from **memory bandwidth savings**, not just faster loop overhead.
- **"NumPy could fuse operations if it wanted to"** — NumPy can't because Python calls each operation separately. By the time `np.sqrt` is called, the `b**2` operation is already finished and its result is in a temporary array. NumPy has no way to "look ahead" at the full expression. JIT compilers solve this by seeing the whole function at once during tracing/compilation.

---

## Two Paradigms: Imperative vs Array-Oriented JIT

### Intuition

Numba and JAX both do JIT compilation, but they force you into opposite programming styles:

**Numba = Imperative (scalar loops)**
You write explicit `for` loops over array indices, as if you were writing C. Numba compiles these scalar operations into machine code using LLVM.

**JAX = Array-Oriented (whole-array operations)**
You write NumPy-like code using `jax.numpy`. JAX traces the array operations, builds a computation graph, and compiles it using XLA. You never write index loops.

The diagram from the lecture maps it out:

```
                    Slow            Fast
                ┌──────────┬──────────────┐
Imperative      │  Python  │  Numba       │
(scalar loops)  │  for i   │  @nb.njit    │
                ├──────────┼──────────────┤
Array-oriented  │  NumPy   │  JAX         │
(whole-array)   │  a + b   │  @jax.jit    │
                └──────────┴──────────────┘
```

Python loops on scalars → slow. NumPy on arrays → faster. **Both Numba and JAX reach the same "fast" column**, just via different paths.

### In Code — Same Formula, Two Styles

```python
import numpy as np

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 5_000_000)
b = rng.uniform(10, 20, 5_000_000)
c = rng.uniform(-0.1, 0.1, 5_000_000)

# === NUMBA STYLE: imperative, scalar loops ===
# You write the loop explicitly — Numba compiles it to machine code

# import numba as nb
# @nb.njit
# def quadratic_numba(a, b, c):
#     n = a.shape[0]
#     out = np.empty(n)
#     for i in range(n):  # ← explicit index loop
#         out[i] = (-b[i] + np.sqrt(b[i]**2 - 4*a[i]*c[i])) / (2*a[i])
#     return out

# === JAX STYLE: array-oriented, no loops ===
# You write NumPy-like code — JAX traces and compiles the graph

# import jax
# import jax.numpy as jnp
# @jax.jit
# def quadratic_jax(a, b, c):
#     return (-b + jnp.sqrt(b**2 - 4*a*c)) / (2*a)  # ← no loop!

# Both produce the same result at the same speed (~2-4 ms for 5M elements)
# but the PROGRAMMING MODEL is fundamentally different.

# === IMPORTANT: they're not locked into one paradigm ===
# Numba CAN do array-oriented: @nb.vectorize
# JAX CAN do imperative: jax.lax.scan, jax.lax.fori_loop
# But each library is OPTIMISED for its primary paradigm.
```

### Common Misconceptions

- **"Imperative JIT is always faster"** — Not necessarily. JAX's XLA compiler can do optimisations (like SIMD vectorisation and memory layout transformations) that a scalar loop can't express. For pure element-wise math, they're roughly equal. For complex array operations (matmul, convolutions), JAX/XLA can be faster.
- **"I have to choose one"** — In practice, many projects use both. Numba for CPU-bound scalar loops that resist vectorisation (see [[Vectorization Expressiveness Limits]]). JAX for GPU-bound ML training where you also need autodiff.
- **"Array-oriented = no loops at all"** — JAX has loop primitives (`jax.lax.scan`, `jax.lax.fori_loop`, `jax.lax.while_loop`) for when you genuinely need iteration. But they're functional (no mutation) and compile down to XLA loops, not Python loops.

---

## First-Call Cost and Caching

### Intuition

Both Numba and JAX have a noticeable delay on the first call — typically 100ms to several seconds, depending on function complexity. This is the compilation step. After that, the compiled function is cached.

**Numba caching:**
- In-memory cache by default (lost when the Python process exits).
- Persistent disk cache with `@nb.njit(cache=True)` — survives process restarts.
- Recompiles if you call with a new input **type** (e.g., float32 vs float64).

**JAX caching:**
- In-memory cache by default.
- Persistent disk cache available via `jax.config.update("jax_compilation_cache_dir", "/path")`.
- Recompiles if you call with a new input **shape** or **dtype**.

### In Code — Measuring First-Call vs Subsequent Calls

```python
import numpy as np
import time

rng = np.random.default_rng(42)
a = rng.uniform(5, 10, 1_000_000)
b = rng.uniform(10, 20, 1_000_000)
c = rng.uniform(-0.1, 0.1, 1_000_000)

# Simulating what you'd see with Numba or JAX:
#
# Call 1 (includes compilation):
#   quadratic_numba(a, b, c)  →  ~500 ms  (compilation + execution)
#
# Call 2 (cached):
#   quadratic_numba(a, b, c)  →  ~2 ms    (execution only)
#
# Call 3 with DIFFERENT TYPES (triggers recompilation):
#   a32 = a.astype(np.float32)
#   quadratic_numba(a32, b32, c32)  →  ~500 ms  (new compilation)

# The practical lesson: don't benchmark the first call!
# Always do a warmup call before timing.

# Bad benchmark:
def bad_benchmark():
    start = time.perf_counter()
    result = np.sqrt(a**2 + b**2)  # pretend this is JIT-compiled
    return (time.perf_counter() - start) * 1000

# Good benchmark:
def good_benchmark():
    _ = np.sqrt(a**2 + b**2)      # warmup (triggers compilation)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = np.sqrt(a**2 + b**2)  # now measuring execution only
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)

print(f"Naive timing:  {bad_benchmark():.2f} ms (includes 'compilation')")
print(f"Proper timing: {good_benchmark():.2f} ms (execution only)")
```

### Common Misconceptions

- **"JIT compilation is free"** — The first call can be 100–1000× slower than subsequent calls. For short scripts that call a function once, JIT gives you negative benefit. JIT shines in long-running programs, servers, and training loops.
- **"I should JIT everything"** — Only JIT functions that are called repeatedly and are performance-critical. The compilation overhead of decorating every function outweighs the benefit.
- **"Numba's `cache=True` makes the first call fast"** — Only on the *second run* of the program. The very first compilation still happens, but the result is saved to disk so the next program run loads it instantly.

---

For Numba-specific details, see [[Numba JIT Deep Dive]].
For JAX's compilation pipeline, see [[JAX Tracing and Compilation]].
For pitfalls that affect both, see [[JIT Pitfalls - Impure Functions and Side Effects]].
For why fusion matters, see [[NumPy Multi-Pass Bottleneck]].
