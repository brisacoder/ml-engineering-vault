---
type: concept
title: "JAX Tracing and Compilation"
libs:
  - "general"
tags:
  - "concept/optimization"
related:
  - "[[JIT Compilation Mechanics]]"
  - "[[Numba JIT Deep Dive]]"
  - "[[JIT Pitfalls - Impure Functions and Side Effects]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[Accelerated Python - GPU Acceleration]]"
created: 2026-05-14
updated: 2026-05-14
---

# JAX Tracing and Compilation

> JAX compiles Python functions through a unique 4-step pipeline: **trace → JaxPr → StableHLO → XLA → execute**. Understanding this pipeline is essential because JAX's limitations (no data-dependent control flow, no dynamic shapes) are direct consequences of the tracing step. This note covers the pipeline in detail and why each limitation exists. For the conceptual overview, see [[JIT Compilation Mechanics]]. For basic usage, see [[Beyond NumPy - NumExpr Numba JAX]].

---

## The 4-Step Compilation Pipeline

### Intuition

When you call a `@jax.jit`-decorated function for the first time, JAX doesn't run your Python code in the normal sense. Instead, it does this:

**Step 1: Trace** — JAX runs your function with **abstract "tracer" objects** instead of real data. These tracers have shape and dtype but no actual values. As your code calls `jnp.sqrt`, `+`, etc., JAX records each operation into a graph.

**Step 2: JaxPr** — The traced operations become a JaxPr (JAX Program Representation) — an intermediate representation (IR) that looks like the "pedantic" version of your code from [[NumPy Multi-Pass Bottleneck]]. Each operation is an explicit node.

**Step 3: StableHLO** — The JaxPr is lowered (translated) into StableHLO, a standardised IR used by multiple ML compilers. This is where the representation stops being Python-specific.

**Step 4: XLA Compilation** — Google's XLA (Accelerated Linear Algebra) compiler takes the StableHLO program and produces optimised machine code. This is where **operation fusion** happens — XLA sees the whole expression and fuses operations into a single pass. The compiled code targets CPU, GPU, or TPU depending on where your data lives.

After step 4, the compiled function is cached. Subsequent calls go directly to the compiled code — no tracing, no compilation.

### In Code — Watching Each Step

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np

rng = np.random.default_rng(42)
a = jnp.asarray(rng.uniform(5, 10, 1000))
b = jnp.asarray(rng.uniform(10, 20, 1000))
c = jnp.asarray(rng.uniform(-0.1, 0.1, 1000))


@jax.jit
def quadratic_jax(a, b, c):
    return (-b + jnp.sqrt(b**2 - 4*a*c)) / (2*a)


# STEP 1+2: Trace the function → get JaxPr
# .trace() runs your function with abstract tracers and captures the graph
traced = quadratic_jax.trace(a, b, c)
print("=== STEP 1+2: JaxPr ===")
print(traced.jaxpr)
# You'll see something like:
#   { lambda ; a:f64[1000] b:f64[1000] c:f64[1000]. let
#       d:f64[1000] = neg b
#       e:f64[1000] = integer_pow[y=2] b
#       f:f64[1000] = mul 4.0 a
#       g:f64[1000] = mul f c
#       h:f64[1000] = sub e g
#       ...
#   }
# This is the "pedantic" decomposition — each line is one operation.

print()

# STEP 3: Lower to StableHLO
lowered = quadratic_jax.lower(a, b, c)
print("=== STEP 3: StableHLO (first 500 chars) ===")
hlo_text = lowered.as_text()
print(hlo_text[:500])
# StableHLO looks like assembly-ish IR with operations like:
#   %0 = stablehlo.negate %arg1
#   %1 = stablehlo.multiply %arg1, %arg1
#   ...

print()

# STEP 4: Compile with XLA
compiled = lowered.compile()
print("=== STEP 4: Compiled! ===")
print(f"Type: {type(compiled)}")

# Execute the compiled program
result = compiled(a, b, c)
print(f"Result shape: {result.shape}, dtype: {result.dtype}")
print(f"First 3 values: {result[:3]}")
```

### Common Misconceptions

- **"JAX compiles Python to machine code"** — JAX doesn't compile *Python*. It *traces* Python to build a computation graph, then compiles *that graph*. Your Python code runs exactly once (during tracing) and is then discarded.
- **"JaxPr and StableHLO are just for debugging"** — They're the actual program that gets compiled. Understanding JaxPr helps you understand JAX's limitations.
- **"XLA is specific to JAX"** — XLA is also used by TensorFlow. It's a general-purpose ML compiler that optimises across operations.

---

## Tracing — How JAX "Sees" Your Code

### Intuition

Tracing is the crucial first step, and it's where most JAX confusion comes from. Here's what happens:

1. JAX creates **tracer objects** — they look like arrays but have no data, only shape and dtype metadata.
2. JAX calls your function with these tracers.
3. Every `jnp.*` operation on a tracer is *recorded* into the graph, not *executed*.
4. Python control flow (`if`, `for`, `while`) runs **during tracing**, using the tracer objects.

The critical consequence: **anything that depends on actual data values cannot be traced.** A Python `if` statement needs a concrete `True` or `False`, but a tracer doesn't have a value — it's just a placeholder.

### In Code — What the Tracer Looks Like

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.jit
def simple_function(x):
    # During tracing, x is NOT a real array — it's a tracer.
    # This print runs ONCE during tracing, showing you the tracer:
    print(f"  During tracing, x = {x}")
    print(f"  Type: {type(x)}")

    # jnp operations are RECORDED, not executed:
    y = x ** 2
    z = jnp.sum(y)
    return z


print("First call (triggers tracing + compilation):")
result = simple_function(jnp.array([1.0, 2.0, 3.0]))
print(f"Result: {result}")

print("\nSecond call (uses cached compiled function):")
result = simple_function(jnp.array([4.0, 5.0, 6.0]))
print(f"Result: {result}")
# Notice: the print inside the function does NOT run on the second call!
```

### Common Misconceptions

- **"JAX runs my function every time I call it"** — No. JAX runs your Python function *once* (during tracing). After that, it runs the compiled XLA program directly. Your Python code is just a way to *describe* the computation.
- **"print() inside @jax.jit works normally"** — It only prints during tracing (first call). On subsequent calls, the print never executes because Python code isn't involved.

---

## Limitation 1: No Data-Dependent Control Flow

### Intuition

This is JAX's most important limitation. During tracing, JAX doesn't have the actual data values — just shapes and dtypes. So a Python `if` that depends on the *value* of an array element cannot be resolved.

```python
if jnp.any(arr > 3):  # ← What is the value of arr? Tracer has no data!
    return jnp.sum(arr)
else:
    return jnp.prod(arr)
```

The tracer can't decide which branch to take, so JAX raises a `ConcretizationTypeError`.

### In Code — The Problem

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ❌ This FAILS because the if-condition depends on data values
@jax.jit
def broken_accumulate(arr):
    if jnp.any(arr > 3):  # needs actual values → ConcretizationTypeError
        return jnp.sum(arr)
    else:
        return jnp.prod(arr)


try:
    broken_accumulate(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
except jax.errors.TracerBoolConversionError as e:
    print(f"Error: {type(e).__name__}")
    print("JAX can't evaluate if-conditions that depend on data values!")
```

### In Code — The Fix: jax.lax.cond

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ✅ Use jax.lax.cond for data-dependent branching
@jax.jit
def fixed_accumulate(arr):
    return jax.lax.cond(
        jnp.any(arr > 3),         # condition (traced, not evaluated)
        lambda a: jnp.sum(a),     # true branch
        lambda a: jnp.prod(a),    # false branch
        arr                        # argument passed to chosen branch
    )


# Both branches are compiled into the program
# The correct one is chosen at RUNTIME (not trace time)
result1 = fixed_accumulate(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(f"[1,2,3,4,5] has elements > 3 → sum = {result1}")

result2 = fixed_accumulate(jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
print(f"[0.5,1,1.5,2,2.5] no elements > 3 → prod = {result2}")
```

### In Code — Python if That DOES Work With @jax.jit

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ✅ Python if on STATIC values (known at trace time) works fine!
@jax.jit
def configurable_norm(arr, use_l1=False):
    # use_l1 is a regular Python bool — known at trace time
    if use_l1:
        return jnp.sum(jnp.abs(arr))
    else:
        return jnp.sqrt(jnp.sum(arr**2))


arr = jnp.array([3.0, 4.0])
print(f"L2 norm: {configurable_norm(arr, use_l1=False)}")  # 5.0
print(f"L1 norm: {configurable_norm(arr, use_l1=True)}")   # 7.0

# NOTE: JAX compiles TWO versions — one for use_l1=True, one for False.
# Each version is cached separately.
```

### Common Misconceptions

- **"I can never use if/else in JAX"** — You can, as long as the condition doesn't depend on array *values*. Conditions on shapes, dtypes, or Python variables are fine — they're resolved at trace time.
- **"jax.lax.cond compiles only the taken branch"** — No. Both branches are compiled. At runtime, XLA executes only the correct one, but both exist in the compiled program.

---

## Limitation 2: No Dynamic Shapes

### Intuition

During tracing, JAX needs to know the **shape** of every intermediate array. If an operation produces an array whose size depends on the data values, JAX can't trace it.

The classic example: boolean indexing. `arr[arr > 3]` produces an array whose length depends on *how many* elements are greater than 3 — which the tracer doesn't know.

### In Code — The Problem and Workarounds

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# ❌ Dynamic shape: output size depends on data
@jax.jit
def broken_filter_sum(arr):
    filtered = arr[arr > 3.0]  # how many elements? tracer doesn't know!
    return jnp.sum(filtered)


try:
    broken_filter_sum(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
except Exception as e:
    print(f"Error: {type(e).__name__}")
    print("Boolean indexing creates dynamic shapes!")


# ✅ Workaround 1: use jnp.where to keep fixed shape
@jax.jit
def fixed_filter_sum_where(arr):
    # Replace unwanted elements with 0, then sum
    return jnp.sum(jnp.where(arr > 3.0, arr, 0.0))


result = fixed_filter_sum_where(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(f"Sum of elements > 3: {result}")  # 9.0


# ✅ Workaround 2: use masking (multiply by boolean mask)
@jax.jit
def fixed_filter_sum_mask(arr):
    mask = arr > 3.0
    return jnp.sum(arr * mask)


result = fixed_filter_sum_mask(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))
print(f"Sum of elements > 3: {result}")  # 9.0
```

### Common Misconceptions

- **"jnp.where creates a dynamic shape"** — No. `jnp.where(condition, x, y)` returns an array with the *same shape* as the inputs — it just picks values element-wise. It's `arr[boolean_mask]` that creates a dynamic shape.
- **"I can never filter arrays in JAX"** — You can, you just have to use fixed-shape alternatives like `jnp.where` or mask-and-reduce patterns.

---

## JAX on GPU — Zero Code Changes

### Intuition

Because JAX compiles to XLA (which targets CPU, GPU, and TPU), the same `@jax.jit` function runs on any device without code changes. Numba requires a completely different API (`numba.cuda`) for GPU programming (see [[Numba JIT Deep Dive#numba.cuda — GPU Programming]]).

With JAX, you just move the data to the GPU and call the same function:

```python
# CPU execution:
result_cpu = quadratic_jax(a_cpu, b_cpu, c_cpu)

# GPU execution (same function, different data location):
a_gpu = jax.device_put(a, jax.devices("gpu")[0])
result_gpu = quadratic_jax(a_gpu, b_gpu, c_gpu)
```

### In Code — Device Management

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# Check available devices
print(f"Available devices: {jax.devices()}")
# CPU-only: [CpuDevice(id=0)]
# With GPU:  [CpuDevice(id=0), GpuDevice(id=0)]


@jax.jit
def compute(x):
    return jnp.sum(x ** 2)


# Data lives on CPU by default
x = jnp.arange(1000.0)
print(f"x lives on: {x.devices()}")

# Explicitly place data on a device
x_placed = jax.device_put(x, jax.devices()[0])
result = compute(x_placed)
print(f"Result: {result}, lives on: {result.devices()}")

# KEY INSIGHT: the @jax.jit function is the SAME.
# XLA compiles a different optimised program depending on the target device.
# On GPU, it uses CUDA kernels. On CPU, it uses AVX/SSE instructions.
# On TPU, it uses TPU-specific instructions.
# Your code doesn't change.
```

### Common Misconceptions

- **"JAX always uses GPU if available"** — By default, yes — JAX picks the "best" available device. But you can force CPU with `jax.config.update("jax_platform_name", "cpu")` or explicitly place data with `jax.device_put`.
- **"Moving data to GPU is free"** — The PCIe bus transfer (CPU RAM → GPU VRAM) takes time. For small arrays, this transfer cost can exceed the computation time. GPU shines when data stays on the device across many operations.

---

## JAX's Functional Loop Primitives

### Intuition

When you genuinely need loops in JAX (see [[Vectorization Expressiveness Limits]]), you can't use Python `for` with data-dependent conditions. Instead, JAX provides functional loop primitives that compile into XLA loops:

| Primitive | Python Equivalent | Use Case |
|-----------|------------------|----------|
| `jax.lax.fori_loop(lo, hi, body, init)` | `for i in range(lo, hi)` | Fixed iteration count |
| `jax.lax.while_loop(cond, body, init)` | `while cond(state)` | Data-dependent exit |
| `jax.lax.scan(f, init, xs)` | `for x in xs: carry = f(carry, x)` | Sequential state + collect outputs |
| `jax.lax.cond(pred, true_fn, false_fn, *args)` | `if pred: ... else: ...` | Data-dependent branching |

### In Code — jax.lax.fori_loop

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# Fixed-count loop: compute x^n using repeated multiplication
@jax.jit
def power_via_loop(x, n):
    def body(i, val):
        return val * x

    return jax.lax.fori_loop(0, n, body, 1.0)


print(f"2^10 = {power_via_loop(2.0, 10)}")  # 1024.0
print(f"3^5  = {power_via_loop(3.0, 5)}")   # 243.0
```

### In Code — jax.lax.scan (Most Useful)

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# scan = fold with intermediate outputs collected
# Perfect for recurrences like exponential moving average

@jax.jit
def ema(values, alpha=0.3):
    """Exponential moving average: y[t] = alpha * x[t] + (1-alpha) * y[t-1]

    This is a recurrence — y[t] depends on y[t-1], so it can't be
    vectorised with NumPy. But jax.lax.scan handles it cleanly.
    """
    def step(carry, x):
        # carry = previous EMA value
        new_carry = alpha * x + (1 - alpha) * carry
        return new_carry, new_carry  # (new carry, output to collect)

    init = values[0]
    _, ema_values = jax.lax.scan(step, init, values[1:])

    # Prepend the initial value
    return jnp.concatenate([jnp.array([init]), ema_values])


prices = jnp.array([100., 102., 98., 103., 101., 105., 99., 97., 104., 106.])
smoothed = ema(prices, alpha=0.3)
print("Prices:   ", prices)
print("EMA(0.3): ", smoothed)
```

### Common Misconceptions

- **"`jax.lax.scan` is just a slow loop"** — No. XLA compiles it into an efficient loop that runs at compiled speed. It's the right tool for sequential computations in JAX (RNNs, time series, state machines).
- **"I should always use jax.lax.fori_loop instead of Python for"** — Only inside `@jax.jit`. Outside of JIT, Python `for` loops work fine with JAX arrays.

---

For the imperative JIT alternative, see [[Numba JIT Deep Dive]].
For pitfalls shared by both, see [[JIT Pitfalls - Impure Functions and Side Effects]].
For the conceptual overview, see [[JIT Compilation Mechanics]].
