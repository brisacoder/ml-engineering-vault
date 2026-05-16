---
type: concept
title: "JIT Pitfalls - Impure Functions and Side Effects"
libs:
  - "numpy"
  - "general"
tags:
  - "concept/optimization"
related:
  - "[[JIT Compilation Mechanics]]"
  - "[[Numba JIT Deep Dive]]"
  - "[[JAX Tracing and Compilation]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
created: 2026-05-14
updated: 2026-05-14
---

# JIT Pitfalls — Impure Functions and Side Effects

> Both Numba and JAX **bake in** the state of the world at compilation time. Global variables, module-level constants, and side effects (like `print`) are captured *once* during the first call and then frozen. Changing a global variable after compilation has **no effect** on the compiled function. This catches everyone at least once, and it's the same bug in both libraries for the same fundamental reason.

---

## The Global Variable Trap

### Intuition

When Numba or JAX compiles your function, they see the values of all variables at that moment. If your function reads a global variable, the *current value* of that global is compiled into the machine code as a constant. Changing the global later doesn't re-trigger compilation — the compiled function keeps using the old value.

This is a direct consequence of how JIT works (see [[JIT Compilation Mechanics]]): the function is compiled *once* and cached. The compiled code has no reference back to your Python global — it's been replaced by a literal constant.

### In Code — The Bug (Numba)

```python
import numpy as np
import numba as nb

# A global variable that controls behavior
use_sum = False


@nb.njit
def accumulate(arr):
    """This function reads the global `use_sum`.
    But Numba captures its value at FIRST CALL and never reads it again.
    """
    if use_sum:
        return np.sum(arr)
    else:
        return np.prod(arr)


arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# First call: use_sum is False → Numba compiles the "prod" path
print(f"use_sum=False: {accumulate(arr)}")  # 120.0 (product) ✅

# Now flip the global
use_sum = True
print(f"use_sum=True:  {accumulate(arr)}")  # 120.0 still! ❌
# Expected 15.0 (sum), got 120.0 (product)
# The compiled function is FROZEN with use_sum=False
```

### In Code — The Same Bug (JAX)

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

use_sum = False


@jax.jit
def accumulate_jax(arr):
    """Same trap with JAX. The if-branch is resolved during TRACING,
    when use_sum is False. The 'True' branch never enters the compiled program.
    """
    if use_sum:
        return jnp.sum(arr)
    else:
        return jnp.prod(arr)


arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# First call: use_sum is False → JAX traces only the "prod" path
print(f"use_sum=False: {accumulate_jax(arr)}")  # 120.0 ✅

# Flip the global
use_sum = True
print(f"use_sum=True:  {accumulate_jax(arr)}")  # 120.0 still! ❌
# The traced program ONLY contains the prod operation.
# JAX never even SAW the sum branch.
```

### In Code — Why JAX's Version Is Extra Subtle

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

use_sum = False


@jax.jit
def accumulate_jax(arr):
    if use_sum:
        return jnp.sum(arr)
    else:
        return jnp.prod(arr)


arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

# We can PROVE the sum branch doesn't exist by inspecting the compiled program:
traced = accumulate_jax.trace(arr)
print("JaxPr (the traced program):")
print(traced.jaxpr)
# You'll see: reduce_prod — no sum anywhere!
# The if/else ran during TRACING (when use_sum was False),
# so JAX only recorded the else branch.

print("\nStableHLO:")
lowered = accumulate_jax.lower(arr)
print(lowered.as_text())
# Only multiplication/reduction operations — the sum literally doesn't exist
```

### Common Misconceptions

- **"This is a JAX bug"** — It's not a bug, it's a fundamental consequence of the trace-compile-cache model. The fix is to write **pure functions**.
- **"Numba is different from JAX here"** — The symptom is identical. Numba resolves the `if` during compilation and inlines the result as a constant. JAX resolves it during tracing. Both freeze the outcome.
- **"Setting use_sum=True before the first call would fix it"** — Only if you never change it again. The real fix is to not use globals at all.

---

## The Fix: Pass Configuration as Arguments

### Intuition

The solution is simple: **make your function pure**. Instead of reading globals, pass everything the function needs as arguments. This way, different argument values trigger different compilations (both Numba and JAX cache per-argument-type/value).

### In Code — The Correct Pattern (Numba)

```python
import numpy as np
import numba as nb


@nb.njit
def accumulate_pure(arr, use_sum):
    """Configuration is an ARGUMENT, not a global.
    Numba compiles separate versions for use_sum=True and use_sum=False.
    """
    if use_sum:
        return np.sum(arr)
    else:
        return np.prod(arr)


arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Product: {accumulate_pure(arr, False)}")  # 120.0 ✅
print(f"Sum:     {accumulate_pure(arr, True)}")   # 15.0  ✅
# Both work correctly because Numba compiles TWO specializations:
# one for use_sum=True, one for use_sum=False.
```

### In Code — The Correct Pattern (JAX)

```python
import jax
import jax.numpy as jnp
import functools

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# Option 1: Use static_argnums to mark non-array arguments
@functools.partial(jax.jit, static_argnums=(1,))
def accumulate_static(arr, use_sum):
    """use_sum is marked as STATIC — JAX treats different values
    as different programs and compiles each separately.
    """
    if use_sum:
        return jnp.sum(arr)
    else:
        return jnp.prod(arr)


arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Product: {accumulate_static(arr, False)}")  # 120.0 ✅
print(f"Sum:     {accumulate_static(arr, True)}")   # 15.0  ✅


# Option 2: Use jax.lax.cond for TRACED branching
# (when the condition depends on array data, not Python bools)
@jax.jit
def accumulate_cond(arr, threshold):
    """Branch based on DATA — both branches compiled, chosen at runtime."""
    return jax.lax.cond(
        jnp.max(arr) > threshold,
        lambda a: jnp.sum(a),
        lambda a: jnp.prod(a),
        arr
    )


print(f"Max > 3 → sum: {accumulate_cond(arr, 3.0)}")   # 15.0 ✅
print(f"Max > 10 → prod: {accumulate_cond(arr, 10.0)}") # 120.0 ✅
```

### Common Misconceptions

- **"static_argnums means the argument is constant"** — No. It means JAX treats each distinct value as a separate program. If you pass 100 different values for a static argument, JAX compiles 100 separate programs. Use it for small sets of options (True/False, enum values), not for continuous values.
- **"I should make everything static"** — Only scalar configuration values that determine program structure (which branch to take, how many iterations). Arrays should never be static — that would trigger recompilation for every new data point.

---

## Side Effects: print, Logging, File I/O

### Intuition

Side effects — anything that interacts with the outside world (printing, logging, writing files, incrementing counters) — only happen at **trace/compile time**, not at runtime. This is because the compiled function doesn't contain your Python code, only the mathematical operations that were traced.

### In Code — print Runs Once

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.jit
def debug_function(x):
    print(f"  [TRACE] This runs during tracing only! x = {x}")
    y = x ** 2
    print(f"  [TRACE] Computed y = {y}")
    return jnp.sum(y)


print("=== First call (tracing happens) ===")
result1 = debug_function(jnp.array([1.0, 2.0, 3.0]))
print(f"  Result: {result1}\n")

print("=== Second call (cached, no tracing) ===")
result2 = debug_function(jnp.array([4.0, 5.0, 6.0]))
print(f"  Result: {result2}\n")
# Notice: NO print output on the second call!

print("=== Third call with DIFFERENT SHAPE (re-traces!) ===")
result3 = debug_function(jnp.array([7.0, 8.0]))
print(f"  Result: {result3}")
# Print DOES run here — new shape triggers re-tracing!
```

### In Code — The Counter Trap

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

call_count = 0


@jax.jit
def counting_function(x):
    global call_count
    call_count += 1  # This only runs during tracing!
    return jnp.sum(x ** 2)


arr = jnp.array([1.0, 2.0, 3.0])

# Call 5 times
for _ in range(5):
    _ = counting_function(arr)

print(f"call_count = {call_count}")  # 1, not 5!
# The counter only incremented during tracing (first call).
# The subsequent 4 calls used the cached compiled function.
```

### In Code — Correct Debugging With jax.debug.print

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


@jax.jit
def debuggable_function(x):
    y = x ** 2
    # jax.debug.print is traced as an OPERATION in the graph,
    # so it runs on every call, not just during tracing
    jax.debug.print("y = {y}", y=y)
    return jnp.sum(y)


print("First call:")
r1 = debuggable_function(jnp.array([1.0, 2.0, 3.0]))

print("Second call:")
r2 = debuggable_function(jnp.array([4.0, 5.0, 6.0]))
# jax.debug.print runs BOTH times!
```

### Common Misconceptions

- **"jax.debug.print has no performance cost"** — It does. It inserts an operation into the XLA graph. Remove it in production code.
- **"Numba has the same print problem"** — Partially. Numba's `print` inside `@nb.njit` does work on every call (Numba handles `print` specially). But Numba still bakes in global values, so the global-variable trap is identical.

---

## The Pure Function Rule

### Intuition

The universal fix for all these pitfalls: **write pure functions**. A pure function:

1. Gets ALL its inputs through arguments (no globals, no closures over mutable state).
2. Returns ALL its outputs through return values (no mutation of external state).
3. Has no side effects (no print, no file I/O, no incrementing counters).

This isn't just good JIT hygiene — it's the prerequisite for JAX's `jax.grad` (automatic differentiation) and `jax.vmap` (automatic batching) to work correctly.

### In Code — Pure vs Impure Checklist

```python
import numpy as np

# ❌ IMPURE: reads global
# scale = 2.0
# @nb.njit
# def impure1(x):
#     return x * scale  # scale baked in at compile time

# ✅ PURE: takes scale as argument
# @nb.njit
# def pure1(x, scale):
#     return x * scale

# ❌ IMPURE: mutates external state
# results_list = []
# @nb.njit
# def impure2(x):
#     results_list.append(np.sum(x))  # side effect

# ✅ PURE: returns the result
# @nb.njit
# def pure2(x):
#     return np.sum(x)

# ❌ IMPURE: uses print for debugging
# @jax.jit
# def impure3(x):
#     print(f"x = {x}")  # only runs during tracing
#     return x ** 2

# ✅ PURE: uses jax.debug.print
# @jax.jit
# def pure3(x):
#     jax.debug.print("x = {}", x)  # runs every call
#     return x ** 2

# The pattern: if it reads or writes ANYTHING outside its arguments
# and return values, it's impure and will bite you with JIT.
```

### Common Misconceptions

- **"I'll just be careful with globals"** — You will forget. Write pure functions and the problem disappears.
- **"This is just a JAX/Numba thing"** — The same principle applies to PyTorch 2.0's `torch.compile`, TensorFlow's `tf.function`, and any other JIT system. Impure functions and JIT don't mix.
- **"Pure functions are harder to write"** — They're actually easier to test, debug, and reason about. The upfront cost of passing configuration as arguments pays off immediately.

---

For Numba-specific details, see [[Numba JIT Deep Dive]].
For JAX's tracing model that causes these issues, see [[JAX Tracing and Compilation]].
For the conceptual overview, see [[JIT Compilation Mechanics]].
