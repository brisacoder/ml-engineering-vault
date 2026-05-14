---
type: concept
title: "Python Closures"
libs:
  - general
tags:
  - concept/architecture
  - concept/functional-programming
related:
  - "[[Python Decorators]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Iterables]]"
  - "[[Python Iteration Protocol]]"
created: 2026-05-14
updated: 2026-05-14
---

# Python Closures

> A **closure** is a function object that remembers values from its enclosing lexical scope even after that scope has finished executing. In Python, a closure is created whenever a nested function references a variable from an outer (non-global) function. The inner function, together with its captured environment, forms the closure. Closures are the mechanism underlying [[Python Decorators]], callback patterns, factory functions, and lazy evaluation strategies used throughout the Python ML ecosystem.

---

## How Closures Work: The LEGB Rule and Cell Objects

Python resolves names using the **LEGB** rule: Local → Enclosing → Global → Built-in. When a nested function references a variable from an enclosing scope, Python creates a **cell object** that both the outer and inner functions share. This cell holds a reference to the captured value, which is why the inner function can access it even after the outer function returns.

```python
def outer() -> callable:
    x: int = 10  # lives in outer's local scope

    def inner() -> int:
        return x  # captures x via a cell object

    return inner


closure = outer()

# The function remembers x=10 even though outer() has returned
print(closure())        # 10

# Inspect the closure's cell objects
print(closure.__closure__)                        # (<cell at 0x...>,)
print(closure.__closure__[0].cell_contents)       # 10

# The free variables that inner captures
print(closure.__code__.co_freevars)               # ('x',)
```

---

## The Running Sum: Your First Closure with State

This is the classic pattern — a closure that accumulates state across calls without using a class or a global variable.

```python
def calc_sum() -> callable:
    total: int = 0

    def add_num(num: int) -> int:
        nonlocal total       # required to rebind total in enclosing scope
        total += num
        return total

    return add_num


running_sum = calc_sum()
print(running_sum(1))    # 1
print(running_sum(1))    # 2
print(running_sum(4))    # 6

# Each call to calc_sum() creates an independent closure
another_sum = calc_sum()
print(another_sum(100))  # 100  — completely independent state
print(running_sum(0))    # 6    — original is unaffected
```

---

## `nonlocal` vs Read-Only Capture

A closure can **read** an enclosing variable freely, but **rebinding** it (assigning with `=`, `+=`, etc.) requires the `nonlocal` declaration. Without it, Python treats the name as a new local variable and raises `UnboundLocalError`.

```python
# --- Read-only capture: no nonlocal needed ---
def make_greeter(greeting: str) -> callable:
    def greet(name: str) -> str:
        return f"{greeting}, {name}!"  # reading greeting is fine
    return greet


hello = make_greeter("Hello")
hola = make_greeter("Hola")
print(hello("Reinaldo"))  # Hello, Reinaldo!
print(hola("Reinaldo"))   # Hola, Reinaldo!


# --- Rebinding requires nonlocal ---
def make_counter() -> callable:
    count: int = 0

    def increment() -> int:
        nonlocal count   # remove this line and get UnboundLocalError
        count += 1
        return count

    return increment


counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

---

## The Late-Binding Gotcha (and How to Fix It)

This is by far the most common closure pitfall. Closures capture **references**, not **values**. When you create closures inside a loop, they all share the same variable — and by the time you call them, that variable holds the final loop value.

```python
# --- THE BUG ---
funcs: list[callable] = []
for i in range(5):
    funcs.append(lambda: i)

# All five lambdas see the SAME i, which is now 4
print([f() for f in funcs])  # [4, 4, 4, 4, 4]  — not what you wanted!


# --- FIX 1: Default argument captures the value at definition time ---
funcs_fixed: list[callable] = []
for i in range(5):
    funcs_fixed.append(lambda i=i: i)  # default arg snapshots current i

print([f() for f in funcs_fixed])  # [0, 1, 2, 3, 4]


# --- FIX 2: Factory function creates a new scope per iteration ---
def make_func(val: int) -> callable:
    def func() -> int:
        return val
    return func

funcs_factory = [make_func(i) for i in range(5)]
print([f() for f in funcs_factory])  # [0, 1, 2, 3, 4]


# --- FIX 3: functools.partial ---
from functools import partial

def identity(x: int) -> int:
    return x

funcs_partial = [partial(identity, i) for i in range(5)]
print([f() for f in funcs_partial])  # [0, 1, 2, 3, 4]
```

---

## Closure Factories: Configurable Functions

Closures shine as **function factories** — functions that produce customized functions. This is functionally equivalent to partial application.

```python
from typing import Callable
import numpy as np
from sklearn.datasets import load_iris

# --- Example 1: A configurable distance metric ---
def make_minkowski(p: int) -> Callable[[np.ndarray, np.ndarray], float]:
    """Factory that returns a Minkowski distance function for a given p."""
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.sum(np.abs(a - b) ** p) ** (1.0 / p)
    return distance


manhattan = make_minkowski(p=1)
euclidean = make_minkowski(p=2)

iris = load_iris()
X = iris.data
a, b = X[0], X[1]

print(f"Manhattan distance: {manhattan(a, b):.4f}")   # p=1
print(f"Euclidean distance: {euclidean(a, b):.4f}")    # p=2


# --- Example 2: Threshold classifier factory ---
def make_threshold_classifier(threshold: float) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a binary classifier that splits at the given threshold."""
    def classify(scores: np.ndarray) -> np.ndarray:
        return (scores >= threshold).astype(int)
    return classify


conservative = make_threshold_classifier(0.8)
aggressive = make_threshold_classifier(0.3)

scores = np.array([0.1, 0.4, 0.6, 0.85, 0.95])
print(f"Conservative: {conservative(scores)}")  # [0 0 0 1 1]
print(f"Aggressive:   {aggressive(scores)}")    # [0 1 1 1 1]
```

---

## Closures as Lightweight State Machines

When you need state but a full class feels like overkill, closures offer a clean middle ground.

```python
from typing import Callable
import numpy as np
from sklearn.datasets import load_iris


def make_online_stats() -> dict[str, Callable]:
    """
    Welford's online algorithm via closure.
    Tracks running mean and variance without storing all values.
    """
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(x: float) -> None:
        nonlocal n, mean, m2
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2

    def get_mean() -> float:
        return mean

    def get_variance() -> float:
        return m2 / n if n > 0 else 0.0

    def get_count() -> int:
        return n

    return {"update": update, "mean": get_mean, "var": get_variance, "count": get_count}


# Stream Iris sepal lengths through our closure-based stats tracker
iris = load_iris()
sepal_lengths = iris.data[:, 0]

stats = make_online_stats()
for val in sepal_lengths:
    stats["update"](val)

print(f"n    = {stats['count']()}")
print(f"mean = {stats['mean']():.4f}")
print(f"var  = {stats['var']():.4f}")

# Verify against numpy
print(f"\nnp.mean = {np.mean(sepal_lengths):.4f}")
print(f"np.var  = {np.var(sepal_lengths):.4f}")
```

---

## Closures with Multiple Returned Functions (Encapsulation)

A single outer function can return multiple inner functions that all share the same enclosed state — a poor man's object, but sometimes exactly right.

```python
def make_accumulator(initial: float = 0.0) -> tuple[callable, callable, callable]:
    """Returns (deposit, withdraw, balance) — all sharing the same state."""
    balance: float = initial

    def deposit(amount: float) -> float:
        nonlocal balance
        balance += amount
        return balance

    def withdraw(amount: float) -> float:
        nonlocal balance
        if amount > balance:
            raise ValueError(f"Insufficient funds: {balance:.2f} < {amount:.2f}")
        balance -= amount
        return balance

    def get_balance() -> float:
        return balance

    return deposit, withdraw, get_balance


deposit, withdraw, balance = make_accumulator(100.0)
print(balance())        # 100.0
print(deposit(50.0))    # 150.0
print(withdraw(30.0))   # 120.0
print(balance())        # 120.0
```

---

## Closures as Callbacks in ML Workflows

Closures are natural fits for callbacks — functions that are invoked at certain points during training or processing.

```python
from typing import Callable
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


def make_score_logger() -> tuple[Callable, Callable]:
    """
    Returns (log_score, get_history) — a callback pair for tracking
    cross-validation scores across experiments.
    """
    history: list[dict] = []

    def log_score(experiment_name: str, scores: np.ndarray) -> None:
        history.append({
            "name": experiment_name,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist(),
        })
        print(f"  [{experiment_name}] mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")

    def get_history() -> list[dict]:
        return history.copy()

    return log_score, get_history


iris = load_iris()
X, y = iris.data, iris.target

log_score, get_history = make_score_logger()

# Run multiple experiments, closure accumulates results
for alpha in [0.0001, 0.001, 0.01]:
    clf = SGDClassifier(alpha=alpha, random_state=42, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    log_score(f"SGD(alpha={alpha})", scores)

print(f"\nTotal experiments logged: {len(get_history())}")
best = max(get_history(), key=lambda h: h["mean"])
print(f"Best: {best['name']} → {best['mean']:.4f}")
```

---

## Closure vs Class: When to Use Which

Closures and classes both encapsulate state + behavior. The choice is often a matter of complexity.

```python
import numpy as np
from sklearn.datasets import load_iris

# --- Closure version: simple, one behavior ---
def make_ema(alpha: float = 0.3) -> callable:
    """Exponential moving average via closure."""
    ema: float | None = None

    def update(value: float) -> float:
        nonlocal ema
        if ema is None:
            ema = value
        else:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    return update


# --- Class version: richer interface ---
class EMA:
    """Exponential moving average with reset and history."""

    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.ema: float | None = None
        self.history: list[float] = []

    def update(self, value: float) -> float:
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        self.history.append(self.ema)
        return self.ema

    def reset(self) -> None:
        self.ema = None
        self.history.clear()


# Compare both on Iris sepal widths
iris = load_iris()
sepal_widths = iris.data[:, 1]

closure_ema = make_ema(alpha=0.1)
class_ema = EMA(alpha=0.1)

closure_results = np.array([closure_ema(v) for v in sepal_widths])
class_results = np.array([class_ema.update(v) for v in sepal_widths])

# They produce identical results
print(f"Max difference: {np.max(np.abs(closure_results - class_results))}")  # 0.0

# Rule of thumb:
# - 1-2 behaviors, no need for repr/comparison/serialization → closure
# - Multiple methods, inheritance, or rich interface → class
```

---

## Closures and Decorators: The Connection

Every [[Python Decorators|decorator]] is built on a closure. The decorator function is the outer scope, and the wrapper function is the closure that captures the original function.

```python
import time
from typing import Callable, Any
from functools import wraps
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def timer(func: Callable) -> Callable:
    """A decorator is a closure that captures `func`."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)  # func is captured from enclosing scope
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper  # wrapper closes over func


@timer
def train_and_score(X: np.ndarray, y: np.ndarray, n_trees: int) -> float:
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    return float(np.mean(scores))


iris = load_iris()
score = train_and_score(iris.data, iris.target, n_trees=100)
print(f"Mean accuracy: {score:.4f}")
```

---

## Parameterized Closures (Nested Three-Deep)

When you need a factory that produces configured closures, you end up with three levels of nesting. This is exactly what parameterized [[Python Decorators|decorators]] do.

```python
from typing import Callable, Any
from functools import wraps
import numpy as np


def clip_output(low: float, high: float) -> Callable:
    """
    Three levels:
      1. clip_output(low, high) — the config layer
      2. decorator(func) — captures func AND low/high
      3. wrapper(*args) — the actual closure that runs
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
            result = func(*args, **kwargs)
            return np.clip(result, low, high)  # low/high from outermost scope
        return wrapper
    return decorator


@clip_output(low=0.0, high=1.0)
def predict_raw(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return X @ weights


np.random.seed(42)
X = np.random.randn(5, 3)
w = np.random.randn(3)

raw = X @ w
clipped = predict_raw(X, w)

print(f"Raw predictions:     {raw}")
print(f"Clipped predictions: {clipped}")
print(f"All in [0,1]: {np.all((clipped >= 0) & (clipped <= 1))}")
```

---

## Closures for Memoization

Closures naturally hold a cache dictionary in their enclosing scope — the basis of memoization.

```python
from typing import Callable, Any


def memoize(func: Callable) -> Callable:
    """Generic memoization via closure-held cache."""
    cache: dict[tuple, Any] = {}

    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    # Expose cache for inspection
    wrapper.cache = cache
    return wrapper


@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(30))                    # 832040 — instant
print(f"Cache size: {len(fibonacci.cache)}")  # 31 entries

# Works for any pure function
@memoize
def expensive_norm(x: float, y: float, z: float) -> float:
    return (x**2 + y**2 + z**2) ** 0.5

print(expensive_norm(3.0, 4.0, 0.0))   # 5.0
print(expensive_norm(3.0, 4.0, 0.0))   # 5.0 — cache hit
print(f"Norm cache: {expensive_norm.cache}")
```

> **Note:** For production use, prefer `functools.lru_cache` or `functools.cache` — they handle maxsize, thread safety, and cache statistics. But understanding the closure-based pattern makes those tools transparent.

---

## Closures for Lazy Evaluation and Deferred Computation

Closures let you capture expensive computation parameters and defer execution until the result is actually needed.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


def make_lazy_evaluation(
    model_class: type,
    X: np.ndarray,
    y: np.ndarray,
    **model_kwargs,
) -> callable:
    """Capture everything needed for evaluation, but don't run it yet."""
    result: dict | None = None

    def evaluate() -> dict:
        nonlocal result
        if result is None:  # only compute once
            clf = model_class(**model_kwargs)
            scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
            result = {
                "model": model_class.__name__,
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
            }
        return result

    return evaluate


iris = load_iris()
X, y = iris.data, iris.target

# Define experiments lazily — nothing runs yet
experiments = [
    make_lazy_evaluation(RandomForestClassifier, X, y, n_estimators=50, random_state=42),
    make_lazy_evaluation(GradientBoostingClassifier, X, y, n_estimators=50, random_state=42),
]

print("Experiments defined, nothing computed yet.")

# Only evaluate when needed
for exp in experiments:
    result = exp()
    print(f"  {result['model']}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")

# Second call returns cached result — no recomputation
print(f"\nCached: {experiments[0]()['model']}")
```

---

## Inspecting Closures at Runtime

Python exposes closure internals through dunder attributes — useful for debugging and understanding what got captured.

```python
def make_scaler(mean: float, std: float) -> callable:
    def scale(x: float) -> float:
        return (x - mean) / std
    return scale


scaler = make_scaler(mean=5.84, std=0.83)

# What did the closure capture?
print(f"Is it a closure? {scaler.__closure__ is not None}")  # True
print(f"Free variables:  {scaler.__code__.co_freevars}")      # ('mean', 'std')

# Inspect captured values
for var_name, cell in zip(scaler.__code__.co_freevars, scaler.__closure__):
    print(f"  {var_name} = {cell.cell_contents}")
# mean = 5.84
# std = 0.83

# Regular (non-closure) functions have __closure__ = None
def plain_func(x: float) -> float:
    return x * 2

print(f"\nplain_func closure: {plain_func.__closure__}")  # None
```

---

## Closures in Data Pipelines: Composable Transforms

Closures compose naturally into data transformation pipelines — each transform captures its configuration.

```python
import numpy as np
from typing import Callable
from sklearn.datasets import load_iris


Transform = Callable[[np.ndarray], np.ndarray]


def make_standardizer(axis: int = 0) -> Transform:
    """Closure captures axis; fit stats captured on first call."""
    stats: dict = {}

    def transform(X: np.ndarray) -> np.ndarray:
        if "mean" not in stats:
            stats["mean"] = np.mean(X, axis=axis, keepdims=True)
            stats["std"] = np.std(X, axis=axis, keepdims=True)
        return (X - stats["mean"]) / (stats["std"] + 1e-8)

    return transform


def make_clipper(low: float, high: float) -> Transform:
    def transform(X: np.ndarray) -> np.ndarray:
        return np.clip(X, low, high)
    return transform


def make_power_transform(power: float) -> Transform:
    def transform(X: np.ndarray) -> np.ndarray:
        return np.sign(X) * np.abs(X) ** power
    return transform


def compose(*transforms: Transform) -> Transform:
    """Compose closures left-to-right into a single pipeline."""
    def pipeline(X: np.ndarray) -> np.ndarray:
        result = X
        for t in transforms:
            result = t(result)
        return result
    return pipeline


iris = load_iris()
X = iris.data

# Build a pipeline from closure-based transforms
preprocess = compose(
    make_standardizer(),
    make_clipper(-3.0, 3.0),
    make_power_transform(0.5),
)

X_processed = preprocess(X)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {X_processed.shape}")
print(f"Output range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
print(f"Output mean (per feature): {np.mean(X_processed, axis=0)}")
```

---

## Summary: Closure Mental Model

| Aspect | Detail |
|---|---|
| **What** | A nested function + its captured enclosing variables |
| **How** | Python stores refs in `__closure__` cell objects |
| **Read vs Write** | Reading is free; rebinding needs `nonlocal` |
| **Gotcha** | Late binding in loops — all closures share one variable |
| **vs Class** | Closures for 1-2 behaviors; classes for richer interfaces |
| **Key uses** | Factories, callbacks, [[Python Decorators\|decorators]], memoization, lazy eval, pipelines |
