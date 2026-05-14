---
type: concept
title: "Python Decorators"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Closures]]"
  - "[[Python Custom Iterables]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Sklearn Pipelines and ColumnTransformer]]"
created: 2026-05-14
updated: 2026-05-14
last_section_added: "Tracing the Value Through the Chain, Stacking Parameterized Decorators"
---

# Python Decorators

> Decorators are functions that take a function and return a (usually modified) function. The `@decorator` syntax is syntactic sugar for `func = decorator(func)`. They are Python's primary mechanism for cross-cutting concerns — logging, timing, caching, validation, retry logic — without polluting the decorated function's body. This note builds from bare mechanics to parameterized decorators, stacking, class-based decorators, and real ML/data-science patterns.

---

## The Core Mechanic: Functions That Replace Functions

A decorator receives a function object, typically defines a wrapper inside, and returns the wrapper. After decoration, the original name is rebound to whatever the decorator returned.

```python
from typing import Callable, Any


def shout(func: Callable[..., str]) -> Callable[..., str]:
    """Decorator that uppercases the return value of a string-returning function."""
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result: str = func(*args, **kwargs)
        return result.upper()
    return wrapper


@shout
def greet(name: str) -> str:
    return f"hello, {name}"


# @shout is equivalent to: greet = shout(greet)
# greet is now wrapper. The original greet body still runs
# because wrapper calls func(*args, **kwargs).

print(greet("Reinaldo"))  # HELLO, REINALDO
print(greet("world"))     # HELLO, WORLD
```

### The "Throw the Function Away" Anti-Pattern

If the wrapper never calls `func()`, the original function body is unreachable:

```python
def decorator(func: Callable) -> Callable:
    def inner(text: str) -> str:
        return f"Passed message: {text}"
    return inner


@decorator
def my_func() -> str:
    return "Hello to Pythonistas"


# my_func IS inner now. The original body is gone.
print(my_func("Hello from Pythonistas"))  # Passed message: Hello from Pythonistas

# The original function object still exists (captured in the closure as `func`),
# but nobody calls it. This is almost always a bug in real code.
```

### The Typical Pattern: Wrap and Delegate

The useful pattern calls `func()` inside the wrapper, optionally transforming inputs or outputs:

```python
def reverse_result(func: Callable[..., str]) -> Callable[..., str]:
    """Decorator that reverses the string returned by func."""
    def wrapper(*args: Any, **kwargs: Any) -> str:
        result: str = func(*args, **kwargs)
        return result[::-1]
    return wrapper


@reverse_result
def say_hello() -> str:
    return "Hello!"


print(say_hello())  # !olleH
```

---

## When Decorators Execute: Definition Time, Not Call Time

A critical point that trips up many people: `@decorator` is an **immediate function call** that happens the instant Python finishes compiling the decorated function. The decorator body runs at definition time — not when the decorated function is eventually called.

```python
import time
from typing import Callable


def decorator(func: Callable) -> Callable:
    print("Decorator being run")  # ← runs at DEFINITION time
    def inner() -> str:
        return "Running inner"
    return inner


@decorator      # Python calls decorator(my_func) RIGHT HERE, RIGHT NOW
def my_func() -> str:
    return "Hello to Pythonistas"


time.sleep(5)   # 5-second pause AFTER the decorator has already run

print(my_func())  # Calls inner(), prints "Running inner"

# Output timeline:
#   "Decorator being run"   ← appears INSTANTLY (definition time)
#   ... 5-second pause ...
#   "Running inner"         ← appears after the pause (call time)
```

### The Execution Sequence, Line by Line

Python executes the module top to bottom. Here's what happens at each step:

```python
# Step 1: import time
#   → time module loaded. Nothing special.

# Step 2: def decorator(func): ...
#   → Python compiles the decorator function body and binds it to the
#     name `decorator`. Nothing inside runs yet.

# Step 3: @decorator applied to def my_func(): ...
#   → Python compiles my_func, creates a function object
#   → IMMEDIATELY calls decorator(my_func)
#   → Inside decorator: print("Decorator being run") executes NOW
#   → inner is defined, returned
#   → Python rebinds: my_func = inner
#   → "Decorator being run" is visible BEFORE time.sleep is reached

# Step 4: time.sleep(5)
#   → Program pauses. Decorator output already on screen.

# Step 5: print(my_func())
#   → Calls my_func, which IS inner, returns "Running inner"
```

### Why This Matters: Import-Time Side Effects

In a real module (not a notebook cell), decorators execute **at import time** — before any of your "main" code runs. This is how frameworks like Flask and Click work:

```python
# Flask uses this: @app.route runs at import time to register the URL
# before any HTTP request arrives.
#
# Simplified version of what Flask does internally:

from typing import Callable, Any


class SimpleRouter:
    """Minimal route registry to show import-time decorator execution."""

    def __init__(self) -> None:
        self.routes: dict[str, Callable] = {}

    def route(self, path: str) -> Callable:
        """Decorator factory: registers func under `path` at definition time."""
        def decorator(func: Callable) -> Callable:
            self.routes[path] = func   # ← registration happens NOW, at import time
            print(f"  Registered route: {path} → {func.__name__}")
            return func                # return func unchanged — no wrapping needed
        return decorator


app = SimpleRouter()


@app.route("/")
def index() -> str:
    return "Welcome home"


@app.route("/about")
def about() -> str:
    return "About page"


@app.route("/api/data")
def api_data() -> dict[str, Any]:
    return {"status": "ok"}


# By this point, ALL routes are registered — no function has been "called" yet.
# The decorator did its work at definition time.

print(f"\nAll registered routes: {list(app.routes.keys())}")
# ['/'] , ['/about'], ['/api/data']

# Simulate dispatching a request:
def dispatch(path: str) -> Any:
    handler = app.routes.get(path)
    if handler is None:
        return "404 Not Found"
    return handler()

print(dispatch("/"))          # Welcome home
print(dispatch("/about"))     # About page
print(dispatch("/missing"))   # 404 Not Found
```

### The Two Phases of a Decorator

```python
# ┌──────────────────────────────────────────────────────────┐
# │  PHASE 1: DEFINITION TIME (when @decorator is applied)   │
# │                                                          │
# │  - Runs ONCE per decorated function                      │
# │  - Happens at import time / module load                  │
# │  - The decorator body executes                           │
# │  - Side effects (registration, validation, print) fire   │
# │  - The wrapper function is CREATED but not called        │
# │                                                          │
# │  Use for: route registration, plugin registration,       │
# │  compile-time validation, schema building                │
# ├──────────────────────────────────────────────────────────┤
# │  PHASE 2: CALL TIME (when the decorated function runs)   │
# │                                                          │
# │  - Runs EVERY TIME the decorated function is called      │
# │  - The wrapper function executes                         │
# │  - Typically wraps/modifies/augments the original call   │
# │                                                          │
# │  Use for: logging, timing, caching, retries,             │
# │  input validation, access control                        │
# └──────────────────────────────────────────────────────────┘
```

### Proof: Decorator Runs Even If You Never Call the Function

```python
from typing import Callable

registry: list[str] = []


def register(func: Callable) -> Callable:
    """Decorator that registers functions. Runs at definition time."""
    registry.append(func.__name__)
    return func  # return func unchanged — pure registration, no wrapping


@register
def train_model() -> None:
    ...  # never called


@register
def evaluate_model() -> None:
    ...  # never called


@register
def export_results() -> None:
    ...  # never called


# None of the functions above were called, but the decorator
# ran three times during definition:
print(registry)  # ['train_model', 'evaluate_model', 'export_results']
```

---

## Preserving Metadata with `functools.wraps`

Without `functools.wraps`, the wrapper steals the decorated function's identity — `__name__`, `__doc__`, `__module__`, and `__annotations__` all point to the wrapper instead of the original. This breaks introspection, help(), and debugging.

```python
import functools


# BAD: without @wraps
def bad_decorator(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


@bad_decorator
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


print(add.__name__)  # wrapper  ← wrong!
print(add.__doc__)   # None     ← lost!


# GOOD: with @wraps
def good_decorator(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


@good_decorator
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


print(multiply.__name__)  # multiply  ← correct
print(multiply.__doc__)   # Multiply two integers.  ← preserved

# functools.wraps also preserves __wrapped__, so you can
# access the original function if needed:
print(multiply.__wrapped__(3, 4))  # 12  ← calls the original directly
```

**Rule of thumb:** always use `@functools.wraps(func)` on your wrapper. There is no good reason to skip it.

---

## Decorators with Arguments (Decorator Factories)

When you need to configure a decorator — `@repeat(n=3)` — you need a **three-layer nesting**: the outermost function takes the arguments, returns the actual decorator, which returns the wrapper.

```python
import functools
from typing import Callable, Any


def repeat(n: int = 2) -> Callable:
    """Decorator factory: call the decorated function n times, return last result."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = None
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


@repeat(n=3)
def say(message: str) -> str:
    print(message)
    return message


# @repeat(n=3) first calls repeat(n=3) → returns decorator
# Then decorator(say) → returns wrapper
# say is now wrapper

result = say("ping")
# Prints:
#   ping
#   ping
#   ping
print(f"Returned: {result}")  # Returned: ping
```

### The Minimal Case: Injecting a Value

The simplest useful parameterized decorator just injects a value into the output. The three-layer nesting is always the same — only the logic inside the wrapper changes:

```python
from typing import Callable


def decorator_with_argument(name: str) -> Callable:
    """Outermost layer: captures the configuration (name)."""
    def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
        """Middle layer: captures the function."""
        def wrapper(text: str) -> str:
            """Inner layer: does the actual work at call time."""
            return func(text) + f" {name}"
        return wrapper
    return decorator


@decorator_with_argument("Mason")
def greeting(text: str) -> str:
    return text[0].upper() + text[1:]


print(greeting("hola"))   # Hola Mason
print(greeting("hello"))  # Hello Mason


# Step by step:
#   1. decorator_with_argument("Mason") runs → returns decorator (name="Mason" captured)
#   2. decorator(greeting) runs → returns wrapper (func=greeting captured)
#   3. greeting is rebound to wrapper
#   4. greeting("hola") calls wrapper("hola") → calls func("hola") + " Mason"
```

### Stacking Parameterized Decorators

Parameterized decorators stack exactly like simple ones — each factory call returns a decorator, and the decorators apply bottom-up:

```python
from typing import Callable


def prefix(text: str) -> Callable:
    def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
        def wrapper(s: str) -> str:
            return text + func(s)
        return wrapper
    return decorator


def suffix(text: str) -> Callable:
    def decorator(func: Callable[[str], str]) -> Callable[[str], str]:
        def wrapper(s: str) -> str:
            return func(s) + text
        return wrapper
    return decorator


@prefix("Dear ")       # outermost
@suffix(", welcome!")   # innermost (applied first)
def formal_name(name: str) -> str:
    return name.title()


print(formal_name("reinaldo"))  # Dear Reinaldo, welcome!

# Equivalent to: prefix("Dear ")(suffix(", welcome!")(formal_name))
# 1. suffix(", welcome!") → decorator that appends ", welcome!"
# 2. decorator(formal_name) → wrapper_suffix
# 3. prefix("Dear ") → decorator that prepends "Dear "
# 4. decorator(wrapper_suffix) → wrapper_prefix
# 5. formal_name("reinaldo") calls wrapper_prefix → calls wrapper_suffix → calls original
```

### Real Example: Retry with Exponential Backoff

```python
import functools
import time
import random
from typing import Callable, Any


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry a function on failure with exponential backoff.

    Three-layer pattern:
      retry(max_attempts=3)  →  returns decorator
      decorator(func)        →  returns wrapper
      wrapper(*args)         →  runs func with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        sleep_time = backoff_base * (2 ** (attempt - 1))
                        print(f"  Attempt {attempt} failed: {e}. "
                              f"Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator


# Simulate a flaky function that fails randomly
call_count: int = 0


@retry(max_attempts=5, backoff_base=0.1, exceptions=(ValueError,))
def flaky_fetch(url: str) -> str:
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError(f"Connection timeout (attempt {call_count})")
    return f"Data from {url}"


call_count = 0
result = flaky_fetch("https://api.example.com/data")
print(result)  # Data from https://api.example.com/data
```

---

## Stacking Decorators

Decorators apply bottom-up (closest to the function first), but execute top-down (outermost wrapper runs first):

```python
import functools
from typing import Callable, Any


def bold(func: Callable[..., str]) -> Callable[..., str]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper


def italic(func: Callable[..., str]) -> Callable[..., str]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper


@bold       # applied second: bold(italic(greet))
@italic     # applied first:  italic(greet)
def greet(name: str) -> str:
    return f"Hello, {name}"


# Execution order: bold's wrapper calls italic's wrapper calls greet
print(greet("world"))  # <b><i>Hello, world</i></b>

# Equivalent to:
# greet = bold(italic(greet))
#
# Reading the stack: the TOP decorator is the OUTERMOST wrapper.
# bold wraps italic wraps greet.
```

### Tracing the Value Through the Chain

The hardest part of stacking is understanding how the return value flows. Each wrapper calls the next wrapper inward, receives its return value, transforms it, and passes it outward. Here's a markdown-formatting example that makes the flow explicit:

```python
from typing import Callable


def make_h1_md(func: Callable[[str], str]) -> Callable[[str], str]:
    """Outermost: wraps the result in a Markdown H1 heading."""
    def wrapper(text: str) -> str:
        inner_result = func(text)          # call the NEXT wrapper inward
        print(f"  make_h1_md received: {inner_result!r}")
        return "# " + inner_result
    return wrapper


def make_bold_md(func: Callable[[str], str]) -> Callable[[str], str]:
    """Innermost: wraps the result in Markdown bold."""
    def wrapper(text: str) -> str:
        inner_result = func(text)          # call the ORIGINAL function
        print(f"  make_bold_md received: {inner_result!r}")
        return "**" + inner_result + "**"
    return wrapper


@make_h1_md     # applied second: make_h1_md(make_bold_md(greeting))
@make_bold_md   # applied first:  make_bold_md(greeting)
def greeting(text: str) -> str:
    return text


# Call chain: make_h1_md's wrapper → make_bold_md's wrapper → greeting
print(greeting("hello"))
# Output:
#   make_bold_md received: 'hello'
#   make_h1_md received: '**hello**'
#   # **hello**
```

The value flows like a pipeline — outward through the wrappers:

```
greeting("hello")
    └─ make_h1_md's wrapper("hello")
         └─ calls func("hello") which is make_bold_md's wrapper
              └─ calls func("hello") which is the original greeting
                   └─ returns "hello"
              └─ returns "**hello**"
         └─ returns "# **hello**"
```

### Stacking Order Matters: Proof by Reversal

Swapping the decorator order changes the output because the outermost decorator transforms last:

```python
from typing import Callable


def make_h1_md(func: Callable[[str], str]) -> Callable[[str], str]:
    def wrapper(text: str) -> str:
        return "# " + func(text)
    return wrapper


def make_bold_md(func: Callable[[str], str]) -> Callable[[str], str]:
    def wrapper(text: str) -> str:
        return "**" + func(text) + "**"
    return wrapper


# Order A: bold is outermost
@make_bold_md
@make_h1_md
def greeting_a(text: str) -> str:
    return text


# Order B: h1 is outermost
@make_h1_md
@make_bold_md
def greeting_b(text: str) -> str:
    return text


print(greeting_a("hello"))  # **# hello**   ← bold wraps everything including "# "
print(greeting_b("hello"))  # # **hello**   ← h1 prefix sits outside the bold

# Manual equivalents to make it crystal clear:
assert greeting_a("hello") == make_bold_md(make_h1_md(lambda t: t))("hello")
assert greeting_b("hello") == make_h1_md(make_bold_md(lambda t: t))("hello")
```

### Stacking Three Decorators: A Data Pipeline

```python
import functools
from typing import Callable, Any

import numpy as np
from sklearn.datasets import load_iris


def log_shape(func: Callable) -> Callable:
    """Log the shape of the returned array."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        result = func(*args, **kwargs)
        print(f"  [{func.__name__}] output shape: {result.shape}")
        return result
    return wrapper


def clip_values(func: Callable) -> Callable:
    """Clip returned array to [0, 10]."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.clip(func(*args, **kwargs), 0.0, 10.0)
    return wrapper


def center_columns(func: Callable) -> Callable:
    """Mean-center each column of the returned 2D array."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> np.ndarray:
        result = func(*args, **kwargs)
        return result - result.mean(axis=0)
    return wrapper


# Stack: log_shape → clip_values → center_columns → get_iris_features
# Applied bottom-up, executed top-down.
@log_shape
@clip_values
@center_columns
def get_iris_features() -> np.ndarray:
    iris = load_iris()
    return iris.data


X = get_iris_features()
# center_columns runs first: centers the raw data
# clip_values runs second: clips the centered values to [0, 10]
# log_shape runs third: prints shape of the final result
print(f"Column means (should be clipped, not zero): {X.mean(axis=0).round(4)}")
```

---

## Class-Based Decorators

Any callable can be a decorator. A class with `__call__` works, and it's useful when you need to maintain state across calls.

```python
import functools
from typing import Callable, Any


class CountCalls:
    """Decorator that counts how many times a function has been called.

    Uses a class instead of a closure because the call count is an
    attribute we want to inspect from outside (same reasoning as the
    iterator-class-for-accessible-state pattern in
    [[Python Custom Iterables#Exercise 4]]).
    """

    def __init__(self, func: Callable) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.func(*args, **kwargs)


@CountCalls
def fibonacci(n: int) -> int:
    """Compute the nth Fibonacci number recursively."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


result = fibonacci(10)
print(f"fibonacci(10) = {result}")        # fibonacci(10) = 55
print(f"Total calls: {fibonacci.count}")  # Total calls: 177
print(f"Name preserved: {fibonacci.__name__}")  # Name preserved: fibonacci
```

### Class-Based Decorator Factory (with Arguments)

```python
import functools
import time
from typing import Callable, Any


class Timer:
    """Decorator that times function execution and stores the last duration.

    Parameterized: Timer(label="training") is a decorator factory.
    """

    def __init__(self, label: str = "Function") -> None:
        self.label = label
        self.last_duration: float = 0.0
        self._func: Callable | None = None

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            self.last_duration = time.perf_counter() - start
            print(f"  [{self.label}] {func.__name__} took "
                  f"{self.last_duration:.4f}s")
            return result
        # Store reference so we can access timer state from the wrapper
        wrapper.timer = self  # type: ignore[attr-defined]
        return wrapper


@Timer(label="sorting")
def sort_large_list(n: int) -> list[int]:
    """Generate and sort n random integers."""
    import random
    data = [random.randint(0, 1_000_000) for _ in range(n)]
    return sorted(data)


result = sort_large_list(500_000)
print(f"Sorted {len(result)} items")
# Access the timer's state through the attached reference:
print(f"Last duration: {sort_large_list.timer.last_duration:.4f}s")
```

---

## Real ML/Data Science Decorator Patterns

### Pattern 1: Dataset Caching Decorator

```python
import functools
import time
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine


def cache_dataset(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Cache a dataset-loading function's result.

    Unlike @functools.lru_cache, this works with DataFrames (which are
    unhashable and would fail with lru_cache). It uses a simple dict
    keyed by the string representation of the arguments.
    """
    cache: dict[str, pd.DataFrame] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
        key = f"{args}_{kwargs}"
        if key not in cache:
            print(f"  [cache miss] Loading {func.__name__}...")
            cache[key] = func(*args, **kwargs)
        else:
            print(f"  [cache hit] Returning cached {func.__name__}")
        return cache[key].copy()  # return a copy to prevent mutation of cached data
    wrapper.cache = cache  # type: ignore[attr-defined]
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
    return wrapper


@cache_dataset
def get_iris() -> pd.DataFrame:
    """Load the Iris dataset as a DataFrame."""
    data = load_iris(as_frame=True)
    df = data.frame  # type: ignore[union-attr]
    return df


@cache_dataset
def get_wine() -> pd.DataFrame:
    """Load the Wine dataset as a DataFrame."""
    data = load_wine(as_frame=True)
    df = data.frame  # type: ignore[union-attr]
    return df


# First call: cache miss
df1 = get_iris()
print(f"Shape: {df1.shape}")  # Shape: (150, 5)

# Second call: cache hit
df2 = get_iris()
print(f"Shape: {df2.shape}")  # Shape: (150, 5)

# Different function: cache miss
df3 = get_wine()
print(f"Wine shape: {df3.shape}")  # Wine shape: (178, 14)

# Mutation safety: modifying df1 doesn't affect the cache
df1.drop(columns=["target"], inplace=True)
df4 = get_iris()
assert "target" in df4.columns  # cache still has the original
print("Cache isolation verified")
```

### Pattern 2: Input Validation Decorator for NumPy Arrays

```python
import functools
from typing import Callable, Any

import numpy as np


def validate_array(
    *,
    ndim: int | None = None,
    min_samples: int | None = None,
    no_nan: bool = False,
    dtype: np.dtype | None = None,
) -> Callable:
    """Decorator factory: validate the first positional arg is a well-formed ndarray.

    Catches shape/type problems BEFORE they produce cryptic errors
    deep inside sklearn or numpy internals.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not args:
                raise TypeError(f"{func.__name__} requires at least one positional arg")
            X = args[0]
            if not isinstance(X, np.ndarray):
                raise TypeError(
                    f"{func.__name__} expected np.ndarray, got {type(X).__name__}"
                )
            if ndim is not None and X.ndim != ndim:
                raise ValueError(
                    f"{func.__name__} expected {ndim}D array, got {X.ndim}D"
                )
            if min_samples is not None and X.shape[0] < min_samples:
                raise ValueError(
                    f"{func.__name__} requires >= {min_samples} samples, "
                    f"got {X.shape[0]}"
                )
            if no_nan and np.isnan(X).any():
                raise ValueError(f"{func.__name__} received array with NaN values")
            if dtype is not None and X.dtype != dtype:
                raise TypeError(
                    f"{func.__name__} expected dtype {dtype}, got {X.dtype}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


@validate_array(ndim=2, min_samples=5, no_nan=True)
def compute_covariance(X: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix of X (samples × features)."""
    # Vectorized: mean-center then matrix multiply
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)


# Valid input — Iris features
from sklearn.datasets import load_iris
iris = load_iris()
X: np.ndarray = iris.data  # (150, 4) float64

cov = compute_covariance(X)
print(f"Covariance matrix shape: {cov.shape}")  # (4, 4)
print(f"Diagonal (variances): {np.diag(cov).round(3)}")

# Invalid inputs caught early with clear messages:
try:
    compute_covariance(X.flatten())  # 1D instead of 2D
except ValueError as e:
    print(f"Caught: {e}")

try:
    compute_covariance(X[:3])  # only 3 samples
except ValueError as e:
    print(f"Caught: {e}")

X_with_nan = X.copy()
X_with_nan[0, 0] = np.nan
try:
    compute_covariance(X_with_nan)
except ValueError as e:
    print(f"Caught: {e}")
```

### Pattern 3: Timing Decorator for Model Training Comparison

```python
import functools
import time
from typing import Callable, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Collect timings in a list for later comparison
timings: list[dict[str, Any]] = []


def timed(func: Callable) -> Callable:
    """Decorator that records function name, duration, and return value metadata."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timings.append({
            "function": func.__name__,
            "elapsed_s": round(elapsed, 4),
        })
        return result
    return wrapper


@timed
def evaluate_logistic(X: np.ndarray, y: np.ndarray) -> float:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


@timed
def evaluate_rf(X: np.ndarray, y: np.ndarray) -> float:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


@timed
def evaluate_gbt(X: np.ndarray, y: np.ndarray) -> float:
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    return float(scores.mean())


# Run all evaluations on the Wine dataset
data = load_wine()
X, y = data.data, data.target

acc_lr = evaluate_logistic(X, y)
acc_rf = evaluate_rf(X, y)
acc_gbt = evaluate_gbt(X, y)

# Build a comparison DataFrame from the collected timings
df_timings = pd.DataFrame(timings)
df_timings["accuracy"] = [acc_lr, acc_rf, acc_gbt]
df_timings = df_timings.sort_values("accuracy", ascending=False)
print(df_timings.to_string(index=False))
```

---

## `functools.lru_cache` — The Standard Library Decorator

Python ships with `functools.lru_cache`, a production-grade memoization decorator. It works on functions with **hashable** arguments only.

```python
import functools


@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Compute Fibonacci with automatic memoization."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


print(fibonacci(50))  # 12586269025  — instant, not 2^50 calls
print(fibonacci.cache_info())
# CacheInfo(hits=48, misses=51, maxsize=128, currsize=51)

# Clear the cache:
fibonacci.cache_clear()
print(fibonacci.cache_info())
# CacheInfo(hits=0, misses=0, maxsize=128, currsize=0)

# For unbounded cache, use maxsize=None:
@functools.lru_cache(maxsize=None)
def factorial(n: int) -> int:
    return 1 if n <= 1 else n * factorial(n - 1)

print(factorial(20))  # 2432902008176640000
```

---

## Common Misconceptions

- **"Decorators are only for functions."** — Classes can be decorated too. `@dataclass`, `@total_ordering`, and `@register` are all class decorators. Anything callable can decorate anything callable.

- **"The `@` syntax is magic."** — It's pure syntactic sugar. `@deco` before `def f():` is identical to `f = deco(f)` after the definition. No magic, no special bytecode.

- **"Decorators run every time the function is called."** — The decorator itself runs **once** at definition time. The *wrapper* it returns runs on every call. This distinction matters for setup-heavy decorators.

- **"I don't need `@functools.wraps`."** — You do. Without it, `help(func)`, `func.__name__`, and debugging tools all show the wrapper's identity. It costs nothing and saves headaches.

- **"Stacked decorators run bottom to top."** — They *apply* bottom to top (closest to `def` first), but they *execute* top to bottom (outermost wrapper is entered first). The distinction matters when decorators have side effects.

---

## Summary: The Three Patterns

```python
# ═══════════════════════════════════════════════════════════════
#  PATTERN 1: SIMPLE DECORATOR (no arguments)
# ═══════════════════════════════════════════════════════════════
#
#  def my_decorator(func):
#      @functools.wraps(func)
#      def wrapper(*args, **kwargs):
#          # ... before ...
#          result = func(*args, **kwargs)
#          # ... after ...
#          return result
#      return wrapper
#
#  @my_decorator
#  def f(): ...
#
# ═══════════════════════════════════════════════════════════════
#  PATTERN 2: DECORATOR FACTORY (with arguments)
# ═══════════════════════════════════════════════════════════════
#
#  def my_decorator(arg1, arg2):       ← factory (takes config)
#      def decorator(func):            ← actual decorator
#          @functools.wraps(func)
#          def wrapper(*args, **kwargs):  ← wrapper
#              ...
#          return wrapper
#      return decorator
#
#  @my_decorator(arg1=x, arg2=y)
#  def f(): ...
#
# ═══════════════════════════════════════════════════════════════
#  PATTERN 3: CLASS-BASED DECORATOR (stateful)
# ═══════════════════════════════════════════════════════════════
#
#  class MyDecorator:
#      def __init__(self, func):         ← receives func
#          functools.update_wrapper(self, func)
#          self.func = func
#          self.state = initial_value
#
#      def __call__(self, *args, **kwargs):  ← acts as wrapper
#          self.state = updated
#          return self.func(*args, **kwargs)
#
#  @MyDecorator
#  def f(): ...
#  f.state  ← accessible from outside
#
# ═══════════════════════════════════════════════════════════════
#  ALWAYS use @functools.wraps(func) or functools.update_wrapper.
#  There is no valid excuse to skip it.
# ═══════════════════════════════════════════════════════════════
```
