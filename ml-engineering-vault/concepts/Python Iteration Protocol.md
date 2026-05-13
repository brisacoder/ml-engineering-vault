---
type: concept
title: "Python Iteration Protocol"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Iteration Deep Dive]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Iterables]]"
  - "[[Python Generators]]"
  - "[[Python Decorators]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Iteration Protocol

> The iteration protocol is the contract that makes `for` loops, unpacking, comprehensions, and `*args` splat work in Python. Understanding the distinction between **iterables**, **iterators**, and **generators** is essential for writing correct, memory-efficient data pipelines.

---

## Iterable vs Iterator vs Generator

### Intuition

Think of it as a book analogy:

- **Iterable** — a bookshelf. You can grab a bookmark (iterator) from it whenever you want. Each bookmark starts at the beginning. You can get multiple independent bookmarks.
- **Iterator** — a bookmark in a book. It remembers where you are. It can only move forward. Once you reach the end, it's done.
- **Generator** — a bookmark in a book that *writes each page on demand*. The pages don't exist until you turn to them.

### Formal Definition

The protocol boils down to two dunder methods:

| Concept   | Must implement         | `iter()` returns       | `next()` works? | Reusable? |
|-----------|------------------------|------------------------|------------------|-----------|
| Iterable  | `__iter__`             | A **new** iterator     | No (not directly)| Yes       |
| Iterator  | `__iter__` + `__next__`| **itself** (`self`)    | Yes              | No        |
| Generator | (automatic from `yield`)| **itself** (`self`)   | Yes              | No        |

Key rule: **an iterator IS an iterable** (it has `__iter__`), but an iterable is NOT necessarily an iterator (a `list` has `__iter__` but not `__next__`).

### In Code

```python
# --- Iterable (list): iter() gives a NEW iterator each time ---
nums: list[int] = [1, 2, 3]

it1 = iter(nums)
it2 = iter(nums)
assert it1 is not it2  # two independent iterators

# --- Iterator: iter() returns ITSELF ---
it = iter(nums)
assert iter(it) is it  # same object — an iterator is its own iterable

# --- Generator: also its own iterator ---
def count_up(n: int):
    for i in range(n):
        yield i

gen = count_up(3)
assert iter(gen) is gen  # generator is its own iterator

# Consuming it
print(list(gen))  # [0, 1, 2]
print(list(gen))  # [] — exhausted, single-pass only!
```

### Common Misconceptions

- **"A list is an iterator"** — No. A list is an *iterable*. It produces iterators via `iter()`, but calling `next()` on a list raises `TypeError`.
- **"You can restart a generator"** — No. Generators are single-pass. Once exhausted, calling `next()` raises `StopIteration`. You must create a new generator object.
- **"`for` loops call `next()` on the iterable directly"** — No. Python first calls `iter()` on the object to get an iterator, *then* calls `next()` on that iterator.

---

## Lazy Evaluation and Generator Functions

### Intuition

A generator function's body **does not run** when you call the function. Python sees the `yield` keyword at definition time, and the call returns a generator object immediately — a suspended coroutine waiting to be poked with `next()`. The body only executes up to the first `yield` when you first call `next()`, then pauses again.

### In Code

```python
from typing import Generator

def my_enumerate(iterable) -> Generator[tuple[int, object], None, None]:
    """Naive enumerate that exposes the iterable/iterator distinction."""
    index: int = 0
    while True:
        item = next(iterable)  # BUG: calls next() directly — needs an iterator, not just an iterable
        yield (index, item)
        index += 1
```

#### Example 1 — Why it doesn't blow up (yet)

```python
result = my_enumerate(["a", "b", "c"])

# This passes! But it's deceptive.
assert iter(result) is iter(result)  # generator is its own iterator
```

This works because:

1. `my_enumerate(["a", "b", "c"])` sees `yield` and returns a generator object **immediately** — the function body hasn't executed a single line yet.
2. `iter(result) is iter(result)` just confirms a generator is its own iterator. No advancement happens.
3. The `next(iterable)` call inside the body — which would fail on a list — **has not been reached**.

#### Example 2 — Passing an iterator directly works fine

```python
inputs = iter(["a", "b", "c"])  # create an explicit iterator
result = my_enumerate(inputs)

assert next(result) == (0, "a")
assert list(inputs) == ["b", "c"]  # inputs was advanced by my_enumerate!
```

This works because `inputs` is already an iterator, so `next(iterable)` inside the function body is valid. Notice that `inputs` and `result` **share the same underlying iterator state** — consuming from one advances the other.

#### Example 3 — Actual blow-up when consuming with a list

```python
result = my_enumerate(["a", "b", "c"])

# Now we actually try to consume the generator:
try:
    next(result)
except TypeError as e:
    print(e)  # 'list' object is not an iterator
```

**This** is where the bug surfaces. The generator body finally runs, hits `next(iterable)` where `iterable` is a plain list, and `next()` requires an iterator.

#### The fix — do what the real `enumerate` does

```python
from typing import Generator, Iterable, TypeVar

T = TypeVar("T")

def my_enumerate(iterable: Iterable[T]) -> Generator[tuple[int, T], None, None]:
    """Correct version: convert iterable to iterator first."""
    iterator = iter(iterable)  # works on lists, sets, generators, anything iterable
    index: int = 0
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        yield (index, item)
        index += 1

# Now it works with any iterable
assert list(my_enumerate(["a", "b", "c"])) == [(0, "a"), (1, "b"), (2, "c")]
assert list(my_enumerate(iter("xyz"))) == [(0, "x"), (1, "y"), (2, "z")]
assert list(my_enumerate(range(3))) == [(0, 0), (1, 1), (2, 2)]
```

---

## Making Your Own Reusable Iterable

### Intuition

If a generator is single-pass, how do you build something that supports multiple `for` loops? Implement `__iter__` as a generator method — each call produces a fresh generator (iterator), but the data source stays put.

### In Code

```python
from typing import Iterator

class Repeat:
    """An iterable (not iterator) that yields a value n times.
    
    Supports multiple independent iterations.
    """

    def __init__(self, value: object, times: int) -> None:
        self.value = value
        self.times = times

    def __iter__(self) -> Iterator:
        for _ in range(self.times):
            yield self.value

r = Repeat("hello", 3)

# Two independent loops — works because __iter__ returns a NEW generator each time
first_pass = list(r)   # ['hello', 'hello', 'hello']
second_pass = list(r)  # ['hello', 'hello', 'hello'] — NOT exhausted

# Confirm it's a proper iterable, not an iterator
assert iter(r) is not iter(r)  # different iterator objects each time
```

### Common Misconceptions

- **"Implementing `__iter__` makes you an iterator"** — Only if `__iter__` returns `self` *and* you also have `__next__`. If `__iter__` returns a new object (like a generator), you're an iterable that *produces* iterators.

---

## Quick Reference: iter() Identity Test

```python
def classify_iteration(obj: object) -> str:
    """Classify an object by its iteration behavior."""
    if not hasattr(obj, "__iter__"):
        return "not iterable"
    if iter(obj) is obj:
        return "iterator (single-pass)"
    return "iterable (multi-pass)"

# Examples
print(classify_iteration([1, 2, 3]))          # iterable (multi-pass)
print(classify_iteration(iter([1, 2, 3])))     # iterator (single-pass)
print(classify_iteration(x for x in [1, 2]))   # iterator (single-pass)
print(classify_iteration({1: "a", 2: "b"}))    # iterable (multi-pass)
print(classify_iteration(range(10)))            # iterable (multi-pass)
print(classify_iteration(42))                   # not iterable
```

The `iter(obj) is obj` test is the canonical way to check whether something is an iterator: iterators return themselves from `__iter__`, iterables return a fresh iterator.
