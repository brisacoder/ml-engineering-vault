---
type: concept
title: "Python Iterator Gotchas"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iteration Deep Dive]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Comprehensions]]"
  - "[[Python Custom Iterables]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Iterator Gotchas

> A collection of traps, surprising behaviors, and subtle bugs that arise from the iterable/iterator/generator distinction. Each gotcha includes a failing example, an explanation of *why* it fails, and the fix.

---

## Gotcha 1: Calling `next()` on an Iterable

### The Trap

```python
numbers: list[int] = [1, 2, 3, 4, 5]

# This looks reasonable — "give me the next item" — but it explodes
next(numbers)
# TypeError: 'list' object is not an iterator
```

### Why It Fails

`next()` requires an **iterator** — an object with `__next__`. A list only has `__iter__` (which *produces* an iterator), not `__next__` itself. A list is an iterable, not an iterator.

```python
# Proof: list has __iter__ but NOT __next__
print(hasattr(numbers, "__iter__"))   # True  — it's iterable
print(hasattr(numbers, "__next__"))   # False — it's NOT an iterator
```

### The Fix

```python
# Wrap in iter() first to get an iterator
it = iter(numbers)  # <list_iterator object>

print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
# it remembers its position, just like a generator
```

---

## Gotcha 2: Exhausted Iterators Silently Produce Empty Results

### The Trap

```python
def squares(n: int):
    """Yield squares of 0..n-1."""
    for i in range(n):
        yield i ** 2

gen = squares(4)

first_pass = list(gen)   # [0, 1, 4, 9] — looks great
second_pass = list(gen)  # [] — wait, where did everything go?!

print(f"{first_pass=}")   # [0, 1, 4, 9]
print(f"{second_pass=}")  # []  ← empty!
```

### Why It Fails

Generators (and all iterators) are **single-pass**. Once exhausted, they stay exhausted. Calling `list()` on an exhausted generator doesn't raise an error — it silently returns an empty list. This is by design: `list()` calls `iter()` → gets the same exhausted generator → calls `next()` → immediately gets `StopIteration` → returns `[]`.

```python
gen = squares(2)
print(next(gen))  # 0
print(next(gen))  # 1

# Now it's exhausted:
try:
    next(gen)
except StopIteration:
    print("Generator is done — permanently")  # this prints

# iter() on an exhausted generator returns the SAME exhausted object
assert iter(gen) is gen  # True — still the same dead generator

# Anything that consumes it gets nothing
print(list(gen))    # []
print(tuple(gen))   # ()
print(sum(gen))     # 0  ← this one's particularly sneaky
```

### The Fix

```python
# Option 1: Re-create the generator each time you need it
first_pass = list(squares(4))
second_pass = list(squares(4))  # fresh generator, fresh data

# Option 2: Materialize once if the data fits in memory
cached: list[int] = list(squares(4))
first_pass = cached[:]   # copy or slice as needed
second_pass = cached[:]

# Option 3: Make a reusable iterable class (see Python Iteration Protocol note)
```

---

## Gotcha 3: Generator Body Is Deferred — Bugs Hide Until Consumption

### The Trap

```python
def my_enumerate(iterable):
    """Buggy: calls next() on iterable directly."""
    index: int = 0
    while True:
        item = next(iterable)  # BUG: iterable might not be an iterator!
        yield (index, item)
        index += 1

# This passes — no error at all!
result = my_enumerate(["a", "b", "c"])
assert iter(result) is iter(result)  # ✓ generator is its own iterator
print("No error yet!")               # prints fine
```

### Why It Doesn't Fail (Yet)

When Python encounters `yield` in a function body, calling that function **does not execute any of the body**. It returns a suspended generator object. The bug (`next()` on a list) lives inside the body, but the body hasn't run. The `assert` just checks generator identity — no advancement.

```python
# The bug only surfaces when you actually CONSUME the generator:
result = my_enumerate(["a", "b", "c"])

try:
    first_item = next(result)  # NOW the body runs, NOW it blows up
except TypeError as e:
    print(f"Bug found: {e}")
    # Bug found: 'list' object is not an iterator
```

### Why This Is Dangerous

Deferred execution means **you can create a broken generator, pass it around, store it in a data structure, even return it from a function — and the error won't appear until someone eventually consumes it**, potentially far from where the bug was introduced.

```python
def build_pipeline(data: list[str]) -> list:
    """Looks fine at creation time..."""
    step1 = my_enumerate(data)          # no error — body hasn't run
    step2 = (f"{i}: {v}" for i, v in step1)  # no error — also deferred
    return step2                        # returns a generator, no error

pipeline = build_pipeline(["x", "y", "z"])
# Everything above succeeded. The bug is hiding.

# Minutes/lines later, someone tries to use it:
try:
    list(pipeline)  # BOOM — TypeError deep in the chain
except TypeError as e:
    print(f"Error surfaces here, but was caused in my_enumerate: {e}")
```

### The Fix

```python
from typing import Iterable, Generator, TypeVar

T = TypeVar("T")

def my_enumerate_fixed(iterable: Iterable[T]) -> Generator[tuple[int, T], None, None]:
    """Always convert to iterator first — works with ANY iterable."""
    iterator = iter(iterable)  # iter() on a list → list_iterator; on a generator → itself
    index: int = 0
    for item in iterator:      # let for-loop handle StopIteration
        yield (index, item)
        index += 1
```

---

## Gotcha 4: Multiple Iterators Share State When You Don't Expect It

### The Trap

```python
inputs = iter(["a", "b", "c", "d"])  # one iterator from the list

# Pass the same iterator to two consumers
from itertools import islice

first_two = list(islice(inputs, 2))   # ['a', 'b']
rest = list(inputs)                    # ['c', 'd'] ← not ['a', 'b', 'c', 'd']!

print(f"{first_two=}")  # ['a', 'b']
print(f"{rest=}")        # ['c', 'd'] — inputs was advanced by islice!
```

### Why It Happens

An iterator is a **single stateful cursor**. When you pass it to `islice`, that function calls `next()` on it, advancing the shared cursor. Anything else reading from the same iterator sees only what's left.

```python
# More dramatic example: the same iterator in a zip
numbers = iter(range(10))

# zip will pull from the SAME iterator for both arguments!
pairs = list(zip(numbers, numbers))
print(pairs)
# [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
# Each pair consumes TWO elements because both slots pull from the same cursor.

# Compare with a list (iterable, not iterator):
pairs_from_list = list(zip(range(10), range(10)))
print(pairs_from_list)
# [(0, 0), (1, 1), (2, 2), ..., (9, 9)]
# zip calls iter() on each argument → gets TWO independent iterators
```

### The Fix

```python
# If you need independent iteration, don't share the iterator.
# Use itertools.tee for independent copies (with memory trade-offs):
from itertools import tee

original = iter(range(5))
copy1, copy2 = tee(original, 2)  # two independent iterators

print(list(copy1))  # [0, 1, 2, 3, 4]
print(list(copy2))  # [0, 1, 2, 3, 4] — independent copy

# WARNING: tee buffers internally. If one copy advances far ahead
# of the other, tee stores all the intermediate values in memory.
# At that point you might as well have used list().
```

---

## Gotcha 5: Generator Expressions Bind Late — The Loop Variable Trap

### The Trap

```python
# Create generators that each reference the loop variable
generators = []
for i in range(3):
    generators.append(x + i for x in [10, 20])

# You might expect: [10, 20], [11, 21], [12, 22]
# But you get:
for gen in generators:
    print(list(gen))
# [12, 22]
# [12, 22]
# [12, 22]  ← all use i=2!
```

### Why It Happens

Generator expressions are **lazy** — they don't evaluate when created. The variable `i` inside each generator expression is a **closure over the loop variable**, not a snapshot of its value. By the time you consume the generators, the loop has finished and `i == 2` for all of them.

```python
# Same thing happens with lambda, list comprehensions in 2.x, etc.
# This is a general late-binding closure issue, amplified by laziness.

# Even this single generator shows the problem:
i = 100
gen = (x + i for x in [1, 2])
i = 999
print(list(gen))  # [1000, 1001] — uses i=999, not i=100
```

### The Fix

```python
# Fix 1: Capture the value with a default argument
generators_fixed = []
for i in range(3):
    generators_fixed.append(x + captured_i for x in [10, 20] for captured_i in [i])
    # Awkward — better approach below

# Fix 2: Use a factory function to create a new scope
def make_gen(i: int):
    """Each call creates a new closure with its own i."""
    return (x + i for x in [10, 20])

generators_fixed = [make_gen(i) for i in range(3)]
for gen in generators_fixed:
    print(list(gen))
# [10, 20]
# [11, 21]
# [12, 22]  ← correct!
```

---

## Gotcha 6: `in` Operator Partially Consumes Iterators

### The Trap

```python
gen = iter(range(10))

# Check if 3 is "in" the generator
print(3 in gen)   # True

# What's left?
print(list(gen))  # [4, 5, 6, 7, 8, 9] — 0, 1, 2, 3 are gone!
```

### Why It Happens

The `in` operator on an iterator calls `next()` repeatedly until it either finds the value (returns `True`) or hits `StopIteration` (returns `False`). It consumes all elements up to and including the match.

```python
gen = iter(range(5))

# Searching for something that doesn't exist consumes EVERYTHING
print(99 in gen)   # False
print(list(gen))   # [] — completely exhausted by the failed search
```

### The Fix

```python
# If you need to check membership AND keep the data, materialize first:
data: list[int] = list(range(10))
print(3 in data)    # True — list.__contains__ doesn't consume anything
print(list(data))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] — still intact
```

---

## Gotcha 7: `sum()`, `max()`, `min()` Silently Accept Exhausted Iterators

### The Trap

```python
gen = (x ** 2 for x in range(5))  # 0, 1, 4, 9, 16

total = sum(gen)       # 30 — correct
maximum = max(gen)     # ValueError: max() arg is an empty sequence
```

Wait, that one at least gives an error. But `sum` is sneakier:

```python
gen = (x ** 2 for x in range(5))

first_sum = sum(gen)   # 30 — correct
second_sum = sum(gen)  # 0  ← silently wrong! sum([]) == 0
```

### Why It Happens

`sum()` has a default start value of `0`. Summing an empty (exhausted) iterator just returns `0`. No error, no warning — just a wrong answer.

```python
# These all silently give "zero-like" results on exhausted iterators:
gen = iter([])
print(sum(gen))            # 0
print(list(gen))           # []
print(tuple(gen))          # ()
print("".join(gen))        # ""
print(all(gen))            # True  ← vacuous truth!
print(any(gen))            # False
print(sorted(gen))         # []
print(dict(gen))           # {}
```

### The Fix

```python
# Be conscious of generator single-pass nature.
# If you need the data more than once, materialize it:
values: list[int] = [x ** 2 for x in range(5)]  # list comp, not generator
total = sum(values)     # 30
maximum = max(values)   # 16 — works because list is multi-pass
```

---

## Gotcha 8: `dict.keys()`, `dict.values()` Are Views, Not Lists — But They ARE Re-iterable

### The Trap (or Non-Trap)

```python
d = {"a": 1, "b": 2, "c": 3}

keys = d.keys()  # dict_keys object — is this an iterator?

# Test: is it its own iterator?
print(iter(keys) is keys)  # False — it's an ITERABLE, not an iterator!

# This means you CAN iterate multiple times:
print(list(keys))  # ['a', 'b', 'c']
print(list(keys))  # ['a', 'b', 'c'] — still works!

# But they ARE live views — they reflect mutations:
d["d"] = 4
print(list(keys))  # ['a', 'b', 'c', 'd'] — updated!
```

### Why This Matters

People often assume `dict.keys()` returns a list (Python 2 behavior) or a generator/iterator. It's neither — it's a **view** (a reusable iterable that reflects the current dict state). This is actually *less* error-prone than generators.

```python
# Compare: dict.keys() (reusable view) vs iter(dict) (single-pass iterator)
keys_view = d.keys()
keys_iter = iter(d)

print(iter(keys_view) is keys_view)  # False — view is a reusable iterable
print(iter(keys_iter) is keys_iter)  # True  — this is a single-pass iterator

# The iterator gets exhausted; the view doesn't
list(keys_iter)  # ['a', 'b', 'c', 'd']
list(keys_iter)  # [] — exhausted
list(keys_view)  # ['a', 'b', 'c', 'd'] — still works
list(keys_view)  # ['a', 'b', 'c', 'd'] — always works
```

---

## Quick Reference: The `iter(x) is x` Diagnostic

```python
def diagnose_iteration(obj: object) -> None:
    """Print everything you need to know about an object's iteration behavior."""
    has_iter = hasattr(obj, "__iter__")
    has_next = hasattr(obj, "__next__")

    print(f"Type:          {type(obj).__name__}")
    print(f"Has __iter__:  {has_iter}")
    print(f"Has __next__:  {has_next}")

    if has_iter:
        is_own_iter = iter(obj) is obj
        print(f"iter(x) is x:  {is_own_iter}")

        if is_own_iter:
            print(f"→ ITERATOR (single-pass, exhaustible)")
        else:
            print(f"→ ITERABLE (multi-pass, reusable)")
    else:
        print(f"→ NOT ITERABLE (can't use in for loop)")

# --- Test it ---
diagnose_iteration([1, 2, 3])
# Type: list | Has __iter__: True | Has __next__: False | iter(x) is x: False → ITERABLE

diagnose_iteration(iter([1, 2, 3]))
# Type: list_iterator | Has __iter__: True | Has __next__: True | iter(x) is x: True → ITERATOR

diagnose_iteration(x for x in [1])
# Type: generator | Has __iter__: True | Has __next__: True | iter(x) is x: True → ITERATOR

diagnose_iteration(range(5))
# Type: range | Has __iter__: True | Has __next__: False | iter(x) is x: False → ITERABLE

diagnose_iteration({1: "a"}.keys())
# Type: dict_keys | Has __iter__: True | Has __next__: False | iter(x) is x: False → ITERABLE

diagnose_iteration(42)
# Type: int | Has __iter__: False | Has __next__: False → NOT ITERABLE
```
