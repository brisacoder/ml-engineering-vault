---
type: concept
title: "Python Custom Iterables"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Iteration Deep Dive]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Custom Iterables

> Now that you know the [[Python Iterator Protocol Rules|iterator protocol]], you can make your own objects work with `for` loops, unpacking, comprehensions, and every other iteration context in Python. There are two paths: making an **iterable** (multi-pass, reusable) or making an **iterator** (single-pass, stateful). The right choice depends on whether your object IS a cursor or HAS a cursor. Based on Python Morsels (PyCon 2026).

---

## How `iter()` Actually Works: It Calls `__iter__`

Before building custom iterables, you need to know what `iter()` does under the hood:

```python
numbers: list[int] = [1, 2, 3, 4]

# These two calls are equivalent:
iterator_a = iter(numbers)          # built-in function
iterator_b = numbers.__iter__()     # dunder method it calls

print(type(iterator_a))  # <class 'list_iterator'>
print(type(iterator_b))  # <class 'list_iterator'>

# Similarly, next() calls __next__:
it = iter(numbers)
print(next(it))           # 1
print(it.__next__())      # 2  — same thing
```

**The rule:** if your object has an `__iter__` method that returns an iterator, your object is iterable. That's it. That's the entire contract.

```python
# The full protocol:
#
# iter(obj)  →  calls obj.__iter__()  →  must return an iterator
# next(it)   →  calls it.__next__()   →  must return next value or raise StopIteration
#
# An ITERABLE needs:  __iter__  (returns a new iterator)
# An ITERATOR needs:  __iter__  (returns self) + __next__  (returns next value)
```

---

## Path 1: Custom Iterable (Multi-Pass) — The Common Case

### Approach A: Delegate to an Existing Iterable

The simplest `__iter__`: build a tuple/list from your data and delegate:

```python
class Point:
    """A 2D point that supports iteration and unpacking.
    
    __iter__ delegates to a tuple — the tuple knows how to
    produce an iterator, so we just ask it to.
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __iter__(self):
        # Build a tuple from our data, then return ITS iterator.
        # iter((self.x, self.y)) returns a tuple_iterator.
        return iter((self.x, self.y))

# Now Point works everywhere iteration is used:
p = Point(1, 2)

# Unpacking (uses iter + next under the hood):
x, y = p
print(x, y)  # 1 2

# list() constructor (calls iter, then next repeatedly):
print(list(p))  # [1, 2]

# for loop:
for coord in Point(3, 4):
    print(coord)  # 3, then 4

# It's multi-pass — each call to iter() gets a fresh tuple_iterator:
p2 = Point(5, 6)
print(list(p2))  # [5, 6]
print(list(p2))  # [5, 6] — works again!
assert iter(p2) is not p2  # True — it's an iterable, not an iterator
```

### Approach B: `__iter__` as a Generator Function (Preferred)

Instead of building a temporary tuple, make `__iter__` itself a generator function using `yield`. This is the **idiomatic** way:

```python
class Point:
    """A 2D point with __iter__ as a generator function.
    
    When __iter__ contains `yield`, it becomes a generator function.
    Each call creates a NEW generator object (an iterator).
    This is why it's multi-pass — fresh generator every time.
    """

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    def __iter__(self):
        # This is a GENERATOR FUNCTION (has yield).
        # Each call returns a NEW generator object.
        yield self.x
        yield self.y

# Works identically:
x, y = Point(1, 2)
print(x, y)        # 1 2
print(list(Point(3, 4)))  # [3, 4]

# Multi-pass proof:
p = Point(5, 6)
print(list(p))  # [5, 6]
print(list(p))  # [5, 6] — fresh generator each call to __iter__
```

### Why Generator `__iter__` Is Preferred

```python
# Approach A (delegation):
#   def __iter__(self):
#       return iter((self.x, self.y))
#   
#   → Creates a temporary tuple in memory
#   → Then creates a tuple_iterator from it
#   → Two objects allocated per iteration
#   → Fine for small data, wasteful for large data

# Approach B (generator):
#   def __iter__(self):
#       yield self.x
#       yield self.y
#   
#   → Creates ONE generator object
#   → Values yielded on demand, nothing stored
#   → Works for any amount of data
#   → Reads more naturally — the yields ARE the iteration

# For a 2-element Point, either approach works.
# For a class wrapping 10 million items, the generator approach
# avoids building a 10-million-item intermediate list.
```

---

## Path 2: Custom Iterator (Single-Pass) — The Stateful Case

Sometimes the object IS the cursor — it tracks position, generates values on the fly, or maintains state that changes as you iterate. In that case, you need an **iterator**, not just an iterable.

### The Two Requirements for an Iterator

```python
# An iterator must implement:
#
# 1. __iter__(self) → return self
#    (This makes the iterator usable in for-loops.
#     All iterators are iterables that return themselves.)
#
# 2. __next__(self) → return the next value OR raise StopIteration
#    (This is what next() calls. It's the "advance the cursor" method.)
```

### Example: Countdown Iterator

```python
class Countdown:
    """A single-pass iterator that counts down from start to 1.
    
    __iter__ returns self → this IS the iterator (not a factory for iterators).
    __next__ decrements and returns, or raises StopIteration at 0.
    """

    def __init__(self, start: int) -> None:
        self.current: int = start

    def __iter__(self):
        # THE defining trait of an iterator: return yourself.
        # This is what makes iter(countdown) is countdown == True.
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration  # signals "I'm done" to for-loops, list(), etc.
        value = self.current
        self.current -= 1
        return value

# Usage with next():
c = Countdown(3)
print(next(c))  # 3
print(next(c))  # 2

# The for-loop / list() picks up where next() left off:
print(list(c))  # [1] — only one value remaining

# Verify it's an iterator:
c2 = Countdown(5)
assert iter(c2) is c2  # True — iterator returns itself

# GOTCHA: single-pass — once exhausted, it's done forever:
c3 = Countdown(2)
print(list(c3))  # [2, 1]
print(list(c3))  # [] — exhausted, no reset mechanism
```

---

## Iterable vs Iterator: The Key Decision

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │  MAKE AN ITERABLE (__iter__ returns new iterator) when:         │
# │                                                                 │
# │  - The object represents DATA (a collection, a view, a range)  │
# │  - Users should be able to iterate it multiple times            │
# │  - Nested for-loops over the same object should work            │
# │  - The object doesn't "move forward" — it sits still            │
# │                                                                 │
# │  → __iter__ uses yield (generator function)                     │
# │  → No __next__ needed                                           │
# │  → iter(obj) is NOT obj                                         │
# │                                                                 │
# │  Examples: Point, ReverseView, Matrix, Deck, DateRange          │
# ├─────────────────────────────────────────────────────────────────┤
# │  MAKE AN ITERATOR (__iter__ returns self) when:                 │
# │                                                                 │
# │  - The object IS a cursor / stream / position tracker           │
# │  - It generates values on the fly (possibly infinite)           │
# │  - It needs to expose state BETWEEN next() calls                │
# │  - Being single-pass is correct behavior                        │
# │                                                                 │
# │  → __iter__ returns self                                        │
# │  → __next__ returns next value or raises StopIteration          │
# │  → iter(obj) IS obj                                             │
# │                                                                 │
# │  Examples: Countdown, RandomNumberGenerator, Sensor, FileReader │
# └─────────────────────────────────────────────────────────────────┘
```

### The Instructor's Advice

> In practice, you rarely need to write iterator classes. A `__iter__` method that uses `yield` is almost always simpler than writing separate `__iter__` and `__next__` methods. Iterator classes are mainly useful when **the object itself needs to maintain state between iterations** — like exposing counters, statistics, or configuration that changes as you iterate.

```python
# The 90% case: just use yield in __iter__
class MyCollection:
    def __init__(self, data: list) -> None:
        self.data = data
    
    def __iter__(self):
        for item in self.data:
            yield item
        # That's it. No __next__, no StopIteration,
        # no self.current tracking. yield handles everything.

# The 10% case: you need an iterator class because you need
# to read attributes BETWEEN calls to next():
class StripLines:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.lf = 0       # ← these counters are the reason
        self.cr = 0       #   we need an iterator class
        self.crlf = 0     #   (can't access locals inside a generator)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        line = next(self.lines)  # raises StopIteration when done
        # ... update self.lf, self.cr, self.crlf ...
        return stripped_line
```

---

## GOTCHA: Dictionary Mutation During Iteration

This isn't about custom iterables, but it's a protocol gotcha that surprises everyone:

```python
d: dict[str, int] = {}

# Get an iterator for the dict
it = iter(d)

# Mutate the dict AFTER getting the iterator
d["a"] = 1

# Now try to use the iterator:
try:
    next(it)
except RuntimeError as e:
    print(e)  # dictionary changed size during iteration
```

### Why It Happens

```python
# When you call iter(d), the dict creates a dict_keyiterator
# that snapshots the dict's internal version number.
# 
# If the dict's size changes (add or remove keys), the version number
# changes. On the next call to __next__, the iterator detects the
# mismatch and raises RuntimeError to prevent undefined behavior.
#
# This is a SAFETY mechanism — iterating a mutating dict could skip
# items, repeat items, or crash. Python chooses to fail loudly.

d = {"a": 1, "b": 2, "c": 3}

# This crashes:
try:
    for key in d:
        if key == "b":
            del d[key]  # RuntimeError: dictionary changed size during iteration
except RuntimeError:
    print("Can't modify dict size while iterating!")

# Fix: iterate over a copy of the keys
d = {"a": 1, "b": 2, "c": 3}
for key in list(d):       # list(d) creates a snapshot of the keys
    if key == "b":
        del d[key]         # safe — we're iterating over the snapshot, not the dict
print(d)  # {'a': 1, 'c': 3}
```

### GOTCHA: Modifying Values (Not Size) Is Allowed

```python
d = {"a": 1, "b": 2, "c": 3}

# Changing VALUES while iterating is fine — the dict size doesn't change:
for key in d:
    d[key] *= 10  # no error
print(d)  # {'a': 10, 'b': 20, 'c': 30}
```

---

## Exercises and Solutions

### Exercise 1: `Point` — 3D Point with Unpacking

```python
class Point:
    """A 3D point that supports iteration and unpacking.
    
    Using yield in __iter__ makes this a reusable iterable.
    Each call to __iter__ creates a fresh generator → multi-pass.
    
    >>> p = Point(2, 3, 6)
    >>> x, y, z = p
    >>> x, y, z
    (2, 3, 6)
    >>> list(p)
    [2, 3, 6]
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def __iter__(self):
        # Generator function → returns a new generator (iterator) each time.
        # This is why Point is a reusable iterable, not a single-pass iterator.
        yield self.x
        yield self.y
        yield self.z

# Tests
p = Point(2, 3, 6)
x, y, z = p
assert (x, y, z) == (2, 3, 6)
assert list(p) == [2, 3, 6]

# Multi-pass:
assert list(p) == [2, 3, 6]  # second time still works

# Works in all iteration contexts:
assert tuple(p) == (2, 3, 6)
assert sum(p) == 11
print("All Point tests passed")
```

### Exercise 2: `RandomNumberGenerator` — Inexhaustible Iterator

```python
import random

class RandomNumberGenerator:
    """An infinite iterator that yields random ints in [low, high].
    
    This MUST be an iterator (not just an iterable) because:
    - It's a stream, not a collection
    - It never exhausts (infinite)
    - iter(obj) is obj must be True (exercise requirement)
    
    __next__ never raises StopIteration → infinite iterator.
    
    >>> rng = RandomNumberGenerator(4, 8)
    >>> next(rng) in range(4, 9)
    True
    >>> iter(rng) is rng
    True
    """

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def __iter__(self):
        # Iterator protocol: return self.
        # This makes iter(rng) is rng == True.
        return self

    def __next__(self) -> int:
        # No StopIteration → this iterator never exhausts.
        # Each call produces a fresh random number.
        return random.randint(self.low, self.high)

# Tests
rng = RandomNumberGenerator(4, 8)
assert iter(rng) is rng  # it's an iterator

# Pull a bunch of values, all should be in [4, 8]
values = [next(rng) for _ in range(1000)]
assert all(4 <= v <= 8 for v in values)
assert min(values) >= 4
assert max(values) <= 8

# It's infinite — never raises StopIteration
# (We can't call list(rng) — it would hang forever!)

# But we CAN use islice to take a finite slice:
from itertools import islice
sample = list(islice(rng, 5))
print(f"Sample: {sample}")  # e.g. [5, 4, 8, 6, 7]
assert len(sample) == 5

print("All RandomNumberGenerator tests passed")
```

### Exercise 3: `ReverseView` — Reusable Reversed View

```python
class ReverseView:
    """A reusable iterable that yields items in reverse order.
    
    Key difference from reversed(): reversed() returns an iterator
    (single-pass). ReverseView is an iterable (multi-pass).
    
    It's also a LIVE VIEW: it reads from the original sequence
    each time you iterate, so mutations to the original are visible.
    
    >>> numbers = [2, 1, 3, 4, 7, 11]
    >>> view = ReverseView(numbers)
    >>> list(view)
    [11, 7, 4, 3, 1, 2]
    >>> list(view)
    [11, 7, 4, 3, 1, 2]
    >>> numbers.append(18)
    >>> list(view)
    [18, 11, 7, 4, 3, 1, 2]
    """

    def __init__(self, sequence) -> None:
        # Store a REFERENCE to the original sequence, don't copy it.
        # This is what makes it a live view — mutations to the
        # original sequence are visible through the view.
        self.sequence = sequence

    def __iter__(self):
        # Generator function → fresh generator each call → multi-pass.
        # Iterate backwards using negative indexing.
        # Reads self.sequence at iteration time, not at __init__ time,
        # so it always reflects the current state of the original.
        for i in range(len(self.sequence) - 1, -1, -1):
            yield self.sequence[i]

# Tests
numbers = [2, 1, 3, 4, 7, 11]
view = ReverseView(numbers)

# Basic reverse:
assert list(view) == [11, 7, 4, 3, 1, 2]

# Multi-pass (reusable):
assert list(view) == [11, 7, 4, 3, 1, 2]

# It's an iterable, NOT an iterator:
assert iter(view) is not view

# Live view — mutations to original are visible:
numbers.append(18)
assert list(view) == [18, 11, 7, 4, 3, 1, 2]

numbers.pop(0)  # remove 2
assert list(view) == [18, 11, 7, 4, 3, 1]

# Compare with reversed() — which is an iterator (single-pass):
rev_iter = reversed(numbers)
assert iter(rev_iter) is rev_iter  # iterator!
list(rev_iter)  # [18, 11, 7, 4, 3, 1]
list(rev_iter)  # [] — exhausted!

print("All ReverseView tests passed")
```

### Why ReverseView Is a View (Not a Copy)

```python
# The key is that __init__ stores a REFERENCE, not a copy:
#
#   self.sequence = sequence    ← reference (view)
#   self.sequence = list(sequence)  ← copy (snapshot)
#
# A view reflects changes to the original:
original = [1, 2, 3]
view = ReverseView(original)
original.append(4)
print(list(view))  # [4, 3, 2, 1] — sees the new 4

# A snapshot would NOT see changes:
# If we had copied: self.sequence = list(sequence)
# original.append(4) would not affect the ReverseView.

# This is the same pattern as dict.keys(), dict.values(), dict.items():
d = {"a": 1}
keys = d.keys()  # view, not copy
d["b"] = 2
print(list(keys))  # ['a', 'b'] — sees the new key
```

### Exercise 4: `strip_lines` — Iterator Class with Accessible State

```python
class StripLines:
    """An iterator that strips line endings and counts them by type.
    
    This is the case where an iterator CLASS is the right tool:
    we need to expose lf, cr, crlf counters as attributes that
    callers can read BETWEEN calls to next(). A generator function's
    local variables are trapped inside — you can't access them.
    
    >>> lines = StripLines(["hello\\r\\n", "world\\n", "foo\\r", "bar"])
    >>> next(lines)
    'hello'
    >>> lines.crlf
    1
    >>> lines.lf
    0
    >>> list(lines)
    ['world', 'foo', 'bar']
    >>> lines.lf
    1
    >>> lines.cr
    1
    >>> lines.crlf
    1
    """

    def __init__(self, lines) -> None:
        # Convert to an iterator — works whether lines is a list, generator,
        # file object, or already an iterator.
        self._lines = iter(lines)

        # These counters are the whole reason we use an iterator class
        # instead of a generator function. They're attributes on self,
        # so callers can inspect them at any point during iteration.
        self.lf: int = 0      # lines ending in \n (but not \r\n)
        self.cr: int = 0      # lines ending in \r (but not \r\n)
        self.crlf: int = 0    # lines ending in \r\n

    def __iter__(self):
        # Iterator protocol: return self.
        return self

    def __next__(self) -> str:
        # Get the next line from the underlying iterator.
        # If it's exhausted, this raises StopIteration — which is
        # exactly what we want. We don't catch it; we let it propagate.
        line: str = next(self._lines)

        # Check endings in order: \r\n first (it contains both \r and \n,
        # so checking \n or \r first would give a false match).
        if line.endswith("\r\n"):
            self.crlf += 1
            return line[:-2]       # strip the 2-char ending
        elif line.endswith("\n"):
            self.lf += 1
            return line[:-1]       # strip the 1-char ending
        elif line.endswith("\r"):
            self.cr += 1
            return line[:-1]       # strip the 1-char ending
        else:
            return line            # no line ending to strip

# Tests
lines = StripLines(["hello\r\n", "world\n", "foo\r", "bar"])

# Pull first item and check state:
assert next(lines) == "hello"
assert lines.crlf == 1
assert lines.lf == 0
assert lines.cr == 0

# Consume the rest:
assert list(lines) == ["world", "foo", "bar"]
assert lines.lf == 1
assert lines.cr == 1
assert lines.crlf == 1

# It's an iterator:
lines2 = StripLines(["test\n"])
assert iter(lines2) is lines2

# Exhaustion:
list(lines2)
assert list(lines2) == []  # single-pass, exhausted

print("All StripLines tests passed")
```

### Why a Generator Function Can't Do This

```python
# Here's the generator function version for comparison:
def strip_lines_gen(lines):
    lf = cr = crlf = 0
    for line in lines:
        if line.endswith("\r\n"):
            crlf += 1
            yield line[:-2]
        elif line.endswith("\n"):
            lf += 1
            yield line[:-1]
        elif line.endswith("\r"):
            cr += 1
            yield line[:-1]
        else:
            yield line
    # Problem: lf, cr, crlf are LOCAL VARIABLES.
    # They're trapped inside the generator frame.
    # There is NO WAY to access them from outside.

gen = strip_lines_gen(["hello\r\n", "world\n"])
next(gen)  # 'hello'

# How do we check the count? We can't!
# gen.crlf → AttributeError: 'generator' object has no attribute 'crlf'

# The iterator class solves this by putting the counters on self,
# where they're accessible as regular attributes at any time.
```

---

## Summary: The Two Patterns

```python
# ═══════════════════════════════════════════════════════════
#  PATTERN 1: CUSTOM ITERABLE (multi-pass, reusable)
# ═══════════════════════════════════════════════════════════
#
#  class MyIterable:
#      def __iter__(self):
#          yield value1
#          yield value2
#          # ... or: for item in self.data: yield item
#
#  - __iter__ is a generator function (has yield)
#  - Each call creates a NEW generator → fresh cursor
#  - iter(obj) is NOT obj
#  - Can iterate multiple times
#  - Use for: collections, views, data holders
#
# ═══════════════════════════════════════════════════════════
#  PATTERN 2: CUSTOM ITERATOR (single-pass, stateful)
# ═══════════════════════════════════════════════════════════
#
#  class MyIterator:
#      def __iter__(self):
#          return self          # ← THE key line
#
#      def __next__(self):
#          if done:
#              raise StopIteration
#          # update state
#          return value
#
#  - __iter__ returns self
#  - __next__ advances and returns (or raises StopIteration)
#  - iter(obj) IS obj
#  - Single-pass — once exhausted, done forever
#  - Use for: streams, generators-with-attributes, infinite sequences
#
# ═══════════════════════════════════════════════════════════
#  RULE OF THUMB:
#  Use yield in __iter__ (Pattern 1) unless you need
#  accessible state between next() calls (Pattern 2).
# ═══════════════════════════════════════════════════════════
```
