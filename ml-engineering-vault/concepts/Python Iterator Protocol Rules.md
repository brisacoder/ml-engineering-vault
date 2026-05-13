---
type: concept
title: "Python Iterator Protocol Rules"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Iteration Deep Dive]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Iterables]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Iterator Protocol Rules

> The iterator protocol is the formal contract that makes all iteration in Python work — `for` loops, unpacking, comprehensions, `*args`, everything. This note walks through the protocol from first principles, building the rules one discovery at a time. Based on Python Morsels (PyCon 2026).

---

## Discovery 1: `next()` Only Works on Iterators

The story starts with a surprise. You know `next()` works on generators. What about lists?

```python
numbers: list[int] = [1, 2, 3, 4, 5]

# Generators work with next() — we know this
def generatorify(iterable):
    for item in iterable:
        yield item

gen = generatorify(numbers)
print(next(gen))  # 1 — works fine

# But lists do NOT work with next()
try:
    next(numbers)
except TypeError as e:
    print(e)  # 'list' object is not an iterator
```

**What we've learned:**

```python
# Fact 1: next() only works on ITERATORS
# Fact 2: lists are NOT iterators (even though they're iterable)
# Fact 3: generators ARE iterators (next() works on them)
```

---

## Discovery 2: `iter()` Bridges the Gap

Python gives us `iter()` to get an iterator from any iterable:

```python
numbers: list[int] = [1, 2, 3, 4, 5]

# iter() on a list returns a list_iterator — a NEW object
it = iter(numbers)
print(type(it))  # <class 'list_iterator'>

# NOW next() works, because we have an iterator
print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
```

### The iterator remembers its position

This is critical: the iterator is a **cursor** that tracks where you are.

```python
# 'it' already advanced to position 3 above.
# A for-loop picks up from where next() left off:
for x in it:
    print(x)
# 4
# 5
# (only the remaining items — the iterator remembers its position)
```

### Want to start over? Get a NEW iterator.

```python
# The original list is untouched — it's not an iterator, it has no cursor.
# Just call iter() again to get a fresh cursor:
it_fresh = iter(numbers)
print(next(it_fresh))  # 1 — back to the beginning

# GOTCHA: the old iterator is still exhausted
print(list(it))  # [] — 'it' from above is done
```

---

## Discovery 3: `iter()` on a Generator Returns Itself

Here's where it gets interesting. What does `iter()` do to a generator?

```python
numbers: list[int] = [1, 2, 3, 4, 5]

def generatorify(iterable):
    """A hand-rolled iter(): turn any iterable into a generator."""
    for item in iterable:
        yield item

g = generatorify(numbers)
print(g)        # <generator object generatorify at 0x...>
print(iter(g))  # <generator object generatorify at 0x...>  ← SAME object!

# Proof: it's not just equal, it's the EXACT SAME object in memory
i = iter(g)
print(i is g)  # True — identity, not just equality
```

**Why?** A generator is already an iterator. It already has a cursor (`__next__`). Asking "give me an iterator for this iterator" returns the iterator itself — there's nothing new to create.

```python
# Compare the two cases:

# Case 1: iter() on a LIST → creates a NEW iterator
list_iter_a = iter(numbers)
list_iter_b = iter(numbers)
print(list_iter_a is list_iter_b)  # False — two independent cursors

# Case 2: iter() on a GENERATOR → returns the SAME object
gen_iter_a = iter(g)
gen_iter_b = iter(g)
print(gen_iter_a is gen_iter_b)    # True — same cursor, same object, same everything
print(gen_iter_a is g)             # True — it's all the same generator
```

### This is the fundamental asymmetry

```python
# Lists are ITERABLE: iter() creates a new independent iterator each time.
#   → You can iterate a list multiple times. Each for-loop gets its own cursor.
#   → Nested for-loops over the same list work perfectly.

# Generators are ITERATORS: iter() returns themselves.
#   → You can only iterate a generator ONCE. There's only one cursor.
#   → A second pass sees nothing — the cursor is at the end, permanently.

gen = generatorify([1, 2, 3])
print(list(gen))  # [1, 2, 3]
print(list(gen))  # [] ← exhausted, iter(gen) returns the same dead cursor
```

---

## The Protocol: Two Rules, Three Caveats

Now we can state the protocol precisely.

### The Two Rules

```python
# RULE 1: An ITERABLE is anything you can get an iterator from using iter()
#
#   iter(iterable) → iterator
#
#   Examples of iterables:
#     list, tuple, dict, set, str, range, bytes, frozenset,
#     dict_keys, dict_values, file objects, generators, ...

# RULE 2: An ITERATOR is an iterable that you can loop over using next()
#
#   next(iterator) → next value
#
#   An iterator must support BOTH iter() and next().
#   (That's why all iterators are also iterables — they have iter().)
```

### The Three Caveats

```python
# CAVEAT 1: Exhaustion = StopIteration
#
# An iterator is "done" when next() raises StopIteration.
it = iter([1])
print(next(it))  # 1
try:
    next(it)     # no more items
except StopIteration:
    print("Iterator exhausted")  # this prints


# CAVEAT 2: iter() on an iterator returns ITSELF
#
# This is the rule that makes generators single-pass.
it = iter([1, 2, 3])
print(iter(it) is it)  # True — always
# This is true for ALL iterators: list_iterator, generator, map, filter, zip...


# CAVEAT 3: Not all iterators are finite
#
# An iterator can yield values forever — next() never raises StopIteration.
from itertools import count

infinite = count(start=0, step=1)  # 0, 1, 2, 3, 4, ... forever
print(next(infinite))  # 0
print(next(infinite))  # 1
print(next(infinite))  # 2
# This will NEVER raise StopIteration — don't call list() on it!

# Another example: reading from a network socket, sensor stream, etc.
# The iterator protocol doesn't require finiteness.
```

### The Hierarchy in One Test

```python
def classify(obj: object) -> str:
    """The iter(x) is x test tells you everything."""
    try:
        iterator = iter(obj)
    except TypeError:
        return "NOT ITERABLE — can't use in for loop, can't call iter()"

    if iterator is obj:
        return "ITERATOR — single-pass, has next(), iter() returns self"
    else:
        return "ITERABLE — multi-pass, iter() returns a new iterator each time"

# Every Python object falls into exactly one of these three buckets:
print(classify(42))                 # NOT ITERABLE
print(classify([1, 2, 3]))         # ITERABLE (multi-pass)
print(classify(iter([1, 2, 3])))   # ITERATOR (single-pass)
print(classify(x for x in [1]))    # ITERATOR (single-pass)
print(classify(range(10)))         # ITERABLE (multi-pass)
print(classify("hello"))           # ITERABLE (multi-pass)
print(classify({}.keys()))         # ITERABLE (multi-pass)
print(classify(map(str, [1])))     # ITERATOR (single-pass)
print(classify(zip([1], [2])))     # ITERATOR (single-pass)
print(classify(open("/dev/null")))  # ITERATOR (single-pass)
```

---

## `generatorify` ≈ `iter()`: Same Job, Different Types

The instructor's `generatorify` is a pedagogical mirror of `iter()`:

```python
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

def generatorify(iterable: Iterable[T]) -> Iterator[T]:
    """Turn any iterable into a generator-based iterator.
    
    Functionally identical to iter(), but:
    - iter() returns a type-specific iterator (list_iterator, dict_keyiterator, ...)
    - generatorify() always returns a generator object
    """
    for item in iterable:
        yield item

numbers: list[int] = [1, 2, 3, 4, 5]

# Both do the same job: iterable → iterator
g = generatorify(numbers)     # <generator object>
i = iter(numbers)             # <list_iterator object>

# Both are iterators (iter returns self, next works)
assert iter(g) is g           # True
assert iter(i) is i           # True

# Both yield the same values
assert list(generatorify(numbers)) == list(iter(numbers))  # [1,2,3,4,5] == [1,2,3,4,5]

# Both are single-pass
g2 = generatorify(numbers)
list(g2)   # [1, 2, 3, 4, 5]
list(g2)   # [] — exhausted, just like iter()
```

**Why does this matter?** Because it shows that generators aren't magic — they're just a convenient syntax for creating objects that follow the iterator protocol. `yield` is syntactic sugar for building an object that has `__iter__` (returns self) and `__next__` (resumes execution until the next yield).

---

## How `for` Loops Work: Desugaring Step by Step

The entire iterator protocol exists to make `for` loops work. Here's the exact desugaring:

```python
# ─── What you write ───
def print_each_v1(iterable) -> None:
    for item in iterable:
        print(item)


# ─── What Python actually does ───
def print_each_v2(iterable) -> None:
    # Step 1: Call iter() on the iterable to get an iterator.
    #   - If iterable is a list → returns a new list_iterator
    #   - If iterable is a generator → returns itself (already an iterator)
    #   - If iterable has no __iter__ → raises TypeError
    iterator = iter(iterable)

    # Step 2: Loop forever, pulling one item at a time with next()
    while True:
        try:
            # Step 3: Get the next value from the iterator
            item = next(iterator)
        except StopIteration:
            # Step 4: StopIteration means the iterator is done → break
            break
        else:
            # Step 5: Execute the loop body with the yielded value
            print(item)


# Both versions behave identically on ANY iterable:
print_each_v1({1, 2, 3})           # works
print_each_v2({1, 2, 3})           # works — same output

print_each_v1(("a", "b", "c"))     # works
print_each_v2(("a", "b", "c"))     # works — same output

print_each_v1(x**2 for x in [3])   # works
print_each_v2(x**2 for x in [3])   # works — same output
```

### GOTCHA: The `while True` runs forever unless the iterator stops it

```python
from itertools import count

# count() is an infinite iterator — it never raises StopIteration
# This for-loop will run forever:
# for n in count(0):
#     print(n)  # 0, 1, 2, 3, ... never stops

# The desugared version makes this obvious:
# iterator = iter(count(0))  → returns count(0) itself
# while True:
#     item = next(iterator)  → always succeeds, never StopIteration
#     print(item)            → runs forever

# You MUST add your own break condition:
for n in count(0):
    if n >= 5:
        break
    print(n)  # 0, 1, 2, 3, 4
```

---

## Every Context That Uses the Protocol

The iterator protocol isn't just for `for` loops. Python uses it **everywhere** iteration happens:

```python
numbers: list[int] = [1, 2, 3, 4, 5]

# ─── 1. for loop ───
# Calls: iter(numbers) → next() repeatedly
for n in numbers:
    print(n)  # 1, 2, 3, 4, 5

# ─── 2. List/set/dict comprehensions ───
# Calls: iter(numbers) → next() repeatedly, collecting results
squares = [n ** 2 for n in numbers]   # [1, 4, 9, 16, 25]

# ─── 3. Generator expression ───
# Creates a lazy generator that will call iter/next when consumed
gen = (n ** 2 for n in numbers)       # nothing computed yet!

# ─── 4. Multiple assignment (unpacking) ───
# Calls: iter(numbers) → next() exactly 5 times
a, b, c, d, e = numbers
print(a, b, c, d, e)  # 1 2 3 4 5

# ─── 5. Star unpacking ───
# Calls: iter(numbers) → next() for first, collects rest into list
first, *rest = numbers
print(first)  # 1
print(rest)   # [2, 3, 4, 5]

# ─── 6. Function argument unpacking (*args) ───
# Calls: iter(my_range) → next() to unpack into positional args
my_range = [1, 6]
result = list(range(*my_range))  # range(1, 6) → [1, 2, 3, 4, 5]

# ─── 7. Constructors: list(), tuple(), set(), dict(), frozenset() ───
# All consume an iterable via iter/next
print(tuple(numbers))  # (1, 2, 3, 4, 5)
print(set(numbers))    # {1, 2, 3, 4, 5}

# ─── 8. Built-in functions that consume iterables ───
print(sum(numbers))          # 15 — iter/next over all items
print(min(numbers))          # 1
print(max(numbers))          # 5
print(sorted(numbers))       # [1, 2, 3, 4, 5]
print(any(n > 4 for n in numbers))   # True — short-circuits!
print(all(n > 0 for n in numbers))   # True

# ─── 9. str.join() ───
letters = ["h", "e", "l", "l", "o"]
print("".join(letters))  # "hello" — iterates over the list

# ─── 10. The `in` operator ───
# On iterators: calls next() until found or StopIteration
# WARNING: this consumes elements! See [[Python Iterator Gotchas#Gotcha 6]]
print(3 in numbers)  # True (on a list, uses __contains__, no consumption)
```

### GOTCHA: Single-pass iterators + multiple consumption contexts

```python
# If you pass a GENERATOR to any of these, the generator is consumed.
# A second consumer sees nothing:
gen = (n ** 2 for n in range(5))  # 0, 1, 4, 9, 16

total = sum(gen)       # 30 — consumes entire generator
remaining = list(gen)  # [] — generator is exhausted!

# If you need the data more than once, materialize first:
data: list[int] = [n ** 2 for n in range(5)]  # list comprehension, NOT generator
total = sum(data)       # 30
remaining = list(data)  # [0, 1, 4, 9, 16] — list is multi-pass
```

---

## `next()` with a Default: Avoiding StopIteration

`next()` has a second argument most people forget about:

```python
# Without default: raises StopIteration on empty iterator
empty = iter([])
try:
    next(empty)
except StopIteration:
    print("Crashed!")  # prints

# With default: returns the default instead of raising
empty = iter([])
value = next(empty, "nothing")  # "nothing" — no exception
print(value)

# This is incredibly useful for "get first or default" patterns:
from typing import Iterable, TypeVar

T = TypeVar("T")

def first(iterable: Iterable[T], default: T | None = None) -> T | None:
    """Safely get the first item from any iterable."""
    return next(iter(iterable), default)

print(first([10, 20, 30]))            # 10
print(first([]))                       # None
print(first([], "empty"))              # "empty"
print(first(x for x in range(0)))     # None
print(first(iter("hello")))           # "h"
```

---

## Building the Protocol from Scratch: `__iter__` and `__next__`

Under the hood, `iter()` calls `__iter__()` and `next()` calls `__next__()`. Here's what that looks like when you build your own:

### A Custom Iterator (Single-Pass)

```python
class CountDown:
    """An ITERATOR that counts down from n to 1.
    
    This is both iterable AND iterator:
    - __iter__ returns self (Rule: iter(iterator) is iterator)
    - __next__ returns the next value or raises StopIteration
    
    Single-pass: once it reaches 0, it's done forever.
    """

    def __init__(self, start: int) -> None:
        self.current: int = start

    def __iter__(self):
        # An iterator returns ITSELF — this is what makes iter(it) is it == True
        return self

    def __next__(self) -> int:
        if self.current <= 0:
            raise StopIteration  # signals exhaustion to for-loops and next()
        value = self.current
        self.current -= 1
        return value

# Usage — works everywhere the protocol is used:
cd = CountDown(3)
print(next(cd))        # 3
print(next(cd))        # 2
print(next(cd))        # 1
# next(cd) would raise StopIteration

# Verify it's an iterator:
cd2 = CountDown(3)
assert iter(cd2) is cd2  # True — iterator returns itself

# Works in for loop:
for n in CountDown(5):
    print(n, end=" ")  # 5 4 3 2 1

# GOTCHA: single-pass — can't re-iterate
cd3 = CountDown(3)
print(list(cd3))  # [3, 2, 1]
print(list(cd3))  # [] — exhausted!
```

### A Custom Iterable (Multi-Pass)

```python
class CountDownReusable:
    """An ITERABLE (not iterator) that counts down from n to 1.
    
    Multi-pass: __iter__ returns a NEW iterator each time, so you
    can iterate over it as many times as you want.
    """

    def __init__(self, start: int) -> None:
        self.start: int = start

    def __iter__(self):
        # Return a FRESH generator each time — this is what makes it reusable.
        # Each call to __iter__ creates a new generator with its own state.
        current = self.start
        while current > 0:
            yield current
            current -= 1

# Multi-pass — works every time:
cd = CountDownReusable(3)
print(list(cd))  # [3, 2, 1]
print(list(cd))  # [3, 2, 1] — fresh iterator each time!
print(list(cd))  # [3, 2, 1] — still works!

# Verify it's an iterable, NOT an iterator:
assert iter(cd) is not cd       # True — iter() returns a NEW object
assert iter(cd) is not iter(cd)  # True — each call returns a DIFFERENT generator

# Nested iteration works:
cd5 = CountDownReusable(3)
for i in cd5:
    for j in cd5:        # gets its OWN independent iterator
        print(f"({i},{j})", end=" ")
# (3,3) (3,2) (3,1) (2,3) (2,2) (2,1) (1,3) (1,2) (1,1)
```

### The Key Difference: What `__iter__` Returns

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │              __iter__ returns self?                              │
# │                                                                 │
# │  YES → It's an ITERATOR                                        │
# │        - Has __next__                                           │
# │        - Single-pass (one cursor, shared by everyone)           │
# │        - Examples: generator, map, filter, zip, file            │
# │                                                                 │
# │  NO  → It's an ITERABLE (that produces iterators)              │
# │        - Does NOT have __next__ (usually)                       │
# │        - Multi-pass (each iter() call → fresh cursor)           │
# │        - Examples: list, tuple, dict, set, str, range           │
# └─────────────────────────────────────────────────────────────────┘

# This is THE test:
def is_iterator(obj: object) -> bool:
    """The canonical way to check: does iter(obj) return obj itself?"""
    return iter(obj) is obj

assert is_iterator(iter([1, 2])) is True    # list_iterator → yes
assert is_iterator([1, 2]) is False         # list → no
assert is_iterator((x for x in [])) is True # generator → yes
assert is_iterator(range(5)) is False       # range → no
```

---

## Infinite Iterators: Protocol Doesn't Require Exhaustion

```python
from itertools import count, cycle, repeat

# count: 0, 1, 2, 3, 4, ... forever
counter = count(start=0, step=2)
print(next(counter))  # 0
print(next(counter))  # 2
print(next(counter))  # 4
# Never raises StopIteration

# cycle: repeats an iterable forever
colors = cycle(["red", "green", "blue"])
print(next(colors))  # red
print(next(colors))  # green
print(next(colors))  # blue
print(next(colors))  # red  ← wraps around!

# repeat: yields the same value n times (or forever)
fives = repeat(5, times=3)
print(list(fives))  # [5, 5, 5]

forever_fives = repeat(5)  # no times= → infinite
print(next(forever_fives))  # 5
# This never stops

# GOTCHA: Never call list() on an infinite iterator!
# list(count())   → hangs forever, eats all memory, then crashes
# list(cycle(x))  → same
# Always use islice or a break condition:
from itertools import islice
print(list(islice(count(), 5)))  # [0, 1, 2, 3, 4] — safely take first 5
```

---

## Summary: The Rules on One Page

```python
# ═══════════════════════════════════════════════════════════
#  THE ITERATOR PROTOCOL — COMPLETE RULES
# ═══════════════════════════════════════════════════════════
#
#  RULE 1: iter(iterable) → iterator
#          Anything with __iter__ is iterable.
#
#  RULE 2: next(iterator) → next value
#          Anything with __iter__ AND __next__ is an iterator.
#
#  CAVEAT 1: StopIteration means "done"
#            next() raises it when the iterator is exhausted.
#
#  CAVEAT 2: iter(iterator) is iterator
#            Calling iter() on an iterator returns itself.
#            This is why iterators are single-pass.
#
#  CAVEAT 3: Exhaustion is not required
#            Iterators can yield values forever.
#
#  CONSEQUENCE: All iterators are iterables (they have __iter__).
#               Not all iterables are iterators (lists lack __next__).
#
#  THE TEST:   iter(x) is x  →  True = iterator, False = iterable
#
#  FOR LOOP:   for item in X:   ←→   it = iter(X)
#                  body                while True:
#                                          try: item = next(it)
#                                          except StopIteration: break
#                                          body
# ═══════════════════════════════════════════════════════════
```
