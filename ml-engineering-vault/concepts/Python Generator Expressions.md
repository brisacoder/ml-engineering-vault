---
type: concept
title: "Python Generator Expressions"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Comprehensions]]"
  - "[[Python Custom Comprehensions]]"
  - "[[Python Custom Iterables]]"
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Iteration Deep Dive]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Generator Expressions

> One character changes everything. Replace `[...]` with `(...)` and you go from building an entire list in memory to producing values one at a time, on demand. Generator expressions are the lazy counterpart to [[Python Comprehensions|list comprehensions]]. They produce **iterators**, not lists — and that single fact explains every behavior, gotcha, and use case. Based on Python Morsels (PyCon 2026).

---

## The Mind-Blowing Moment: `[]` vs `()`

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# ─── List comprehension: BRACKETS → builds the full list RIGHT NOW ───
squares_list = [n**2 for n in numbers]
print(squares_list)
# [4, 1, 9, 16, 49, 121, 324]
# All 7 values computed and stored in memory. Done.

# ─── Generator expression: PARENTHESES → builds NOTHING yet ───
squares_gen = (n**2 for n in numbers)
print(squares_gen)
# <generator object <genexpr> at 0x...>
# Zero values computed. Zero memory used (beyond the tiny generator object).
# It's a PROMISE to compute values when asked.
```

### What Just Happened?

```python
# The list comprehension:
#   1. Iterates over ALL of numbers
#   2. Computes n**2 for EVERY item
#   3. Stores ALL results in a new list
#   4. Returns that list
#   → You have a list. You can index it, slice it, len() it, iterate it 100 times.

# The generator expression:
#   1. Creates a generator object (an ITERATOR)
#   2. That's it. Nothing else runs.
#   → You have an iterator. You can call next() on it. You can iterate it ONCE.
#   → Each value is computed on-demand, one at a time, when you ask for it.
```

---

## Lazy Evaluation: Nothing Runs Until You Pull

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]
squares_gen = (n**2 for n in numbers)

# At this point, n**2 has NOT been computed for any element.
# The generator is sitting idle, waiting.

# Pull the first value:
print(next(squares_gen))  # 4   → only NOW does it compute 2**2
print(next(squares_gen))  # 1   → only NOW does it compute 1**2
print(next(squares_gen))  # 9   → only NOW does it compute 3**2

# The remaining 4 values (4**2, 7**2, 11**2, 18**2) have NOT been computed yet.
# They don't exist anywhere. They'll be computed when (if!) we ask for them.

# Pull the rest with a for loop:
for sq in squares_gen:
    print(sq, end=" ")
# 16 49 121 324
# The for-loop calls next() internally until StopIteration.
```

### Proof: Side Effects Show When Code Actually Runs

```python
# Add a print() to see exactly when computation happens:
def square_verbose(n: int) -> int:
    print(f"  Computing {n}**2")
    return n ** 2

# List comprehension: ALL prints happen immediately
print("Creating list comprehension:")
squares_list = [square_verbose(n) for n in [2, 3, 4]]
#   Computing 2**2
#   Computing 3**2
#   Computing 4**2     ← all three computed NOW
print(f"Result: {squares_list}")  # [4, 9, 16]

print()

# Generator expression: NOTHING prints yet
print("Creating generator expression:")
squares_gen = (square_verbose(n) for n in [2, 3, 4])
print("Generator created, no computation yet!")
# (silence — nothing computed)

print("Now pulling values:")
print(next(squares_gen))
#   Computing 2**2     ← computed NOW, on demand
# 4
print(next(squares_gen))
#   Computing 3**2     ← computed NOW, on demand
# 9
# 4**2 still hasn't been computed!
```

---

## A Generator Expression IS an Iterator

This is the core insight that connects everything back to the [[Python Iterator Protocol Rules|iterator protocol]]:

```python
numbers: list[int] = [1, 2, 3]

gen = (n * 10 for n in numbers)

# ─── It's an iterator: iter() returns itself ───
print(iter(gen) is gen)  # True
# This is THE test (see [[Python Iterator Protocol Rules#The Hierarchy in One Test]])

# ─── It has __next__: next() works ───
print(next(gen))  # 10
print(next(gen))  # 20
print(next(gen))  # 30

# ─── It raises StopIteration when exhausted ───
try:
    next(gen)
except StopIteration:
    print("Exhausted!")  # prints

# ─── It's single-pass: second iteration gets nothing ───
gen2 = (n * 10 for n in numbers)
print(list(gen2))  # [10, 20, 30]
print(list(gen2))  # [] ← exhausted, permanently
```

### Compare: List Comprehension Is an Iterable (Not Iterator)

```python
numbers: list[int] = [1, 2, 3]

lst = [n * 10 for n in numbers]

# ─── It's an iterable: iter() returns a NEW iterator each time ───
print(iter(lst) is lst)  # False — list is NOT its own iterator

# ─── It's multi-pass: iterate as many times as you want ───
print(list(lst))  # [10, 20, 30]
print(list(lst))  # [10, 20, 30] — still works!
print(list(lst))  # [10, 20, 30] — always works!

# ─── It supports indexing, slicing, len ───
print(lst[0])      # 10
print(lst[-1])     # 30
print(len(lst))    # 3
print(lst[1:])     # [20, 30]
# Generators support NONE of these.
```

---

## Memory: Why Generator Expressions Exist

```python
import sys

# ─── List comprehension: stores ALL values ───
big_list = [n ** 2 for n in range(1_000_000)]
print(sys.getsizeof(big_list))   # ~8,448,728 bytes (~8 MB)

# ─── Generator expression: stores NOTHING ───
big_gen = (n ** 2 for n in range(1_000_000))
print(sys.getsizeof(big_gen))    # ~200 bytes (just the generator object)

# The generator is ~40,000x smaller.
# It produces 1,000,000 values but never holds more than ONE at a time.
```

### When This Matters: Passing to Functions

```python
numbers = range(10_000_000)  # 10 million items

# BAD: builds a 10-million-item list, THEN passes to sum()
total = sum([n ** 2 for n in numbers])  # ~80 MB of memory for the list

# GOOD: generator feeds values one at a time to sum()
total = sum(n ** 2 for n in numbers)    # ~200 bytes — O(1) memory
# sum() pulls one value, adds it, discards it, pulls the next.

# Both give the exact same answer. The generator version uses ~400,000x less memory.
print(total)  # 333333283333335000000
```

### Syntax Sugar: Parentheses Are Optional Inside Function Calls

```python
# When a generator expression is the ONLY argument, you can drop the extra parens:

# These are identical:
total = sum((n ** 2 for n in range(10)))  # explicit parens around genexpr
total = sum(n ** 2 for n in range(10))    # genexpr parens double as function call parens

# With multiple arguments, you MUST keep the genexpr parens:
result = max((n ** 2 for n in range(10)), default=0)  # genexpr needs its own parens
# result = max(n ** 2 for n in range(10), default=0)  # SyntaxError!
```

---

## Generator Expressions vs Generator Functions

They produce the same thing — a generator (iterator) — but the syntax is different:

```python
numbers: list[int] = [1, 2, 3, 4, 5]

# ─── Generator EXPRESSION: one-liner, like a comprehension ───
gen_expr = (n ** 2 for n in numbers if n % 2 == 0)

# ─── Generator FUNCTION: uses yield, can have complex logic ───
def gen_func(nums: list[int]):
    for n in nums:
        if n % 2 == 0:
            yield n ** 2

gen_from_func = gen_func(numbers)

# Both produce generators — same type, same behavior:
print(type(gen_expr))       # <class 'generator'>
print(type(gen_from_func))  # <class 'generator'>

print(list(gen_expr))       # [4, 16]
print(list(gen_from_func))  # [4, 16]
```

### When to Use Which

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │  GENERATOR EXPRESSION  (n**2 for n in items)                    │
# │                                                                 │
# │  Use when:                                                      │
# │  - Simple transform and/or filter                               │
# │  - One line is readable                                         │
# │  - Passing directly to a consuming function                     │
# │                                                                 │
# │  Examples:                                                      │
# │    sum(x**2 for x in data)                                      │
# │    max(len(w) for w in words)                                   │
# │    ",".join(str(n) for n in numbers)                            │
# │    any(n > 100 for n in scores)                                 │
# ├─────────────────────────────────────────────────────────────────┤
# │  GENERATOR FUNCTION  def gen(): ... yield ...                   │
# │                                                                 │
# │  Use when:                                                      │
# │  - Complex logic (multiple conditions, state, try/except)       │
# │  - You need to yield from multiple places                       │
# │  - Logic doesn't fit in one expression                          │
# │  - You want to name it for reuse                                │
# │                                                                 │
# │  Examples:                                                      │
# │    def fibonacci(n): ...                                        │
# │    def read_chunks(file, size): ...                              │
# │    def flatten_recursive(nested): ...                           │
# └─────────────────────────────────────────────────────────────────┘
```

---

## Gotchas

### GOTCHA 1: You Can Only Consume a Generator Expression Once

```python
gen = (n ** 2 for n in [1, 2, 3])

# First consumption — works perfectly
total = sum(gen)        # 14
print(f"Sum: {total}")

# Second consumption — silent disaster
maximum = max(gen, default=0)  # 0! Not 9!
# gen is exhausted. max() sees an empty iterator. Returns default.

# Without the default, it would crash:
gen2 = (n ** 2 for n in [1, 2, 3])
list(gen2)  # [1, 4, 9]
try:
    max(gen2)  # no default → ValueError
except ValueError as e:
    print(e)   # max() arg is an empty sequence
```

**Fix:** If you need the data more than once, use a list comprehension.

```python
# Materialize when you need multiple passes:
squares: list[int] = [n ** 2 for n in [1, 2, 3]]
total = sum(squares)     # 14
maximum = max(squares)   # 9 — works because list is multi-pass
```

### GOTCHA 2: You Can't Index, Slice, or `len()` a Generator

```python
gen = (n ** 2 for n in range(10))

# None of these work:
try:
    gen[0]          # TypeError: 'generator' object is not subscriptable
except TypeError:
    pass

try:
    len(gen)        # TypeError: object of type 'generator' has no len()
except TypeError:
    pass

try:
    gen[2:5]        # TypeError: 'generator' object is not subscriptable
except TypeError:
    pass

# A generator only knows about the NEXT value. It has no concept of
# "how many items" or "which item is at position 3".
# It's a stream, not a container.
```

**Fix:** Materialize to a list if you need random access.

```python
# Or use itertools.islice for lazy slicing without materializing everything:
from itertools import islice

gen = (n ** 2 for n in range(1_000_000))
third_through_fifth = list(islice(gen, 2, 5))  # [4, 9, 16]
# Only computed 5 values, not 1 million.
```

### GOTCHA 3: Late Binding — The Generator Captures Variables, Not Values

```python
x = 10
gen = (x + i for i in range(3))

# Change x BEFORE consuming the generator:
x = 1000

print(list(gen))  # [1000, 1001, 1002] — NOT [10, 11, 12]!
# The generator expression captured the VARIABLE x, not its VALUE 10.
# When consumed, it reads x's current value: 1000.
```

This is the same late-binding behavior as closures — see [[Python Iterator Gotchas#Gotcha 5 Generator Expressions Bind Late — The Loop Variable Trap|Iterator Gotchas: Late Binding]].

```python
# The for-variable (i) is NOT late-bound — it's fresh each iteration.
# Only FREE variables (x) are looked up at consumption time.

# More dramatic example:
multipliers = []
for m in range(4):
    multipliers.append(n * m for n in [1, 2, 3])

# All generators captured the variable `m`, not its value at creation time.
# By now, m == 3 for all of them:
for gen in multipliers:
    print(list(gen))
# [3, 6, 9]  ← all use m=3
# [3, 6, 9]
# [3, 6, 9]
# [3, 6, 9]
```

### GOTCHA 4: Passing a Generator to Multiple Arguments Doesn't Duplicate It

```python
gen = (n for n in [1, 2, 3])

# You might think this compares the sequence to itself:
print(list(zip(gen, gen)))
# [(1, 2), (3,)] — NOT [(1, 1), (2, 2), (3, 3)]!

# zip pulls from the SAME iterator for both arguments.
# First call: zip asks left for next → gets 1, asks right for next → gets 2 → (1, 2)
# Second call: left → 3, right → StopIteration → done.

# Compare with a list (multi-pass iterable):
lst = [1, 2, 3]
print(list(zip(lst, lst)))
# [(1, 1), (2, 2), (3, 3)] — each argument gets its own independent iterator
```

### GOTCHA 5: `bool()` and `if` Don't Tell You If a Generator Has Items

```python
gen = (n for n in [])  # empty generator

# This is ALWAYS truthy — even for an empty generator:
print(bool(gen))  # True!
if gen:
    print("This prints!")  # prints! Even though gen has zero items.

# Why? A generator OBJECT always exists (it's not None).
# bool() checks if the object is truthy, not if it has items.
# The generator hasn't been consumed yet, so Python can't know it's empty
# without actually calling next() — which would consume an item.

# To check if a generator has items, you must consume:
gen = (n for n in [])
first = next(gen, None)  # None means empty
if first is not None:
    print(f"Has items, first is {first}")
else:
    print("Empty!")  # prints
```

---

## Chaining Generator Expressions: Lazy Pipelines

One of the most powerful patterns — chain multiple generator expressions to build a processing pipeline that computes everything lazily:

```python
# Process a large dataset without ever loading it all into memory:
raw_data: list[str] = ["  Alice,85  ", "  Bob,92  ", "  Carol,78  ", "  Dave,95  "]

# Each step is a generator — nothing computed until the final consumption
stripped   = (line.strip() for line in raw_data)            # lazy
parsed     = (line.split(",") for line in stripped)         # lazy
records    = ((name, int(score)) for name, score in parsed) # lazy
high_scores = (
    (name, score)
    for name, score in records
    if score >= 90
)                                                           # still lazy!

# NOW consume — all four generators execute in lockstep, one item at a time:
for name, score in high_scores:
    print(f"{name}: {score}")
# Bob: 92
# Dave: 95

# At no point was more than ONE item in memory across the entire pipeline.
```

### How the Pipeline Executes

```python
# The chain is: high_scores → records → parsed → stripped → raw_data
#
# When we call next(high_scores):
#   high_scores asks records for next item
#     records asks parsed for next item
#       parsed asks stripped for next item
#         stripped asks raw_data for next item
#         stripped yields "Alice,85"
#       parsed yields ["Alice", "85"]
#     records yields ("Alice", 85)
#   high_scores checks: 85 >= 90? No → skip, pull next
#     (whole chain fires again for Bob)
#   high_scores checks: 92 >= 90? Yes → yield ("Bob", 92)
#
# It's like a conveyor belt: each item flows through ALL stages
# before the next item enters. One item in flight at a time.
```

---

## The Decision Flowchart: `[...]` vs `(...)`

```python
# ┌─────────────────────────────────────────────────────────────┐
# │  Do I need to iterate the result MORE THAN ONCE?            │
# │                                                             │
# │  YES → Use LIST COMPREHENSION  [expr for x in iter]        │
# │        (Also if you need len, indexing, slicing)            │
# │                                                             │
# │  NO  → Am I passing to a single consuming function?         │
# │        (sum, max, min, any, all, "".join, sorted, ...)     │
# │                                                             │
# │        YES → Use GENERATOR EXPRESSION  (expr for x in iter)│
# │              Or drop the parens: sum(x for x in iter)       │
# │                                                             │
# │        NO  → Is the data too large to fit in memory?        │
# │                                                             │
# │              YES → Use GENERATOR EXPRESSION (must)          │
# │              NO  → Use LIST COMPREHENSION (safer)           │
# └─────────────────────────────────────────────────────────────┘

# Examples of each decision:

# "I need len() later" → list
results = [process(x) for x in data]
print(f"Processed {len(results)} items")

# "Just passing to sum()" → generator
total = sum(x.price for x in orders)

# "10 billion rows" → generator (no choice)
total = sum(parse_row(line) for line in open("huge_file.csv"))

# "Small data, only iterating once, but I might add len() later" → list (safer)
filtered = [x for x in items if x.active]
```

---

## Common Patterns with Generator Expressions

```python
from typing import Iterable

# ─── Pattern 1: Conditional aggregation ───
scores: list[int] = [85, 92, 78, 95, 88, 72, 91]
passing_avg = sum(s for s in scores if s >= 80) / sum(1 for s in scores if s >= 80)
print(f"Average passing score: {passing_avg:.1f}")  # 90.2

# ─── Pattern 2: String building ───
words: list[str] = ["hello", "world", "python"]
csv_line = ",".join(w.upper() for w in words)
print(csv_line)  # HELLO,WORLD,PYTHON

# ─── Pattern 3: any/all with conditions (short-circuit!) ───
numbers: list[int] = [2, 4, 6, 8, 10]
all_even = all(n % 2 == 0 for n in numbers)  # True — stops at first False
has_big  = any(n > 100 for n in numbers)     # False — must check all

# ─── Pattern 4: Finding first match ───
names: list[str] = ["alice", "bob", "carol"]
first_long = next((n for n in names if len(n) > 3), None)
print(first_long)  # "alice" — stops after first match, doesn't scan rest

# ─── Pattern 5: Flattening with a generator ───
matrix: list[list[int]] = [[1, 2], [3, 4], [5, 6]]
flat = list(val for row in matrix for val in row)
print(flat)  # [1, 2, 3, 4, 5, 6]

# ─── Pattern 6: Enumerate with generator ───
data: list[str] = ["a", "b", "c"]
indexed = list((i, v.upper()) for i, v in enumerate(data))
print(indexed)  # [(0, 'A'), (1, 'B'), (2, 'C')]
```

---

## Summary: One Character, Two Worlds

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# ┌──────────────────────────────────────────────────────────────┐
# │           LIST COMPREHENSION          GENERATOR EXPRESSION   │
# │           [n**2 for n in numbers]     (n**2 for n in numbers)│
# ├──────────────────────────────────────────────────────────────┤
# │ Type      list                        generator              │
# │ Eagerness EAGER (all values now)      LAZY (on demand)       │
# │ Memory    O(n) — stores everything    O(1) — one at a time   │
# │ Passes    Multi-pass (reusable)       Single-pass (one shot) │
# │ Indexing  lst[i] ✓                    gen[i] ✗               │
# │ len()     len(lst) ✓                  len(gen) ✗             │
# │ bool()    bool([]) → False            bool(gen) → ALWAYS True│
# │ iter()    iter(lst) is lst → False    iter(gen) is gen → True│
# │ Protocol  ITERABLE                    ITERATOR               │
# └──────────────────────────────────────────────────────────────┘
```
