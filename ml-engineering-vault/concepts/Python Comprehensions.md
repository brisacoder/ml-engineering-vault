---
type: concept
title: "Python Comprehensions"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Comprehensions]]"
  - "[[Python Custom Iterables]]"
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Iteration Deep Dive]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Comprehensions

> Comprehensions are Python's declarative syntax for building collections from iterables. They replace the "create empty container → loop → append" pattern with a single expression. Under the hood, they use the exact same [[Python Iterator Protocol Rules|iterator protocol]] as `for` loops. Based on Python Morsels (PyCon 2026).

---

## Python's `for` Loop Is a `for each` Loop

### The Key Insight

Python's `for` loop is fundamentally different from C/Java/JS-style `for` loops. There is no index variable, no length check, no incrementing. Python iterates over *items*, not indices.

```python
# ─── What Python does ───
numbers: list[int] = [1, 2, 3, 5, 7]
for number in numbers:
    print(number)
# 1, 2, 3, 5, 7
# No index. No len(). No numbers[i]. Just... each item.

# ─── What C/Java/JS programmers write (DON'T do this in Python) ───
# This works but it's unpythonic — you're fighting the language:
for i in range(len(numbers)):    # anti-pattern!
    print(numbers[i])

# ─── If you need the index AND the value, use enumerate ───
for i, number in enumerate(numbers):
    print(f"Index {i}: {number}")
# Index 0: 1
# Index 1: 2
# ...
```

### Why This Matters for Comprehensions

Because `for` iterates over items (not indices), comprehensions naturally express *transformations* and *filters* on items. You think in terms of "what do I want to do to each item?" — not "how do I access position i?"

---

## The List Comprehension: Anatomy

### The Pattern It Replaces

Every list comprehension replaces this exact pattern:

```python
my_favorite_numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# ─── The loop-and-append pattern ───
doubled_numbers: list[int] = []            # 1. Create empty list
for n in my_favorite_numbers:              # 2. Loop over iterable
    doubled_numbers.append(n * 2)          # 3. Append transformed item

print(doubled_numbers)  # [4, 2, 6, 8, 14, 22, 36]
```

### The Comprehension Equivalent

```python
# ─── One line ───
doubled_numbers: list[int] = [n * 2 for n in my_favorite_numbers]
print(doubled_numbers)  # [4, 2, 6, 8, 14, 22, 36]
```

### The Anatomy: Three Parts

```python
# A list comprehension has exactly this structure:
#
#   [ expression   for variable in iterable ]
#     ──────────   ────────────────────────
#     │            │
#     │            └─ The loop: iterates over any iterable
#     │               (list, range, generator, dict, file, ...)
#     │
#     └─ The transform: applied to each item
#        This becomes one element in the output list.

# The mapping from loop-and-append is mechanical:
#
#   result = []                          ┐
#   for variable in iterable:            ├─→  [expression for variable in iterable]
#       result.append(expression)        ┘
```

### Multi-Line Formatting (Preferred for Readability)

```python
# For anything non-trivial, break it across lines.
# The convention: one clause per line, indented.
doubled_numbers: list[int] = [
    n * 2                           # expression (what to produce)
    for n in my_favorite_numbers    # loop (where to get items)
]

# This is the SAME comprehension — just easier to read, review, and modify.
print(doubled_numbers)  # [4, 2, 6, 8, 14, 22, 36]
```

---

## Filtering with `if`

### The Pattern It Replaces

```python
names: list[str] = ["Alice", "Bob", "Christy", "Jules"]

# ─── Loop-and-append with a condition ───
vowel_names: list[str] = []
for name in names:
    if name[0].lower() in "aeiou":      # filter condition
        vowel_names.append(name)

print(vowel_names)  # ['Alice']
```

### The Comprehension Equivalent

```python
# Add an `if` clause after the `for`:
#
#   [ expression   for variable in iterable   if condition ]
#     ──────────   ────────────────────────   ────────────
#     │            │                          │
#     │            └─ The loop                └─ The filter: only items where
#     │                                          condition is True make it in
#     └─ The transform

vowel_names: list[str] = [
    name                                    # expression: keep name as-is
    for name in names                       # loop over names
    if name[0].lower() in "aeiou"           # filter: first char is a vowel
]
print(vowel_names)  # ['Alice']
```

### GOTCHA: `if` Position Matters

```python
numbers: list[int] = [1, 2, 3, 4, 5, 6]

# ─── if AFTER for = FILTER (exclude items) ───
evens: list[int] = [n for n in numbers if n % 2 == 0]
print(evens)  # [2, 4, 6] — only even numbers kept

# ─── if BEFORE for = CONDITIONAL EXPRESSION (ternary) ───
labels: list[str] = ["even" if n % 2 == 0 else "odd" for n in numbers]
print(labels)  # ['odd', 'even', 'odd', 'even', 'odd', 'even'] — ALL items, just labeled

# These are completely different!
# Filter:     [expr for x in iter if condition]        → fewer items
# Ternary:    [expr_if_true if cond else expr_if_false for x in iter] → same # of items
```

---

## Nested Loops in Comprehensions

### The Pattern It Replaces

```python
matrix: list[list[int]] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
]

# ─── Nested loop to flatten ───
flat: list[int] = []
for row in matrix:           # outer loop
    for item in row:         # inner loop
        flat.append(item)

print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### The Comprehension Equivalent

```python
# Nested for clauses — ORDER MATCHES the loop nesting (left-to-right = outer-to-inner)
flat: list[int] = [
    item                    # expression: each individual value
    for row in matrix       # outer loop (comes FIRST)
    for item in row         # inner loop (comes SECOND)
]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

### GOTCHA: Reading Order Is Counterintuitive

```python
# The common mistake: people try to read comprehensions as
# "item for (item in row) for (row in matrix)" — inner first.
#
# But the correct reading is TOP-DOWN, matching the loop nesting:
#
#   for row in matrix:        ←→   for row in matrix
#       for item in row:      ←→       for item in row
#           flat.append(item) ←→           item
#
# The EXPRESSION (item) comes first in the comprehension,
# but the FOR clauses maintain their natural nesting order.

# Proof — this is the same:
flat_loop: list[int] = []
for row in matrix:
    for item in row:
        flat_loop.append(item)

flat_comp: list[int] = [item for row in matrix for item in row]

assert flat_loop == flat_comp  # True — identical output
```

### Nested Loops with a Filter

```python
# You can add `if` after any `for` clause:
matrix: list[list[int]] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Flatten, but only keep even numbers
even_flat: list[int] = [
    item
    for row in matrix
    for item in row
    if item % 2 == 0        # filter applies to the inner loop variable
]
print(even_flat)  # [2, 4, 6, 8]
```

---

## Set and Dict Comprehensions

The same syntax works for sets and dicts — just change the brackets:

```python
words: list[str] = ["hello", "world", "hello", "python"]

# ─── Set comprehension: {expression for var in iterable} ───
unique_lengths: set[int] = {len(w) for w in words}
print(unique_lengths)  # {5, 6} — duplicates removed (both "hello" have len 5)

# ─── Dict comprehension: {key: value for var in iterable} ───
word_lengths: dict[str, int] = {w: len(w) for w in words}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}
# Note: duplicate key 'hello' is overwritten (last one wins)

# ─── Dict comprehension with filter ───
long_words: dict[str, int] = {
    w: len(w)
    for w in words
    if len(w) > 5
}
print(long_words)  # {'python': 6}
```

### GOTCHA: There Is No Tuple Comprehension

```python
# This is NOT a tuple comprehension — it's a GENERATOR EXPRESSION:
result = (n * 2 for n in range(5))
print(type(result))  # <class 'generator'> — NOT tuple!

# To make a tuple, wrap in tuple():
result_tuple: tuple[int, ...] = tuple(n * 2 for n in range(5))
print(result_tuple)  # (0, 2, 4, 6, 8)

# See [[Python Iterator Protocol Rules]] for why generators matter here.
```

---

## From Comprehensions to Generator Expressions

List comprehensions build the entire result list in memory. Generator expressions use the same syntax but produce values **lazily** — one at a time, on demand.

```python
import sys

numbers: list[int] = list(range(1_000_000))

# ─── List comprehension: builds entire list in memory ───
squares_list = [n ** 2 for n in numbers]
print(sys.getsizeof(squares_list))  # ~8 MB

# ─── Generator expression: produces values lazily ───
squares_gen = (n ** 2 for n in numbers)
print(sys.getsizeof(squares_gen))   # ~200 bytes (just the generator object)

# The generator hasn't computed anything yet!
# Values are produced one-at-a-time when consumed:
print(next(squares_gen))  # 0 — first value computed NOW
print(next(squares_gen))  # 1 — second value computed NOW

# Passing a generator to sum() is memory-efficient:
# sum() pulls one value at a time, never stores the full sequence.
total = sum(n ** 2 for n in range(1_000_000))  # no brackets = generator expression
print(total)  # 333332833333500000
```

### GOTCHA: Generator Expressions Are Single-Pass (Iterators!)

```python
# A generator expression is an ITERATOR — see [[Python Iterator Gotchas#Gotcha 2]]
gen = (n * 2 for n in [1, 2, 3])

first = list(gen)   # [2, 4, 6]
second = list(gen)  # [] ← exhausted!

# A list comprehension is a LIST — multi-pass, safe to reuse
lst = [n * 2 for n in [1, 2, 3]]

first = list(lst)   # [2, 4, 6]
second = list(lst)  # [2, 4, 6] ← works every time
```

### When to Use Which

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │  USE LIST COMPREHENSION  [...]  when:                           │
# │  - You need to iterate the result multiple times                │
# │  - You need len(), indexing, or slicing                         │
# │  - The result is small enough to fit in memory                  │
# │  - You're building a data structure to keep                     │
# │                                                                 │
# │  USE GENERATOR EXPRESSION  (...)  when:                         │
# │  - You're passing directly to a function: sum(), max(), join()  │
# │  - The input is huge and you only need one pass                 │
# │  - You're chaining transformations (pipeline)                   │
# │  - Memory matters more than speed                               │
# └─────────────────────────────────────────────────────────────────┘

# Example: sum of squares — generator is perfect (one pass, no need to store)
total = sum(n ** 2 for n in range(10))  # generator — O(1) memory

# Example: need to check length AND iterate — list is needed
results = [process(item) for item in data]
print(f"Got {len(results)} results")   # need len() → must be a list
for r in results:
    save(r)
```

---

## Comprehension Gotchas

### GOTCHA: Variable Leaking (Python 2 Legacy — Fixed in Python 3)

```python
# In Python 2, comprehension variables leaked into the enclosing scope.
# In Python 3, they DON'T:
x = "before"
squares = [x for x in range(5)]  # 'x' is scoped to the comprehension
print(x)  # "before" — NOT 4! (would be 4 in Python 2)

# But NOTE: for-loop variables DO still leak:
for x in range(5):
    pass
print(x)  # 4 — for-loop variable leaks into enclosing scope
```

### GOTCHA: Don't Overuse Comprehensions

```python
# BAD: Comprehension for side effects (don't do this)
[print(x) for x in range(5)]  # "works" but creates a useless list of Nones

# GOOD: Use a for-loop when you have side effects
for x in range(5):
    print(x)

# BAD: Comprehension so complex it's unreadable
result = [
    transform(item)
    for group in data
    if group.active
    for item in group.items
    if item.score > threshold
    if not item.excluded
]

# GOOD: Break it into steps or use a generator function
def filtered_items(data, threshold):
    for group in data:
        if not group.active:
            continue
        for item in group.items:
            if item.score > threshold and not item.excluded:
                yield transform(item)

result = list(filtered_items(data, threshold))
```

### GOTCHA: Comprehensions with Walrus Operator (Python 3.8+)

```python
# The walrus operator := lets you compute something once and reuse it:
data: list[str] = ["hello", "hi", "hey", "howdy", "greetings"]

# Without walrus — computes len(w) twice:
long_with_lengths = [(w, len(w)) for w in data if len(w) > 3]

# With walrus — computes len(w) once, reuses:
long_with_lengths = [
    (w, length)
    for w in data
    if (length := len(w)) > 3   # assign AND test in one shot
]
print(long_with_lengths)
# [('hello', 5), ('howdy', 5), ('greetings', 9)]

# GOTCHA: the walrus variable leaks to the enclosing scope!
print(length)  # 9 — unlike comprehension variables, walrus leaks
```

---

## Exercises and Solutions

### Exercise 1: `get_vowel_names` — Filter Names Starting with a Vowel

```python
def get_vowel_names(names: list[str]) -> list[str]:
    """Return names that start with a vowel (case-insensitive).
    
    This is a pure FILTER — no transformation, just selection.
    The comprehension has an expression (name) and a filter (if ...).
    
    >>> get_vowel_names(["Alice", "Bob", "Christy", "Jules"])
    ['Alice']
    >>> get_vowel_names(["Scott", "Arthur", "Jan", "elizabeth"])
    ['Arthur', 'elizabeth']
    """
    # Step 1 (mental model): write the for-loop version first
    # result = []
    # for name in names:
    #     if name[0].lower() in "aeiou":
    #         result.append(name)
    # return result

    # Step 2: mechanically translate to comprehension
    return [
        name                                # expression: keep the name unchanged
        for name in names                   # loop over all names
        if name[0].lower() in "aeiou"       # filter: first letter is a vowel
    ]

# Tests
assert get_vowel_names(["Alice", "Bob", "Christy", "Jules"]) == ["Alice"]
assert get_vowel_names(["Scott", "Arthur", "Jan", "elizabeth"]) == ["Arthur", "elizabeth"]
assert get_vowel_names([]) == []
assert get_vowel_names(["Eve", "Iris", "Uma"]) == ["Eve", "Iris", "Uma"]  # all vowels
assert get_vowel_names(["Bob", "Carl"]) == []  # no vowels
print("All get_vowel_names() tests passed")
```

### Exercise 2: `flatten` — Flatten a Matrix (List of Lists)

```python
def flatten(matrix: list[list[int]]) -> list[int]:
    """Flatten a list of lists into a single list.
    
    Uses a nested comprehension. Remember: for-clause order matches
    the nesting order of the equivalent for-loops (outer first).
    
    >>> flatten([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # Step 1: the nested for-loop version
    # result = []
    # for row in matrix:           # outer loop
    #     for item in row:         # inner loop
    #         result.append(item)
    # return result

    # Step 2: translate — for clauses keep the same order
    return [
        item                       # expression: each individual value
        for row in matrix          # outer loop → first for clause
        for item in row            # inner loop → second for clause
    ]

# Tests
matrix = [[row * 3 + incr for incr in range(1, 4)] for row in range(4)]
assert matrix == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
assert flatten(matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
assert flatten([]) == []                    # empty matrix
assert flatten([[], [], []]) == []          # matrix of empty rows
assert flatten([[1]]) == [1]                # single element
print("All flatten() tests passed")
```

### Exercise 3: `words_containing` — Filter Words by Letter

```python
def words_containing(words: list[str], letter: str) -> list[str]:
    """Return words that contain the given letter (case-insensitive).
    
    This is a filter comprehension — the `in` operator on strings
    checks for substring membership.
    
    >>> words_containing(["Explicit", "is", "better", "than", "implicit"], "b")
    ['better']
    >>> words_containing(["Explicit", "is", "better", "than", "implicit"], "i")
    ['Explicit', 'is', 'implicit']
    """
    # Note: the check is case-insensitive — lowercase both sides
    return [
        word                                    # expression: keep the word as-is
        for word in words                       # loop over all words
        if letter.lower() in word.lower()       # filter: letter appears in word
    ]

# Tests
zen: list[str] = ["Explicit", "is", "better", "than", "implicit"]
assert words_containing(zen, "b") == ["better"]
assert words_containing(zen, "i") == ["Explicit", "is", "implicit"]
assert words_containing(zen, "x") == ["Explicit"]          # capital X matched
assert words_containing(zen, "z") == []                    # no matches
assert words_containing([], "a") == []                     # empty input
assert words_containing(["AAA", "aaa"], "a") == ["AAA", "aaa"]  # case-insensitive
print("All words_containing() tests passed")
```

---

## The Comprehension → Loop Translation Table

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │ COMPREHENSION                  │ EQUIVALENT LOOP                │
# ├────────────────────────────────┼────────────────────────────────┤
# │ [expr for x in iter]           │ for x in iter:                 │
# │                                │     result.append(expr)        │
# ├────────────────────────────────┼────────────────────────────────┤
# │ [expr for x in iter if cond]   │ for x in iter:                 │
# │                                │     if cond:                   │
# │                                │         result.append(expr)    │
# ├────────────────────────────────┼────────────────────────────────┤
# │ [expr for x in A for y in B]   │ for x in A:                    │
# │                                │     for y in B:                │
# │                                │         result.append(expr)    │
# ├────────────────────────────────┼────────────────────────────────┤
# │ {expr for x in iter}           │ (same but result = set())      │
# ├────────────────────────────────┼────────────────────────────────┤
# │ {k: v for x in iter}           │ (same but result = {},         │
# │                                │  result[k] = v)                │
# ├────────────────────────────────┼────────────────────────────────┤
# │ (expr for x in iter)           │ generator — lazy, single-pass  │
# │                                │ see [[Python Iterator Gotchas]] │
# └─────────────────────────────────────────────────────────────────┘
```

---

## Bonus: Comprehensions Use the Iterator Protocol

Under the hood, every comprehension calls `iter()` on the iterable, then `next()` repeatedly — exactly like a `for` loop (see [[Python Iterator Protocol Rules#How `for` Loops Work Desugaring Step by Step|for-loop desugaring]]).

```python
# This means comprehensions work on ANY iterable:
from typing import Iterator

def fibonacci(n: int) -> Iterator[int]:
    """Generator that yields first n Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Comprehension over a generator — works because generators are iterable
fib_squares: list[int] = [f ** 2 for f in fibonacci(8)]
print(fib_squares)  # [0, 1, 1, 4, 9, 25, 64, 169]

# Comprehension over a file — files are iterators (line by line)
# [line.strip() for line in open("data.txt")]

# Comprehension over dict.items() — view object, iterable
prices: dict[str, float] = {"apple": 1.50, "banana": 0.75, "cherry": 3.00}
expensive: dict[str, float] = {
    fruit: price
    for fruit, price in prices.items()
    if price > 1.00
}
print(expensive)  # {'apple': 1.5, 'cherry': 3.0}

# GOTCHA: comprehension over a GENERATOR exhausts it (single-pass!)
gen = fibonacci(5)
first = [f for f in gen]   # [0, 1, 1, 2, 3]
second = [f for f in gen]  # [] ← generator exhausted by first comprehension
```
