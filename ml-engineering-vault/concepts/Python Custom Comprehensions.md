---
type: concept
title: "Python Custom Comprehensions"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Custom Iterables]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Custom Comprehensions

> Python gives you list, set, and dict comprehensions built-in. But there's no `Counter` comprehension, no `frozenset` comprehension, no `deque` comprehension. Turns out you don't need one — **any function or class that accepts an iterable can be paired with a [[Python Generator Expressions|generator expression]] to create a comprehension-like pattern**. This is one of the most powerful compositional ideas in Python. Based on Python Morsels (PyCon 2026).

---

## The Core Insight

```python
# Python has these built-in comprehensions:
list_comp   = [x**2 for x in range(5)]     # list comprehension
set_comp    = {x**2 for x in range(5)}     # set comprehension
dict_comp   = {x: x**2 for x in range(5)} # dict comprehension

# But there's no syntax for these:
# tuple_comp  = ???
# counter_comp = ???
# frozenset_comp = ???
# deque_comp = ???

# The trick: generator expression + iterable-accepting function = custom comprehension
#
#   function_or_class( expression for var in iterable )
#                      ─────────────────────────────────
#                      this is a generator expression
#
# The function consumes the generator lazily. Same effect as a comprehension,
# same memory efficiency, just different brackets.
```

---

## Reducers: Collapsing an Iterable into a Single Value

Reducers consume an entire iterable and produce one result. Pair them with a generator expression and you get a "comprehension" that answers a question about your data.

### `any()` and `all()` — Boolean Questions

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# "Are ALL numbers positive?" — like a universal quantifier (∀)
print(all(n > 0 for n in numbers))   # True
# all() short-circuits: stops at the first False.
# On a generator, this means it won't even compute the rest.

# "Is ANY number greater than 20?" — like an existential quantifier (∃)
print(any(n > 20 for n in numbers))  # False
# any() short-circuits: stops at the first True.

# ─── Think of these as comprehension patterns ───
# all(condition for x in data)  ≈  "every x satisfies condition"
# any(condition for x in data)  ≈  "at least one x satisfies condition"
```

#### GOTCHA: Short-Circuiting Only Works with Generators

```python
# With a generator expression — short-circuits, lazy:
result = any(n > 3 for n in range(1_000_000_000))
# Finds 4 almost immediately, stops. Never looks at the other 999,999,995 items.

# With a list comprehension — computes ALL values first, THEN checks:
# result = any([n > 3 for n in range(1_000_000_000)])
# This builds a 1-billion-item list in memory BEFORE any() even starts.
# Don't do this.
```

### `sum()` — Numeric Aggregation

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# Sum of squares — no intermediate list needed:
total = sum(n**2 for n in numbers)
print(total)  # 524

# Sum with a filter:
even_sum = sum(n for n in numbers if n % 2 == 0)
print(even_sum)  # 24  (2 + 4 + 18)

# Counting items that match a condition (True == 1, False == 0):
count_big = sum(1 for n in numbers if n > 5)
print(count_big)  # 2  (11 and 18)

# This is a "count comprehension" — there's no built-in syntax for it,
# but sum(1 for ...) is the idiomatic way.
```

### `min()` and `max()` — Extremes

```python
words: list[str] = ["Python", "is", "a", "beautiful", "language"]

# Longest word length:
longest = max(len(w) for w in words)
print(longest)  # 9  ("beautiful")

# Shortest word:
shortest_word = min(words, key=len)
print(shortest_word)  # "a"

# NOTE: min/max with a genexpr vs min/max with key= are different patterns:
#   max(len(w) for w in words)  → returns the LENGTH (an int)
#   max(words, key=len)         → returns the WORD (a str)
```

### `str.join()` — String "Comprehension"

```python
numbers: list[int] = [2, 1, 3, 4, 7, 11, 18]

# join() needs an iterable of strings — pair it with a generator:
csv_line = ", ".join(str(n) for n in numbers)
print(csv_line)  # '2, 1, 3, 4, 7, 11, 18'

# This is effectively a "string comprehension":
#   separator.join(transform(x) for x in iterable)
#
# More examples:
names: list[str] = ["alice", "bob", "carol"]
titled = " | ".join(name.title() for name in names)
print(titled)  # 'Alice | Bob | Carol'

# HTML list items:
items = "\n".join(f"  <li>{name}</li>" for name in names)
print(items)
#   <li>alice</li>
#   <li>bob</li>
#   <li>carol</li>
```

#### GOTCHA: `join()` Only Accepts Strings

```python
numbers = [1, 2, 3]

# This crashes — join needs strings, not ints:
try:
    ", ".join(numbers)
except TypeError as e:
    print(e)  # sequence item 0: expected str instance, int found

# Fix: transform to strings in the generator expression
print(", ".join(str(n) for n in numbers))  # '1, 2, 3'
```

---

## Iterable-Accepting Constructors: Build Any Collection

### `tuple()` — The Missing Tuple Comprehension

```python
# Python has no tuple comprehension syntax.
# (expr for x in iter) is a GENERATOR EXPRESSION, not a tuple!
# See [[Python Generator Expressions#GOTCHA There Is No Tuple Comprehension]]

# But tuple() + generator expression = tuple comprehension:
squares: tuple[int, ...] = tuple(n**2 for n in range(5))
print(squares)       # (0, 1, 4, 9, 16)
print(type(squares)) # <class 'tuple'>
```

### `frozenset()` — Immutable Set Comprehension

```python
# No frozenset comprehension syntax, but:
text = "hello world hello python world"
unique_words: frozenset[str] = frozenset(
    w.lower() for w in text.split()
)
print(unique_words)    # frozenset({'hello', 'python', 'world'})
print("hello" in unique_words)  # True

# Compare with the set comprehension (mutable):
mutable_set: set[str] = {w.lower() for w in text.split()}
# frozenset is hashable — can be used as a dict key or set element.
# Regular sets cannot.
```

### `collections.Counter` — Frequency Comprehension

```python
from collections import Counter

bridge: str = """
If you read the album cover by now
You know that my name is what my name is
When I came in here to try and
Do this, something I've never done before
Mr. Jones, Booker T., said to me
Don't worry about it
Just do what you do
And do it good
"""

# Counter + generator = word frequency "comprehension"
# The generator handles the cleaning (lowercase, strip punctuation).
# Counter handles the counting.
words: Counter[str] = Counter(
    w.strip(".,!?\"'")          # expression: clean each word
    for w in bridge.lower().split()  # loop: split into words
)

print(words.most_common(5))
# [('do', 4), ('you', 3), ('my', 2), ('name', 2), ('is', 2)]

# Without the generator, you'd need a loop:
# words = Counter()
# for w in bridge.lower().split():
#     words[w.strip(".,!?\"'")] += 1
# The generator version is one expression.
```

### `collections.deque` — Deque Comprehension

```python
from collections import deque

# Fixed-size sliding window of recent squares:
recent: deque[int] = deque(
    (n**2 for n in range(20)),
    maxlen=5
)
print(recent)  # deque([256, 289, 324, 361], maxlen=5)
# Only the last 5 values are kept — deque discards older ones.
```

### `dict()` with Generator of Tuples

```python
# dict comprehension: {k: v for ...}
# But you can also build a dict from an iterable of (key, value) pairs:
names: list[str] = ["alice", "bob", "carol"]

name_lengths: dict[str, int] = dict(
    (name, len(name))       # each item is a (key, value) tuple
    for name in names
)
print(name_lengths)  # {'alice': 5, 'bob': 3, 'carol': 5}

# When is this useful over a dict comprehension?
# When you're transforming an existing iterable of pairs:
raw_pairs = [("  Alice  ", "  25  "), ("  Bob  ", "  30  ")]
clean: dict[str, int] = dict(
    (k.strip(), int(v.strip()))
    for k, v in raw_pairs
)
print(clean)  # {'Alice': 25, 'Bob': 30}
```

---

## The General Pattern

```python
# ┌─────────────────────────────────────────────────────────────────┐
# │  BUILT-IN COMPREHENSION          CUSTOM COMPREHENSION           │
# ├─────────────────────────────────────────────────────────────────┤
# │  [expr for x in iter]            list(expr for x in iter)      │
# │  {expr for x in iter}            set(expr for x in iter)       │
# │  {k: v for x in iter}            dict((k,v) for x in iter)    │
# │                                                                 │
# │  (no syntax)                     tuple(expr for x in iter)     │
# │  (no syntax)                     frozenset(expr for x in iter) │
# │  (no syntax)                     deque(expr for x in iter)     │
# │  (no syntax)                     Counter(expr for x in iter)   │
# │  (no syntax)                     bytes(expr for x in iter)     │
# │  (no syntax)                     bytearray(expr for x in iter) │
# │                                                                 │
# │  (no syntax)                     sum(expr for x in iter)       │
# │  (no syntax)                     any(cond for x in iter)       │
# │  (no syntax)                     all(cond for x in iter)       │
# │  (no syntax)                     min(expr for x in iter)       │
# │  (no syntax)                     max(expr for x in iter)       │
# │  (no syntax)                     ", ".join(expr for x in iter) │
# │  (no syntax)                     sorted(expr for x in iter)    │
# └─────────────────────────────────────────────────────────────────┘
#
# The LEFT column is syntactic sugar. The RIGHT column shows that
# generator expressions make the sugar unnecessary — any iterable-
# accepting callable can act as a comprehension target.
```

---

## Why This Works: The Iterator Protocol

The reason all of this composes so cleanly is the [[Python Iterator Protocol Rules|iterator protocol]]. Every function/class above does the same thing internally:

```python
# When you write:
total = sum(n**2 for n in [1, 2, 3])

# Python does:
# 1. Creates a generator object from the expression
# 2. Passes it to sum()
# 3. sum() calls iter(generator) → gets the generator back (it's already an iterator)
# 4. sum() calls next(generator) → gets 1, adds to accumulator
# 5. sum() calls next(generator) → gets 4, adds to accumulator
# 6. sum() calls next(generator) → gets 9, adds to accumulator
# 7. sum() calls next(generator) → StopIteration → return accumulator (14)
#
# At no point was a list of [1, 4, 9] created.
# Each value was computed, consumed, and discarded — O(1) memory.
```

---

## Composing Multiple Patterns

The real power shows when you chain these patterns:

```python
from collections import Counter
from typing import Iterable

data: list[dict[str, str | int]] = [
    {"name": "Alice", "dept": "Engineering", "salary": 120_000},
    {"name": "Bob", "dept": "Engineering", "salary": 95_000},
    {"name": "Carol", "dept": "Marketing", "salary": 85_000},
    {"name": "Dave", "dept": "Marketing", "salary": 90_000},
    {"name": "Eve", "dept": "Engineering", "salary": 110_000},
]

# "What's the average salary of engineers?"
# → filter (genexpr with if) + reducer (sum) + count
engineers = [e for e in data if e["dept"] == "Engineering"]
avg_salary = sum(e["salary"] for e in engineers) / len(engineers)
print(f"Avg engineering salary: ${avg_salary:,.0f}")  # $108,333

# "Which department has the most people?"
# → Counter "comprehension" + most_common
dept_counts = Counter(e["dept"] for e in data)
print(dept_counts.most_common(1))  # [('Engineering', 3)]

# "Are all salaries above 80k?"
# → all() "comprehension"
print(all(e["salary"] > 80_000 for e in data))  # True

# "Comma-separated list of names in marketing?"
# → join "comprehension" with filter
marketing_names = ", ".join(
    e["name"]
    for e in data
    if e["dept"] == "Marketing"
)
print(marketing_names)  # 'Carol, Dave'

# "Sorted unique departments?"
# → sorted + set "comprehension"
depts = sorted(e["dept"] for e in data)
print(depts)  # ['Engineering', 'Engineering', 'Engineering', 'Marketing', 'Marketing']

unique_depts = sorted({e["dept"] for e in data})
print(unique_depts)  # ['Engineering', 'Marketing']
```

---

## Gotchas

### GOTCHA: `sum()` on an Exhausted Generator Returns 0

```python
gen = (n**2 for n in [1, 2, 3])
first = sum(gen)    # 14 — correct
second = sum(gen)   # 0  — WRONG, silent failure!

# sum() of an empty iterable is 0 (its start value).
# An exhausted generator IS an empty iterable.
# No error, no warning. See [[Python Iterator Gotchas#Gotcha 7]].
```

### GOTCHA: `sorted()` Returns a List, Not a Generator

```python
gen = (n**2 for n in [3, 1, 2])

# sorted() must see ALL values to sort them → it materializes internally.
# The return value is a plain list, not a generator.
result = sorted(gen)
print(type(result))  # <class 'list'>
print(result)        # [1, 4, 9]

# So using a generator expression with sorted() doesn't save memory —
# sorted() builds the full list anyway. Use it for the SYNTAX convenience,
# not for memory savings.
```

### GOTCHA: Parentheses Syntax with Multiple Arguments

```python
# When the genexpr is the ONLY argument, outer parens are optional:
total = sum(n**2 for n in range(5))  # clean — parens double up

# When there are OTHER arguments, genexpr needs its OWN parens:
total = sum((n**2 for n in range(5)), start=100)  # extra parens needed
# total = sum(n**2 for n in range(5), start=100)  # SyntaxError!

# Same with max/min:
biggest = max((n**2 for n in range(5)), default=0)  # OK
# biggest = max(n**2 for n in range(5), default=0)  # SyntaxError!
```
