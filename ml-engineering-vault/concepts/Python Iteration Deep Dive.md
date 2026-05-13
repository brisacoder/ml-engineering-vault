---
type: concept
title: "Python Iteration Deep Dive"
libs:
  - general
tags:
  - concept/architecture
related:
  - "[[Python Iteration Protocol]]"
  - "[[Python Iterator Protocol Rules]]"
  - "[[Python Iterator Gotchas]]"
  - "[[Python Comprehensions]]"
  - "[[Python Generator Expressions]]"
  - "[[Python Custom Iterables]]"
created: 2026-05-13
updated: 2026-05-13
---

# Python Iteration Deep Dive

> A thorough walkthrough of how Python's iteration machinery works under the hood — from the `for` loop desugaring to the protocol rules — with exercises that test real understanding.

---

## How `for` Loops Actually Work

### The Illusion

When you write this:

```python
for item in [10, 20, 30]:
    print(item)
```

Python does **not** index into the list with `[0]`, `[1]`, `[2]`. It uses the **iterator protocol**. Here's what actually happens, step by step.

### The Reality: Desugared `for` Loop

```python
def print_each(iterable) -> None:
    """This is exactly what a for loop does under the hood."""

    # Step 1: Get an iterator from the iterable
    # This calls iterable.__iter__()
    iterator = iter(iterable)

    # Step 2: Repeatedly call next() until StopIteration
    while True:
        try:
            item = next(iterator)  # calls iterator.__next__()
        except StopIteration:
            break                  # iterator exhausted → exit loop
        else:
            # Step 3: Execute the loop body with the yielded value
            print(item)

# Works with ANY iterable — lists, sets, dicts, generators, files, custom classes
print_each([1, 2, 3])        # 1, 2, 3
print_each({1, 2, 3})        # order varies (set)
print_each("abc")            # a, b, c
print_each(range(3))         # 0, 1, 2
print_each(x**2 for x in range(3))  # 0, 1, 4
```

### Why This Design?

The two-step protocol (`iter()` then `next()`) decouples **"what can be looped over"** from **"the state of a particular traversal"**:

```python
numbers: list[int] = [10, 20, 30]

# A list is iterable — you can start multiple independent traversals
iter1 = iter(numbers)  # one cursor
iter2 = iter(numbers)  # a completely separate cursor

print(next(iter1))  # 10
print(next(iter1))  # 20
print(next(iter2))  # 10 — iter2 is independent, starts from the beginning

# This is why nested for loops over the same list work:
for i in numbers:        # gets iterator A
    for j in numbers:    # gets iterator B (independent!)
        pass             # both iterate fully, no conflict
```

---

## The Five Rules of the Iterator Protocol

Everything flows from these rules. Internalize them and the gotchas become obvious.

```python
# Rule 1: iter(iterable) → returns an iterator
nums = [1, 2, 3]
it = iter(nums)
print(type(it))  # <class 'list_iterator'>

# Rule 2: next(iterator) → returns the next value
print(next(it))  # 1
print(next(it))  # 2

# Rule 3: next(iterator) raises StopIteration when exhausted
print(next(it))  # 3
try:
    next(it)
except StopIteration:
    print("No more items")  # prints this

# Rule 4: iter(iterator) → returns the SAME iterator (itself)
it2 = iter(nums)
assert iter(it2) is it2  # True — iterators are their own iterables

# Rule 5: Exhaustion is permanent — no reset
assert list(it) == []  # it was exhausted above, stays exhausted
```

### The Hierarchy Visualized

```python
# Everything in Python's iteration world:
#
#   NOT ITERABLE          ITERABLE                    ITERATOR
#   (no __iter__)         (__iter__ returns            (__iter__ returns self,
#                          new iterator)               has __next__)
#
#   int, float,           list, tuple, dict,          list_iterator,
#   None, bool            set, str, range,            generator,
#                         dict_keys, dict_values,      map, filter, zip,
#                         frozenset, bytes             file objects,
#                                                      csv.reader
#
#   can't use in for      can use in for loop         can use in for loop
#   can't call iter()     iter() → new iterator       iter() → itself
#   can't call next()     can't call next()           CAN call next()
#   ─────────────         multi-pass ✓                single-pass only ✗
```

---

## Every Place Python Uses the Iterator Protocol

It's not just `for` loops. The protocol powers *all* of these:

```python
data = [3, 1, 4, 1, 5]

# 1. for loops (obviously)
for x in data:
    pass

# 2. Comprehensions (list, dict, set, generator expressions)
squares = [x**2 for x in data]       # [9, 1, 16, 1, 25]

# 3. Unpacking / multiple assignment
a, b, c, d, e = data                  # a=3, b=1, c=4, d=1, e=5

# 4. Star unpacking
first, *rest = data                    # first=3, rest=[1, 4, 1, 5]

# 5. Function argument unpacking
def add(a, b): return a + b
args = [3, 7]
print(add(*args))                      # 10 — *args iterates the list

# 6. Built-in functions that consume iterables
print(sum(data))                       # 14
print(sorted(data))                    # [1, 1, 3, 4, 5]
print(min(data), max(data))            # 1 5
print(list(reversed(data)))            # [5, 1, 4, 1, 3]
print(any(x > 4 for x in data))       # True
print(all(x > 0 for x in data))       # True

# 7. String joining
letters = ["h", "e", "l", "l", "o"]
print("".join(letters))                # "hello"

# 8. Constructor calls
print(tuple(data))                     # (3, 1, 4, 1, 5)
print(set(data))                       # {1, 3, 4, 5}
print(dict(zip("abc", data)))          # {'a': 3, 'b': 1, 'c': 4}

# 9. `in` membership test (on iterators — calls next() until found)
gen = iter(data)
print(3 in gen)                        # True (consumes up to and including 3)

# 10. collections and itertools
from collections import Counter
from itertools import chain
print(Counter(data))                   # Counter({1: 2, 3: 1, 4: 1, 5: 1})
print(list(chain([1], [2], [3])))      # [1, 2, 3]
```

---

## `generatorify` Is Just a Custom `iter()`

The instructor's `generatorify` function is essentially a hand-rolled `iter()`:

```python
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

def generatorify(iterable: Iterable[T]) -> Iterator[T]:
    """Turn any iterable into a generator (which is an iterator).
    
    This is conceptually what iter() does, except iter() returns
    a type-specific iterator (list_iterator, dict_keyiterator, etc.)
    while this always returns a generator object.
    """
    for item in iterable:  # for-loop calls iter() internally
        yield item

numbers: list[int] = [1, 2, 3, 4, 5]

# generatorify returns a generator — which IS an iterator
g = generatorify(numbers)
print(type(g))          # <class 'generator'>
print(iter(g) is g)     # True — generator is its own iterator

# iter() returns a list_iterator — also an iterator, different type
i = iter(numbers)
print(type(i))          # <class 'list_iterator'>
print(iter(i) is i)     # True — list_iterator is also its own iterator

# Both behave the same way:
print(next(g))  # 1
print(next(i))  # 1
```

---

## `next()` with a Default Value

`next()` accepts an optional second argument — a default value returned instead of raising `StopIteration`. This is one of those things that cleans up a lot of code:

```python
# Without default — must catch StopIteration
it = iter([])
try:
    value = next(it)
except StopIteration:
    value = "nothing here"

# With default — one clean line
it = iter([])
value = next(it, "nothing here")  # "nothing here" — no exception
print(value)

# Practical use: safely get the first item of any iterable
from typing import Iterable, TypeVar

T = TypeVar("T")

def first(iterable: Iterable[T], default: T | None = None) -> T | None:
    """Return first item, or default if empty."""
    return next(iter(iterable), default)

print(first([10, 20, 30]))    # 10
print(first([]))               # None
print(first([], "empty"))      # "empty"
print(first(x for x in []))   # None
```

---

## Exercises and Solutions

### Exercise 1: `first` — Return the First Item of Any Iterable

```python
from typing import Iterable, TypeVar

T = TypeVar("T")

def first(iterable: Iterable[T]) -> T:
    """Return the first item in any iterable.
    
    Key insight: iter() gives us an iterator from ANY iterable,
    then next() gives us the first item from that iterator.
    
    >>> first([1, 2, 3])
    1
    >>> first(iter([10, 20]))
    10
    >>> first("hello")
    'h'
    >>> first(x**2 for x in range(5))
    0
    """
    # iter() handles both iterables (lists) and iterators (generators).
    # If it's already an iterator, iter() returns it unchanged.
    # If it's an iterable, iter() creates a new iterator.
    return next(iter(iterable))

# Test cases
assert first(iter([1, 2])) == 1        # works with iterators
assert first([1, 2]) == 1              # works with iterables
assert first("abc") == "a"             # works with strings
assert first(range(10)) == 0           # works with range
assert first(x**2 for x in [3]) == 9  # works with generators
print("All first() tests passed")
```

### Exercise 2: `is_iterator` — Check If Something Is an Iterator

```python
def is_iterator(obj: object) -> bool:
    """Return True if obj is an iterator (not just iterable).
    
    The canonical test: an iterator IS its own iterator.
    iter(iterator) returns the exact same object.
    
    An iterable that isn't an iterator returns a NEW iterator
    from iter(), so iter(obj) is NOT obj.
    
    IMPORTANT: this test doesn't consume any items!
    
    >>> is_iterator(iter([]))
    True
    >>> is_iterator([1, 2])
    False
    >>> is_iterator((x for x in []))
    True
    """
    # iter() would raise TypeError on non-iterables,
    # but we only care about iterables here.
    # The `is` check is the key — identity, not equality.
    return iter(obj) is obj

# Test cases
assert is_iterator(iter([])) is True          # list_iterator → True
assert is_iterator([1, 2]) is False           # list → False
assert is_iterator(iter([1, 2])) is True      # list_iterator → True

# Generators are always iterators
def gen():
    yield 4
assert is_iterator(gen()) is True

# Critically: calling is_iterator doesn't consume the iterator
i = iter([1, 2])
assert is_iterator(i) is True
assert list(i) == [1, 2]  # still fully intact!

# Non-iterator iterables
assert is_iterator(range(5)) is False         # range is reusable
assert is_iterator({1: "a"}.keys()) is False  # dict_keys is reusable
assert is_iterator("hello") is False          # str is reusable

print("All is_iterator() tests passed")
```

### Exercise 3: `all_same` — Check If All Items Are Equal

```python
from typing import Iterable

def all_same(iterable: Iterable) -> bool:
    """Return True if all items in the iterable are equal.
    
    Must work with:
    - Any iterable (including generators — single-pass)
    - Any items that support == (including unhashable ones like lists)
    - Short-circuit: return False as soon as we find a mismatch
    
    Strategy: grab the first item, then compare everything else to it.
    
    >>> all_same(n % 2 for n in [3, 5, 7, 9])
    True
    >>> all_same(n % 2 for n in [3, 5, 7, 8])
    False
    >>> all_same([])
    True
    """
    iterator = iter(iterable)  # works with any iterable

    # Get the first item. If the iterable is empty, all_same is vacuously True.
    sentinel = object()  # unique object that can't equal any real value
    first = next(iterator, sentinel)

    if first is sentinel:
        return True  # empty iterable

    # Compare every subsequent item to the first.
    # This naturally short-circuits: the moment we see a mismatch, we stop.
    for item in iterator:
        if item != first:
            return False  # mismatch found — stop immediately

    return True  # every item matched

# Test cases
assert all_same(n % 2 for n in [3, 5, 7, 8]) is False
assert all_same(n % 2 for n in [3, 5, 7, 9]) is True
assert all_same([]) is True                          # empty
assert all_same([42]) is True                        # single item
assert all_same([1, 1, 1, 1]) is True
assert all_same([1, 1, 2, 1]) is False
assert all_same([[1, 2], [1, 2], [1, 2]]) is True    # unhashable items
assert all_same([[1, 2], [1, 3]]) is False

# Verify short-circuit: should NOT consume entire generator
def exploding_after_two():
    """Yields 1, 2, then would raise — but we should stop before that."""
    yield 1
    yield 2
    raise RuntimeError("Should not reach here")

assert all_same(exploding_after_two()) is False  # stops at 2 != 1

print("All all_same() tests passed")
```

### Exercise 4: `minmax` — Min and Max in a Single Pass

```python
from typing import Iterable, TypeVar
from typing import SupportsLessThan  # Python 3.12+, or use a Protocol

def minmax(iterable: Iterable) -> tuple:
    """Return (min, max) of an iterable in a single pass.
    
    Why single-pass matters: if the input is a generator, you can only 
    iterate it once. Calling min() then max() would exhaust it on the 
    first call. We must track both in ONE traversal.
    
    This also means O(1) memory — we never store the full sequence.
    
    >>> minmax([9, 5, 2, 8])
    (2, 9)
    >>> minmax(n**2 for n in [9, 5, 2, 8])
    (4, 81)
    """
    iterator = iter(iterable)

    # Initialize min and max with the first element
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("minmax() arg is an empty sequence")

    current_min = first
    current_max = first

    # Single pass through the rest
    for item in iterator:
        if item < current_min:
            current_min = item
        # Note: not elif! An item could be both a new min AND max
        # (only if there's one element, but the logic must be correct)
        if item > current_max:
            current_max = item

    return (current_min, current_max)

# Test cases
assert minmax(n**2 for n in [9, 5, 2, 8]) == (4, 81)
assert minmax([3, 1, 4, 1, 5, 9]) == (1, 9)
assert minmax([42]) == (42, 42)                  # single element
assert minmax(iter([-5, -1, -10])) == (-10, -1)  # negatives

# Verify it works with generators (single-pass requirement)
gen = (x * 10 for x in range(1, 6))
assert minmax(gen) == (10, 50)

print("All minmax() tests passed")
```

### Exercise 5: `csv_columns` — Parse CSV into Column-Based Dict

```python
import csv
from collections import defaultdict
from typing import IO

def csv_columns(file: IO[str]) -> defaultdict[str, list[str]]:
    """Read a CSV file and return a dict mapping headers to column data.
    
    Uses csv.reader to correctly handle commas inside quoted fields.
    Processes rows one at a time — no need to load entire file into memory.
    
    >>> from io import StringIO
    >>> csv_columns(StringIO('h1,h2\\r\\n1,2\\r\\n3,4\\r\\n'))
    defaultdict(<class 'list'>, {'h1': ['1', '3'], 'h2': ['2', '4']})
    """
    reader = csv.reader(file)  # reader is an ITERATOR over rows

    # First row is the header — uses next() to grab just the first row
    # without consuming the rest
    headers: list[str] = next(reader)

    # Build column-based storage
    columns: defaultdict[str, list[str]] = defaultdict(list)

    # Remaining rows are data — the for loop picks up where next() left off
    # because reader is an iterator (single cursor, shared state)
    for row in reader:
        for header, value in zip(headers, row):
            columns[header].append(value)

    return columns

# Test
from io import StringIO

result = csv_columns(StringIO("h1,h2\r\n1,2\r\n3,4\r\n"))
assert dict(result) == {"h1": ["1", "3"], "h2": ["2", "4"]}

# Test with quoted commas (csv module handles this)
result2 = csv_columns(StringIO('name,city\r\n"Doe, Jane",Portland\r\n'))
assert result2["name"] == ["Doe, Jane"]  # comma preserved inside quotes
assert result2["city"] == ["Portland"]

# Test with empty file (just headers)
result3 = csv_columns(StringIO("col_a,col_b\r\n"))
assert dict(result3) == {}  # no data rows, but no error either

print("All csv_columns() tests passed")
```

#### Why `csv_columns` Is an Iterator Lesson

The key insight is **`csv.reader` is an iterator**. When we call `next(reader)` to grab headers, the reader advances. The subsequent `for row in reader` loop picks up from where `next()` left off — it doesn't restart from the beginning. This is the shared-cursor behavior of iterators in action.

```python
from io import StringIO

file = StringIO("header\r\nrow1\r\nrow2\r\n")
reader = csv.reader(file)

# reader is an iterator — verify:
assert iter(reader) is reader  # True

# next() advances it
print(next(reader))  # ['header']

# for-loop continues from where next() stopped
for row in reader:
    print(row)
# ['row1']
# ['row2']
```

---

## Mental Model: The Iteration Decision Tree

When Python encounters `for x in obj`, this is the decision path:

```python
# Python's internal logic (simplified):
#
# 1. Does obj have __iter__?
#    YES → call obj.__iter__() to get an iterator
#    NO  → Does obj have __getitem__?
#           YES → use integer indexing (legacy protocol, 0, 1, 2, ...)
#           NO  → raise TypeError("object is not iterable")
#
# 2. Now we have an iterator. Repeatedly call iterator.__next__():
#    - Got a value? → assign to loop variable, run body
#    - StopIteration raised? → exit loop
#    - Any other exception? → propagate (loop crashes)

# Example of the legacy __getitem__ protocol:
class OldStyle:
    """No __iter__, but for-loop still works via __getitem__."""
    def __getitem__(self, index: int) -> str:
        if index >= 3:
            raise IndexError  # signals end of iteration
        return f"item_{index}"

for item in OldStyle():
    print(item)
# item_0
# item_1
# item_2
```
