---
type: cheatsheet
title: "NumPy Pandas Vectorization"
libs:
  - "numpy"
  - "pandas"
tags:
  - "task/data-wrangling"
related:
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[Accelerated Python - GPU Acceleration]]"
  - "[[GPU Library Decision Guide]]"
  - "[[Pandas Plotting]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[Array-Oriented Programming]]"
  - "[[NumPy Array Thinking Patterns]]"
source: "Summary table from accelerated_python_part1_cpu.ipynb"
created: 2026-04-22
updated: 2026-04-22
---

# NumPy & Pandas Vectorization Cheat Sheet

> **Rule of thumb:** if the loop body is a pure transformation (no I/O, no external state), there's a vectorised replacement.

---

## String Operations

| Task | NumPy | Pandas |
|------|-------|--------|
| Length of N strings | `np.char.str_len(arr)` | `s.str.len()` |
| Normalise / strip / lower | `np.char.lower(arr)` | `s.str.strip().str.lower()` |
| Regex extract to columns | — | `s.str.extract(r"(?P<a>...)")` |
| Multi-pattern flag | `np.char.find` | `s.str.contains("a\|b\|c")` |
| Batch template render | `np.vectorize(fn)` | `df.apply(render, axis=1)` |

---

## Log & Time Series Processing

| Task | NumPy | Pandas |
|------|-------|--------|
| Parse structured log | — | `s.str.extract(LOG_RE)` |
| Reformat timestamps | — | `pd.to_datetime` + `.dt.strftime` |
| Pivot count table | — | `groupby().size().unstack()` |
| Rolling window stat | — | `rolling(w).quantile(0.95)` |

---

## Data Transformation & Lookups

| Task | NumPy | Pandas |
|------|-------|--------|
| Dictionary lookup | — | `s.map(d)` |
| if-elif-else on array | `np.select(conds, vals)` | `pd.cut` / `np.select` |
| Deduplication | `np.unique(arr)` | `s.drop_duplicates()` |
| Set diff / intersect | `np.setdiff1d` / `np.intersect1d` | `pd.Index.difference` |
| Flatten JSON → DataFrame | — | `pd.json_normalize(dicts)` |
| Path / extension ops | — | `s.str.extract(r"(\.\w+)$")` |

---

## Performance & Numerical

| Task | NumPy | Pandas |
|------|-------|--------|
| Top-N (O(n)) | `np.argpartition(arr, -N)[-N:]` | `s.nlargest(N)` |
| Clip + normalise | `np.clip(arr, lo, hi)` | `s.clip(lo, hi)` |
| Config diff | — | `df.compare(df2)` |
| IP range classification | `np.vectorize(ip_to_int)` + masks | — |

---

For worked examples of every row, see [[Accelerated Python - CPU Vectorization]].
For the GPU equivalents, see [[Accelerated Python - GPU Acceleration]].
For why NumPy still has a performance ceiling, see [[NumPy Multi-Pass Bottleneck]].
For algorithms that can't be vectorised at all, see [[Vectorization Expressiveness Limits]].
