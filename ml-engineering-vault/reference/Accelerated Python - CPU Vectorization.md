---
type: reference
title: "Accelerated Python - CPU Vectorization"
libs:
  - "numpy"
  - "pandas"
tags:
  - "task/data-wrangling"
  - "task/visualization"
  - "task/feature-engineering"
related:
  - "[[Accelerated Python - GPU Acceleration]]"
  - "[[Pandas Plotting]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[GPU Library Decision Guide]]"
  - "[[NumPy Multi-Pass Bottleneck]]"
  - "[[Beyond NumPy - NumExpr Numba JAX]]"
  - "[[Vectorization Expressiveness Limits]]"
  - "[[Array-Oriented Programming]]"
  - "[[NumPy Array Thinking Patterns]]"
source: "Notebook — accelerated_python_part1_cpu.ipynb"
created: 2026-04-22
updated: 2026-04-22
---

# Accelerated Python — CPU Vectorization

> **Notebook:** [[accelerated_python_part1_cpu.ipynb]]
> **Source:** 18 copy-pasteable examples showing where NumPy/Pandas vectorised operations replace Python loops.
> **Mental model:** A NumPy `ndarray` or Pandas `Series` is like a SQL column — operations apply to every element simultaneously, with no Python loop overhead.

---

## Part 1 — String Operations

NumPy and Pandas have a full vectorised string API (`np.char.*` and `Series.str.*`).

| # | Example | Key API |
|---|---------|---------|
| 1 | Batch string lengths | `np.char.str_len`, `np.vectorize(len)` |
| 2 | Normalise/sanitise identifiers | `Series.str.strip().str.lower().str.replace()` |
| 3 | Batch regex extraction → DataFrame | `Series.str.extract(r"(?P<name>...)")` |
| 4 | Batch string templating | `Series` string concatenation with `+`, `DataFrame.apply` |
| 5 | Multi-pattern matching | `Series.str.contains("a|b|c", regex=True)` |

---

## Part 2 — Log Processing

Treat 1M log lines as a single data structure instead of parsing line-by-line.

| # | Example | Key API |
|---|---------|---------|
| 6 | Parse log → structured DataFrame | `Series.str.extract(LOG_RE)` + `pd.to_datetime` + `groupby` |
| 7 | Bulk timestamp reformatting | `pd.to_datetime` + `.dt.strftime` |
| 8 | Pivot log levels into frequency table | `groupby().size().unstack()` |
| 9 | Rolling SLA: p95 over sliding window | `rolling("5min").quantile(0.95)` |

---

## Part 3 — Data Transformation, Lookups & Conditional Logic

ETL steps, config hydration, and "enrich this record" operations.

| # | Example | Key API |
|---|---------|---------|
| 10 | Bulk value mapping (status codes → descriptions) | `Series.map(dict)` |
| 11 | Vectorised if-elif-else | `np.where`, `np.select` |
| 12 | Deduplication and set operations | `np.unique`, `np.setdiff1d`, `np.intersect1d`, `np.union1d` |
| 13 | Batch JSON field extraction | `pd.json_normalize` |
| 14 | Batch file path manipulation | `Series.str.extract`, `Series.str.contains`, `Series.str.replace` |

---

## Part 4 — Performance, Sorting & Numerical Utilities

CI scripts, metrics pipelines, config validation, release tooling.

| # | Example | Key API |
|---|---------|---------|
| 15 | Top-N (O(n) partial sort) | `np.argpartition` |
| 16 | Clip, min-max scale, percentile rank | `np.clip`, `Series.rank(pct=True)` |
| 17 | Config diff between two snapshots | `DataFrame.compare` |
| 18 | IP address validation & CIDR range check | `np.vectorize(ip_to_int)` + integer mask arithmetic |

---

## Key Takeaway

> If you're writing a `for` loop over a list and the loop body is a pure transformation (no I/O, no external state), NumPy/Pandas will be faster **and** shorter.

See [[NumPy Pandas Vectorization]] for the full cheat sheet table, [[Accelerated Python - GPU Acceleration]] for the GPU counterpart using CuPy and cuDF, and [[NumPy Multi-Pass Bottleneck]] for why NumPy still leaves performance on the table (and how [[Beyond NumPy - NumExpr Numba JAX|NumExpr, Numba, and JAX]] fix it). For algorithms that resist vectorization entirely, see [[Vectorization Expressiveness Limits]].
