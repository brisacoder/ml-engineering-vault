---
type: reference
title: "Accelerated Python - GPU Acceleration"
libs:
  - "numpy"
  - "pandas"
  - "cupy"
  - "cudf"
tags:
  - "task/data-wrangling"
  - "task/deployment"
related:
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[NumPy Pandas Vectorization]]"
  - "[[GPU Library Decision Guide]]"
source: "Notebook — accelerated_python_part2_gpu.ipynb"
created: 2026-04-22
updated: 2026-04-22
---

# Accelerated Python — GPU Acceleration

> **Notebook:** [[accelerated_python_part2_gpu.ipynb]]
> **Prerequisite:** [[Accelerated Python - CPU Vectorization]] (Part 1)
> **Covers:** CuPy (NumPy on GPU), cuDF (Pandas on GPU), cudf.pandas (zero-code-change mode)

---

## Compatibility Matrix

```
CUDA       : 11.8 or 12.0/12.2/12.5
Python     : 3.10, 3.11, 3.12
pandas     : 2.0.x – 2.2.x (cuDF pins to a specific minor)
RAPIDS     : 24.10, 24.12, 25.02
```

**Known landmines:** cuDF pins pandas minor version; `cudf.pandas` must import *before* `import pandas`; ~95% API coverage with silent CPU fallback; RAPIDS conflicts with PyTorch conda envs; Windows is WSL2-only.

---

## Part 1 — CuPy: NumPy on the GPU

`cp.array` is `np.array` that lives in VRAM. Every `np.*` has a `cp.*` equivalent.

| # | Example | Key Concept |
|---|---------|-------------|
| 1 | Three-line GPU migration | `cp.clip`, `cp.asnumpy` — change 3 lines, get 100x+ |
| 2 | Custom CUDA kernels inline | `cp.ElementwiseKernel` — write CUDA C from Python, no `.cu` files |
| 3 | Keep data on-device (pipeline chaining) | Avoid `cp.asnumpy` per step — one transfer in, one out |

**The anti-pattern:** bouncing data CPU↔GPU at every step. Three round-trips on 80 MB ≈ 240 MB of PCIe traffic — can be *slower* than pure NumPy.

---

## Part 2 — cuDF: Pandas on the GPU

cuDF operates on DataFrames with **native GPU string support** — what CuPy cannot do.

| # | Example | Key Concept |
|---|---------|-------------|
| 4 | `cudf.read_csv` — file → GPU, no CPU DataFrame | GPU-accelerated CSV parsing via libcudf |
| 5 | cuDF string operations (the killer feature) | `.str.contains`, `.str.extract`, `.str.replace` — same API, ~20x faster |
| 6 | GPU groupby + aggregation | GPU hash-based groupby, 20-40x vs pandas |
| 7 | GPU merge (hash join) | `df.merge()` on GPU, 25-50x vs pandas |

---

## Part 3 — cudf.pandas: Zero-Code-Change Mode

```python
import cudf.pandas          # must come BEFORE import pandas
cudf.pandas.install()
import pandas as pd         # now GPU-backed
```

| # | Example | Key Concept |
|---|---------|-------------|
| 8 | Existing pandas code runs on GPU unchanged | `cudf.pandas` intercepts all pandas calls |
| 9 | Detecting CPU fallbacks with the profiler | `cudf.pandas.Profiler()` logs which ops fell back |

**Common fallbacks:** `df.apply(fn, axis=1)`, `MultiIndex` ops, `pivot_table`, sparse dtypes — see notebook for fixes.

---

## Part 4 — End-to-End GPU Pipeline

| # | Example | Key Concept |
|---|---------|-------------|
| 10 | Log ingest → parse → PII scrub → join → rolling SLA → alerts | All on GPU, single `to_pandas()` at the end. ~35x vs pandas. |

---

## The One Rule That Matters

> **Transfer once. Process everything on-device. Transfer once back.**

Every extra `cp.asnumpy()` / `df.to_pandas()` in a hot path is a PCIe round-trip that can cost more than the GPU computation itself.

See [[GPU Library Decision Guide]] for the decision matrix and realistic speedup expectations.
