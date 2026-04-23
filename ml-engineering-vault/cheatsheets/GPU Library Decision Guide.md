---
type: cheatsheet
title: "GPU Library Decision Guide"
libs:
  - "cupy"
  - "cudf"
  - "numpy"
  - "pandas"
tags:
  - "task/deployment"
related:
  - "[[Accelerated Python - GPU Acceleration]]"
  - "[[Accelerated Python - CPU Vectorization]]"
  - "[[NumPy Pandas Vectorization]]"
source: "Summary from accelerated_python_part2_gpu.ipynb"
created: 2026-04-22
updated: 2026-04-22
---

# GPU Library Decision Guide

---

## Which Library for Which Job?

| Task | CuPy | cuDF | cudf.pandas |
|------|:----:|:----:|:-----------:|
| Numerical array ops (clip, norm, sort) | **best** | — | via cuDF |
| Custom CUDA kernels | **only** | — | — |
| String operations (regex, extract, replace) | none | **best** | yes |
| groupby / join / rolling | — | **best** | yes |
| Drop-in for existing pandas code | — | — | **best** |
| Read CSV/Parquet directly to GPU | — | yes | yes |
| Interop with PyTorch/JAX tensors | yes | partial | — |

---

## Realistic Speedup Expectations

| Operation | Typical GPU speedup |
|-----------|:-------------------:|
| Elementwise float arithmetic | 50–200x |
| Custom fused kernel (`ElementwiseKernel`) | 1.5–3x vs unfused CuPy |
| String regex (cuDF) | 15–25x vs pandas |
| groupby aggregation | 20–40x |
| Hash join (merge) | 25–50x |
| End-to-end pipeline (string+join+rolling) | 20–40x |
| Round-trip (including PCIe transfer) | 1–5x (transfers dominate for small N) |

---

## Install Verification

```bash
nvidia-smi                                        # driver alive?
nvcc --version                                    # CUDA toolkit matches?
python -c "import cupy; cupy.show_config()"       # CuPy finds CUDA?
python -c "import cudf; print(cudf.__version__)"  # cuDF loads?
python -c "import cudf.pandas; cudf.pandas.install(); import pandas; print(type(pandas.DataFrame()))"
# should print: <class 'cudf.core.dataframe.DataFrame'>
```

---

For worked examples, see [[Accelerated Python - GPU Acceleration]].
For CPU-only vectorization patterns, see [[Accelerated Python - CPU Vectorization]].
