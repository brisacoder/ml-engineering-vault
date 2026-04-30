---
type: recipe
title: "RAPIDS Environment Setup"
libs:
  - cupy
  - cudf
  - scikit-learn
tags:
  - task/deployment
  - task/environment
related:
  - "[[Accelerated Python - GPU Acceleration]]"
  - "[[GPU Library Decision Guide]]"
created: 2026-04-23
updated: 2026-04-23
---

# RAPIDS Environment Setup

> Setting up RAPIDS (cuDF, cuML, CuPy) via `uv` is notoriously fragile because of narrow dependency pins — especially pandas. This recipe provides a verified `pyproject.toml`, a discovery script to find the latest compatible versions, and a verification script to confirm everything works.

---

## Why This Is Hard

RAPIDS has **three constraints** that make naive `uv add cudf-cu12` fail:

1. **All RAPIDS packages must share the same CalVer** — cudf 25.02 + cuml 24.12 = broken. They share C++ libraries (libcudf, libcuml) that are ABI-coupled.

2. **cudf pins pandas to a narrow range** — as of 25.02, `pandas>=2.0,<2.2.4dev0`. Install pandas 2.2.4+ and cudf refuses to resolve.

3. **RAPIDS wheels live on `pypi.nvidia.com`**, not pypi.org — `uv` needs the extra index configured explicitly.

---

## Files

The environment setup lives in `recipes/rapids-gpu-env/`:

```
rapids-gpu-env/
├── pyproject.toml                  # uv project with all pins
├── discover_rapids_versions.py     # finds latest RAPIDS CalVer + exact pins
├── tighten_pins.py                 # bumps lower bounds to latest compatible
└── verify_rapids_env.py            # post-install smoke test
```

---

## Step-by-Step Setup

### 1. Discover the Latest Versions

```bash
cd recipes/rapids-gpu-env
python discover_rapids_versions.py
```

This downloads the actual wheel METADATA for `cudf-cu12` and `cuml-cu12` from `pypi.nvidia.com` and prints their **exact** `Requires-Dist` pins. Update `pyproject.toml` accordingly.

### 2. Update the CalVer in pyproject.toml

Update the `~=` pins to match the CalVer from step 1:

```toml
"cudf-cu12~=26.4",
"cuml-cu12~=26.4",
"cugraph-cu12~=26.4",
```

### 3. Tighten Lower Bounds

```bash
python3 tighten_pins.py          # dry run — shows what would change
python3 tighten_pins.py --write  # applies changes
```

This bumps every `>=X.Y` floor to the latest version that still satisfies any upper bounds. Prevents pulling in ancient versions that happen to match.

### 4. Resolve and Install

```bash
uv lock --dry-run --python 3.12    # check for conflicts BEFORE installing
uv sync --python 3.12              # install everything
```

### 5. Verify

```bash
uv run verify_rapids_env.py
```

This checks: CUDA driver, CuPy compute, cuDF groupby, cudf.pandas availability, cuML logistic regression, and CalVer alignment across all RAPIDS packages.

---

## The Dependency Pins (RAPIDS 26.4 — April 2026)

Extracted from the actual `cudf-cu12` and `cuml-cu12` wheel METADATA via `discover_rapids_versions.py`, then tightened to latest compatible with `tighten_pins.py`:

| Package | RAPIDS allows | Tightened to | Notes |
|---|---|---|---|
| Python | `>=3.11` (cp311-abi3) | `>=3.12,<3.14` | 3.10 dropped. Stable ABI means 3.13 works. |
| pandas | `>=2.0,<2.4.0` | `>=2.3.3,<2.4.0` | **#1 conflict source.** cuDF patches pandas internals. |
| numpy | `>=1.26,<3.0` | `>=2.4.4,<3` | cupy 14.x narrows to `>=2.0,<2.6`. |
| pyarrow | `>=19.0.0` | `>=24.0.0` | No upper bound. Was `<18` in 25.02. |
| cupy-cuda12x | `>=13.6.0,!=14.0.0` | `>=14.0.1,!=14.0.0` | 14.0.0 excluded (bug). |
| numba | `>=0.60.0,<0.65.0` | `>=0.64.0,<0.65.0` | cuDF's internal JIT compilation. |
| numba-cuda | `>=0.22.2,<0.29.0` | `>=0.28.2,<0.29.0` | Separate package now (was bundled). |
| scikit-learn | `>=1.5` | `>=1.8.0` | No upper bound. cuML wraps sklearn API. |
| scipy | `>=1.14.0` | `>=1.17.1` | cuML dependency. |

---

## Common Failures

### "No matching distribution found for cudf-cu12"

`uv` can't reach `pypi.nvidia.com`. Check that the `[[tool.uv.index]]` section is present and your network allows HTTPS to `pypi.nvidia.com`.

### Resolution conflict on pandas

cudf pins `pandas<2.2.4` but another package requires `pandas>=2.2.4`. Fix: pin pandas explicitly in your `pyproject.toml` to cudf's range.

### "CUDA driver version is insufficient"

The RAPIDS `-cu12` wheels bundle the CUDA runtime, but you still need a **compatible NVIDIA driver** (>= 525.60.13 for CUDA 12.x). Check with `nvidia-smi`.

### cuml import fails with sklearn error

cuml wraps sklearn and breaks on major sklearn releases. Pin `scikit-learn` to whatever cuml's wheel METADATA declares.

---

## PyTorch + RAPIDS in the Same Environment

This is possible but conflict-prone. PyTorch bundles its own CUDA runtime and cuDNN. If you need both:

1. Use PyTorch's CUDA 12.4 wheels: `--extra-index-url https://download.pytorch.org/whl/cu124`
2. Test resolution with `uv lock --dry-run` before committing
3. Consider separate environments if conflicts persist

See [[Accelerated Python - GPU Acceleration]] for the worked examples and [[GPU Library Decision Guide]] for choosing between CuPy, cuDF, and cudf.pandas.
