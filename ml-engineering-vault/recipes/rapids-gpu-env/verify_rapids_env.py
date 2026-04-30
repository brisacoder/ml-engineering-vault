#!/usr/bin/env python3
"""Verify that all RAPIDS components are installed and compatible.

Run after `uv sync` to confirm everything works together.

Usage:
    uv run verify_rapids_env.py
"""
from __future__ import annotations

import sys


def check(label: str, fn: callable) -> bool:
    """Run a check function and print pass/fail.

    Args:
        label: Human-readable name for the check.
        fn: Zero-argument callable that raises on failure.

    Returns:
        True if the check passed, False otherwise.
    """
    try:
        result = fn()
        print(f"  ✓  {label}: {result}")
        return True
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        return False


def main() -> None:
    """Run all compatibility checks."""
    print("=" * 60)
    print("  RAPIDS ENVIRONMENT VERIFICATION")
    print("=" * 60)

    results: list[bool] = []

    # ── Python version ────────────────────────────────────────────
    print(f"\n  Python: {sys.version}")

    # ── Core libraries ────────────────────────────────────────────
    print("\n── Core Libraries ──")
    results.append(check("numpy", lambda: __import__("numpy").__version__))
    results.append(check("pandas", lambda: __import__("pandas").__version__))
    results.append(check("pyarrow", lambda: __import__("pyarrow").__version__))
    results.append(check("numba", lambda: __import__("numba").__version__))
    results.append(check("scikit-learn", lambda: __import__("sklearn").__version__))

    # ── GPU: CUDA ─────────────────────────────────────────────────
    print("\n── GPU / CUDA ──")

    def check_nvidia_smi() -> str:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        return out or "no GPU found"

    results.append(check("nvidia-smi", check_nvidia_smi))

    # ── CuPy ──────────────────────────────────────────────────────
    print("\n── CuPy ──")

    def check_cupy() -> str:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        ver = cp.__version__
        cuda_ver = cp.cuda.runtime.runtimeGetVersion()
        return f"{ver} (CUDA runtime {cuda_ver}, {n} device(s))"

    def check_cupy_compute() -> str:
        import cupy as cp
        a = cp.arange(1_000_000, dtype=cp.float32)
        result = float(cp.sum(a))
        expected = 999999 * 1000000 / 2
        assert abs(result - expected) < 1.0, f"got {result}, expected {expected}"
        return f"sum of 1M floats = {result:.0f} ✓"

    results.append(check("cupy import", check_cupy))
    results.append(check("cupy compute", check_cupy_compute))

    # ── cuDF ──────────────────────────────────────────────────────
    print("\n── cuDF ──")

    def check_cudf() -> str:
        import cudf
        return cudf.__version__

    def check_cudf_compute() -> str:
        import cudf
        import pandas as pd
        pdf = pd.DataFrame({"a": range(100_000), "b": range(100_000)})
        gdf = cudf.from_pandas(pdf)
        result = gdf.groupby("a").sum().reset_index()
        assert len(result) == 100_000
        return f"groupby on 100k rows ✓ (type: {type(gdf).__module__})"

    results.append(check("cudf import", check_cudf))
    results.append(check("cudf compute", check_cudf_compute))

    # ── cudf.pandas ───────────────────────────────────────────────
    print("\n── cudf.pandas ──")

    def check_cudf_pandas() -> str:
        import cudf.pandas
        return "available (activate with: import cudf.pandas; cudf.pandas.install())"

    results.append(check("cudf.pandas", check_cudf_pandas))

    # ── cuML ──────────────────────────────────────────────────────
    print("\n── cuML ──")

    def check_cuml() -> str:
        import cuml
        return cuml.__version__

    def check_cuml_compute() -> str:
        import cuml
        import numpy as np
        from cuml.linear_model import LogisticRegression as cuLR
        X = np.random.randn(1000, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32)
        model = cuLR()
        model.fit(X, y)
        acc = (model.predict(X) == y).mean()
        return f"LogisticRegression acc={acc:.3f} ✓"

    results.append(check("cuml import", check_cuml))
    results.append(check("cuml compute", check_cuml_compute))

    # ── Version compatibility matrix ──────────────────────────────
    print("\n── Version Matrix ──")
    try:
        import numpy as np
        import pandas as pd
        import cudf
        import cupy as cp

        print(f"  numpy   : {np.__version__}")
        print(f"  pandas  : {pd.__version__}")
        print(f"  cudf    : {cudf.__version__}")
        print(f"  cupy    : {cp.__version__}")

        # Check CalVer alignment
        cudf_calver = ".".join(cudf.__version__.split(".")[:2])
        try:
            import cuml
            cuml_calver = ".".join(cuml.__version__.split(".")[:2])
            if cudf_calver != cuml_calver:
                print(f"\n  ⚠  CalVer MISMATCH: cudf={cudf_calver} cuml={cuml_calver}")
                print("     All RAPIDS packages must share the same CalVer!")
            else:
                print(f"\n  ✓  CalVer aligned: {cudf_calver}")
        except ImportError:
            pass
    except ImportError:
        print("  (some imports failed — see above)")

    # ── Summary ───────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉 Environment is fully operational!")
    else:
        print("  ⚠  Some checks failed — review above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
