#!/usr/bin/env python3
"""Tighten lower-bound pins in pyproject.toml to the latest compatible versions.

For each dependency, queries PyPI (or pypi.nvidia.com) to find the highest
version that satisfies the existing constraints, then rewrites the pin so
the minimum == the latest compatible release.

Example:
    "matplotlib>=3.8"           → "matplotlib>=3.10.1"
    "pandas>=2.0,<2.4.0"       → "pandas>=2.3.2,<2.4.0"
    "scikit-learn>=1.5"         → "scikit-learn>=1.6.1"

ZERO external dependencies — stdlib only (tomllib, urllib, re).

Usage:
    python3 tighten_pins.py              # dry-run (shows what would change)
    python3 tighten_pins.py --write      # rewrite pyproject.toml in place
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


# ── Version parsing (stdlib-only, no `packaging` needed) ──────────────────────

def parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of ints.

    Strips pre-release/dev suffixes for comparison purposes.

    Args:
        v: Version string like "2.3.1" or "26.4.0".

    Returns:
        Tuple of ints, e.g., (2, 3, 1).
    """
    # Strip pre-release markers: "2.4.0dev0" → "2.4.0", "3.0a0" → "3.0"
    clean = re.sub(r"(a|b|rc|dev|post)\d*$", "", v.strip())
    parts: list[int] = []
    for p in clean.split("."):
        digits = re.match(r"(\d+)", p)
        if digits:
            parts.append(int(digits.group(1)))
    return tuple(parts)


def version_satisfies(
    version: str,
    constraints: list[tuple[str, str]],
) -> bool:
    """Check if a version satisfies all constraints.

    Args:
        version: Version string to check.
        constraints: List of (operator, version) pairs,
                     e.g., [(">=", "2.0"), ("<", "2.4.0")].

    Returns:
        True if the version satisfies all constraints.
    """
    v = parse_version(version)
    for op, bound in constraints:
        b = parse_version(bound)
        if op == ">=" and not (v >= b):
            return False
        if op == ">" and not (v > b):
            return False
        if op == "<=" and not (v <= b):
            return False
        if op == "<" and not (v < b):
            return False
        if op == "==" and not (v == b):
            return False
        if op == "!=" and v == b:
            return False
        if op == "~=":
            # ~=X.Y means >=X.Y,<(X+1).0 — compatible release
            if not (v >= b):
                return False
            upper = (b[0] + 1,) if len(b) == 2 else (*b[:-2], b[-2] + 1)
            if not (v < upper):
                return False
    return True


def parse_constraints(spec: str) -> list[tuple[str, str]]:
    """Parse a PEP 440 version specifier into (operator, version) pairs.

    Args:
        spec: Version specifier like ">=2.0,<2.4.0,!=2.1.0".

    Returns:
        List of (operator, version) tuples.
    """
    constraints: list[tuple[str, str]] = []
    for part in spec.split(","):
        part = part.strip()
        m = re.match(r"(~=|==|!=|>=|<=|>|<)(.+)", part)
        if m:
            constraints.append((m.group(1), m.group(2).strip()))
    return constraints


# ── PyPI queries ──────────────────────────────────────────────────────────────

def get_latest_version(
    package: str,
    index: str = "pypi",
) -> str | None:
    """Get the latest non-prerelease version from PyPI JSON API.

    Args:
        package: Package name.
        index: "pypi" for pypi.org, "nvidia" for pypi.nvidia.com.

    Returns:
        Latest version string, or None on failure.
    """
    if index == "pypi":
        url = f"https://pypi.org/pypi/{package}/json"
        try:
            data = json.loads(
                urlopen(Request(url, headers={"User-Agent": "tighten-pins/1.0"}),
                        timeout=15).read()
            )
            return data["info"]["version"]
        except (URLError, KeyError) as e:
            print(f"  WARN: could not fetch {package} from PyPI: {e}")
            return None

    # For NVIDIA index, we don't have a JSON API — use simple index scraping
    return None


def get_all_versions(package: str) -> list[str]:
    """Get all available versions from PyPI, newest first.

    Args:
        package: Package name.

    Returns:
        List of version strings sorted newest first.
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        data = json.loads(
            urlopen(Request(url, headers={"User-Agent": "tighten-pins/1.0"}),
                    timeout=15).read()
        )
        versions = list(data.get("releases", {}).keys())
        # Filter out pre-releases
        stable = [
            v for v in versions
            if not re.search(r"(a|b|rc|dev|post)\d*$", v)
            and not re.search(r"\.(dev|pre|alpha|beta)", v)
        ]
        stable.sort(key=parse_version, reverse=True)
        return stable
    except (URLError, KeyError) as e:
        print(f"  WARN: could not fetch versions for {package}: {e}")
        return []


def find_highest_compatible(
    package: str,
    constraints: list[tuple[str, str]],
) -> str | None:
    """Find the highest version on PyPI that satisfies all constraints.

    Args:
        package: Package name.
        constraints: List of (operator, version) pairs.

    Returns:
        Highest compatible version string, or None if none found.
    """
    versions = get_all_versions(package)
    for v in versions:
        if version_satisfies(v, constraints):
            return v
    return None


# ── pyproject.toml parsing ────────────────────────────────────────────────────

# Regex to parse a dependency line like:
#   "pandas>=2.0,<2.4.0"
#   "cupy-cuda12x>=13.6.0,!=14.0.0"
#   "scikit-learn>=1.5"
#   "numba-cuda[cu12]>=0.22.2,<0.29.0"
DEP_PATTERN = re.compile(
    r'^"'                          # opening quote
    r'(?P<name>[a-zA-Z0-9_.-]+)'   # package name
    r'(?:\[[^\]]+\])?'             # optional extras like [cu12]
    r'(?P<verspec>[><=!~][^"]*?)'  # version specifiers
    r'"'                           # closing quote
)

# RAPIDS packages — skip these (pinned by CalVer, not by latest)
RAPIDS_PACKAGES = {
    "cudf-cu12", "cuml-cu12", "cugraph-cu12",
    "dask-cudf-cu12", "raft-dask-cu12", "pylibraft-cu12",
}

# Packages on NVIDIA index (not on PyPI)
NVIDIA_PACKAGES = RAPIDS_PACKAGES | {
    "rmm-cu12", "libcudf-cu12", "libcuml-cu12",
    "pylibcudf-cu12", "cuda-python", "cuda-toolkit",
}


def process_pyproject(
    path: Path,
    write: bool = False,
) -> None:
    """Read pyproject.toml, tighten pins, optionally write back.

    Args:
        path: Path to pyproject.toml.
        write: If True, rewrite the file in place.
    """
    content = path.read_text()
    lines = content.splitlines(keepends=True)

    changes: list[tuple[str, str, str, str, str]] = []  # name, old_min, new_min, upper, line

    new_lines: list[str] = []
    for line in lines:
        m = DEP_PATTERN.search(line.strip())
        if not m:
            new_lines.append(line)
            continue

        name = m.group("name")
        verspec = m.group("verspec")

        # Skip RAPIDS CalVer packages
        if name.lower() in {p.lower() for p in RAPIDS_PACKAGES}:
            new_lines.append(line)
            continue

        # Skip packages not on regular PyPI
        if name.lower() in {p.lower() for p in NVIDIA_PACKAGES}:
            new_lines.append(line)
            continue

        # Skip ~= pins (already tight)
        if verspec.startswith("~="):
            new_lines.append(line)
            continue

        constraints = parse_constraints(verspec)
        if not constraints:
            new_lines.append(line)
            continue

        # Find current lower bound
        lower_bounds = [(op, v) for op, v in constraints if op in (">=", ">")]
        upper_bounds = [(op, v) for op, v in constraints if op in ("<", "<=")]
        exclusions = [(op, v) for op, v in constraints if op == "!="]

        if not lower_bounds:
            new_lines.append(line)
            continue

        current_min_op, current_min = lower_bounds[0]

        # Query PyPI for the highest compatible version
        best = find_highest_compatible(name, constraints)

        if best is None:
            print(f"  {name:25s}  SKIP (no compatible version found on PyPI)")
            new_lines.append(line)
            continue

        if parse_version(best) <= parse_version(current_min):
            print(f"  {name:25s}  OK   (already at latest: {current_min})")
            new_lines.append(line)
            continue

        # Build new version specifier
        new_parts = [f">={best}"]
        for op, v in upper_bounds:
            new_parts.append(f"{op}{v}")
        for op, v in exclusions:
            new_parts.append(f"{op}{v}")
        new_verspec = ",".join(new_parts)

        # Reconstruct the dependency string
        # Preserve extras if present
        extras_match = re.search(r'(\[[^\]]+\])', line)
        extras = extras_match.group(1) if extras_match else ""

        old_dep_str = m.group(0)  # full match including quotes
        new_dep_str = f'"{name}{extras}{new_verspec}"'

        new_line = line.replace(old_dep_str, new_dep_str)
        new_lines.append(new_line)

        changes.append((name, current_min, best, verspec, new_verspec))
        print(f"  {name:25s}  {verspec:30s} → {new_verspec}")

    # Summary
    print(f"\n{'=' * 70}")
    if not changes:
        print("  No changes needed — all pins are already at latest compatible.")
    else:
        print(f"  {len(changes)} pins to tighten:")
        for name, old_min, new_min, old_spec, new_spec in changes:
            print(f"    {name:25s}  {old_min:>10s} → {new_min}")

        if write:
            path.write_text("".join(new_lines))
            print(f"\n  ✓ Wrote updated {path}")
            print("  Run `uv lock --dry-run` to verify resolution.")
        else:
            print(f"\n  Dry run — no files changed.")
            print(f"  Re-run with --write to apply:\n")
            print(f"    python3 {sys.argv[0]} --write")


def main() -> None:
    """Entry point."""
    write = "--write" in sys.argv

    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print(f"ERROR: {pyproject} not found in current directory.")
        sys.exit(1)

    print("=" * 70)
    print("  TIGHTEN DEPENDENCY PINS")
    print("=" * 70)
    print(f"  Mode: {'WRITE' if write else 'DRY RUN'}")
    print(f"  File: {pyproject.resolve()}\n")

    process_pyproject(pyproject, write=write)


if __name__ == "__main__":
    main()
