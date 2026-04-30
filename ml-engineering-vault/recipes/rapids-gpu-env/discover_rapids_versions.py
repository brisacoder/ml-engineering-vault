#!/usr/bin/env python3
"""Discover the latest compatible RAPIDS versions and their dependency pins.

Run this BEFORE editing pyproject.toml to find the correct version numbers.
Requires network access — run on your local machine, not in a sandbox.

ZERO external dependencies — uses only Python stdlib (urllib, zipfile, etc.).

Usage:
    python3 discover_rapids_versions.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError


# ── Helpers ───────────────────────────────────────────────────────────────────

USER_AGENT = "rapids-discovery/1.0"


def _fetch(url: str, timeout: int = 30) -> bytes:
    """Fetch raw bytes from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Response body as bytes.
    """
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_text(url: str, timeout: int = 30) -> str:
    """Fetch text content from a URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Response body decoded as UTF-8.
    """
    return _fetch(url, timeout).decode("utf-8")


class _SimpleIndexParser(HTMLParser):
    """Parse a PEP 503 simple index page to extract package filenames/links."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []  # (href, text)
        self._current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    self._current_href = value

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self.links.append((self._current_href, data.strip()))
            self._current_href = None


def get_versions_from_simple_index(
    package: str,
    index_url: str = "https://pypi.org/simple",
) -> list[str]:
    """Scrape a PEP 503 simple index page to find available versions.

    Args:
        package: Package name (e.g., "cudf-cu12").
        index_url: Base URL of the simple index.

    Returns:
        List of version strings, newest first (sorted by version tuple).
    """
    url = f"{index_url}/{package}/"
    try:
        html = _fetch_text(url)
    except URLError as e:
        print(f"  ERROR fetching {url}: {e}")
        return []

    parser = _SimpleIndexParser()
    parser.feed(html)

    # Extract versions from wheel/tar filenames
    # Wheel: package_name-VERSION-cp3XX-...whl
    # Sdist: package-name-VERSION.tar.gz
    versions: set[str] = set()
    pkg_normalized = re.sub(r"[-_.]+", "[_-]", package)
    pattern = re.compile(
        rf"^{pkg_normalized}-(\d+\.\d+[\w.]*?)(?:-|\.tar)",
        re.IGNORECASE,
    )
    for href, text in parser.links:
        filename = text or href.rsplit("/", 1)[-1].split("#")[0]
        m = pattern.match(filename)
        if m:
            versions.add(m.group(1))

    def version_key(v: str) -> tuple[int, ...]:
        return tuple(int(x) for x in re.findall(r"\d+", v))

    return sorted(versions, key=version_key, reverse=True)


def download_wheel(
    package: str,
    index_url: str = "https://pypi.org/simple",
    python_version: str = "cp312",
    platform: str = "linux",
) -> bytes | None:
    """Download the first matching wheel for a package.

    Prefers manylinux cp312 wheels. Falls back to any available wheel.

    Args:
        package: Package name.
        index_url: Base URL of the simple index.
        python_version: CPython version tag to prefer (e.g., "cp312").
        platform: Platform substring to match (e.g., "linux").

    Returns:
        Wheel file contents as bytes, or None if not found.
    """
    url = f"{index_url}/{package}/"
    try:
        html = _fetch_text(url)
    except URLError as e:
        print(f"  ERROR fetching {url}: {e}")
        return None

    parser = _SimpleIndexParser()
    parser.feed(html)

    # Filter to .whl files only
    wheel_links: list[tuple[str, str]] = [
        (href, text) for href, text in parser.links
        if text.endswith(".whl") or href.split("#")[0].endswith(".whl")
    ]

    if not wheel_links:
        print(f"  No wheels found for {package}")
        return None

    # Find the latest version first
    versions = get_versions_from_simple_index(package, index_url)
    if not versions:
        return None

    latest = versions[0]

    # Among wheels for the latest version, prefer cp312 + linux
    pkg_normalized = re.sub(r"[-_.]+", "_", package)
    candidates: list[tuple[str, int]] = []
    for href, text in wheel_links:
        filename = text or href.rsplit("/", 1)[-1].split("#")[0]
        if latest.replace(".", "_") in filename.replace(".", "_") or latest in filename:
            score = 0
            if python_version in filename:
                score += 10
            if platform in filename:
                score += 5
            if "manylinux" in filename:
                score += 3
            # Resolve relative URLs
            if href.startswith("http"):
                full_url = href.split("#")[0]
            else:
                full_url = f"{index_url}/{package}/{href}".split("#")[0]
            candidates.append((full_url, score))

    if not candidates:
        # Fallback: try any wheel
        for href, text in wheel_links[-5:]:
            if href.startswith("http"):
                full_url = href.split("#")[0]
            else:
                full_url = f"{index_url}/{package}/{href}".split("#")[0]
            candidates.append((full_url, 0))

    candidates.sort(key=lambda x: x[1], reverse=True)

    for wheel_url, _score in candidates[:3]:
        try:
            print(f"  Downloading: {wheel_url.rsplit('/', 1)[-1][:80]}...")
            return _fetch(wheel_url, timeout=120)
        except URLError as e:
            print(f"  Failed: {e}")
            continue

    return None


def extract_metadata(wheel_bytes: bytes) -> dict[str, str | list[str]]:
    """Extract METADATA from a wheel file (which is a zip archive).

    Args:
        wheel_bytes: Raw bytes of the .whl file.

    Returns:
        Dict with 'version', 'requires_python', and 'requires_dist' keys.
    """
    metadata: dict[str, str | list[str]] = {"requires_dist": []}

    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/METADATA") or name.endswith(".dist-info/METADATA"):
                content = zf.read(name).decode("utf-8")
                for line in content.splitlines():
                    if line.startswith("Version:"):
                        metadata["version"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Requires-Python:"):
                        metadata["requires_python"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Requires-Dist:"):
                        dep = line.split(":", 1)[1].strip()
                        assert isinstance(metadata["requires_dist"], list)
                        metadata["requires_dist"].append(dep)
                break

    return metadata


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the discovery process and print a compatibility report."""
    nvidia_index = "https://pypi.nvidia.com"  # no /simple — NVIDIA uses flat layout
    pypi_index = "https://pypi.org/simple"

    print("=" * 70)
    print("  RAPIDS VERSION DISCOVERY")
    print("=" * 70)
    print(f"  Python: {sys.version.split()[0]}")

    # ── 1. Find latest cudf-cu12 versions ─────────────────────────────
    print("\n[1/5] Checking available cudf-cu12 versions...")
    cudf_versions = get_versions_from_simple_index("cudf-cu12", nvidia_index)
    if cudf_versions:
        print(f"  Latest:  {cudf_versions[0]}")
        print(f"  Recent:  {', '.join(cudf_versions[:8])}")
    else:
        print("  Could not fetch versions from pypi.nvidia.com.")
        print("  Check network access to https://pypi.nvidia.com")
        return

    # ── 2. Extract cudf-cu12 wheel METADATA ───────────────────────────
    print(f"\n[2/5] Inspecting cudf-cu12 wheel metadata...")
    cudf_wheel = download_wheel("cudf-cu12", nvidia_index)
    if cudf_wheel:
        cudf_meta = extract_metadata(cudf_wheel)
        print(f"  Version:         {cudf_meta.get('version', '?')}")
        print(f"  Requires-Python: {cudf_meta.get('requires_python', '?')}")
        print(f"  Dependencies:")
        for dep in cudf_meta.get("requires_dist", []):
            dep_str = str(dep)
            is_extra = "extra ==" in dep_str
            marker = ""
            if not is_extra and any(
                k in dep_str.lower()
                for k in ["pandas", "numpy", "pyarrow", "cupy", "numba"]
            ):
                marker = "  ◄── CRITICAL"
            # Skip test/optional extras to reduce noise
            if is_extra:
                continue
            print(f"    {dep_str}{marker}")
    else:
        print("  Could not download cudf-cu12 wheel.")
        cudf_meta = {}

    # ── 3. Extract cuml-cu12 wheel METADATA ───────────────────────────
    print(f"\n[3/5] Inspecting cuml-cu12 wheel metadata...")
    cuml_wheel = download_wheel("cuml-cu12", nvidia_index)
    if cuml_wheel:
        cuml_meta = extract_metadata(cuml_wheel)
        print(f"  Version:         {cuml_meta.get('version', '?')}")
        print(f"  Requires-Python: {cuml_meta.get('requires_python', '?')}")
        print(f"  Dependencies:")
        for dep in cuml_meta.get("requires_dist", []):
            dep_str = str(dep)
            is_extra = "extra ==" in dep_str
            marker = ""
            if not is_extra and any(
                k in dep_str.lower()
                for k in ["scikit", "numpy", "cupy", "pandas", "numba", "scipy"]
            ):
                marker = "  ◄── CRITICAL"
            if is_extra:
                continue
            print(f"    {dep_str}{marker}")
    else:
        print("  Could not download cuml-cu12 wheel.")
        cuml_meta = {}

    # ── 4. Extract cugraph-cu12 wheel METADATA ────────────────────────
    print(f"\n[4/5] Inspecting cugraph-cu12 wheel metadata...")
    cugraph_wheel = download_wheel("cugraph-cu12", nvidia_index)
    if cugraph_wheel:
        cugraph_meta = extract_metadata(cugraph_wheel)
        print(f"  Version:         {cugraph_meta.get('version', '?')}")
        print(f"  Requires-Python: {cugraph_meta.get('requires_python', '?')}")
    else:
        print("  Could not download cugraph-cu12 wheel.")

    # ── 5. Check cupy-cuda12x on regular PyPI ─────────────────────────
    print(f"\n[5/5] Checking cupy-cuda12x on PyPI...")
    try:
        cupy_data = json.loads(_fetch_text(
            "https://pypi.org/pypi/cupy-cuda12x/json"
        ))
        info = cupy_data["info"]
        print(f"  Latest version:  {info['version']}")
        print(f"  Requires-Python: {info.get('requires_python', '?')}")
        deps = info.get("requires_dist") or []
        for dep in deps:
            if "numpy" in dep.lower():
                print(f"  numpy pin:       {dep}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUGGESTED pyproject.toml UPDATES")
    print("=" * 70)

    cudf_ver = cudf_meta.get("version", "")
    if cudf_ver:
        # Extract CalVer: "25.2.1" → "25.2"
        parts = str(cudf_ver).split(".")
        if len(parts) >= 2:
            calver = f"{parts[0]}.{parts[1]}"
            print(f"\n  RAPIDS CalVer: {calver}")
            print(f'    "cudf-cu12~={calver}",')
            print(f'    "cuml-cu12~={calver}",')
            print(f'    "cugraph-cu12~={calver}",')

    # Extract critical pins from cudf metadata (skip test extras)
    cudf_deps = cudf_meta.get("requires_dist", [])
    assert isinstance(cudf_deps, list)
    print("\n  Dependency pins from cudf-cu12 wheel:")
    for dep in cudf_deps:
        dep_str = str(dep)
        if "extra ==" in dep_str:
            continue
        dep_lower = dep_str.lower()
        if any(k in dep_lower for k in ["pandas", "numpy", "pyarrow", "cupy", "numba"]):
            clean = dep_str.split(";")[0].strip()
            print(f'    "{clean}",')

    cuml_deps = cuml_meta.get("requires_dist", [])
    assert isinstance(cuml_deps, list)
    print("\n  Dependency pins from cuml-cu12 wheel:")
    for dep in cuml_deps:
        dep_str = str(dep)
        if "extra ==" in dep_str:
            continue
        if any(k in dep_str.lower() for k in ["scikit", "sklearn", "scipy", "numpy", "cupy"]):
            clean = dep_str.split(";")[0].strip()
            print(f'    "{clean}",')

    print(f"\n  Python: {cudf_meta.get('requires_python', '?')}")
    print()


if __name__ == "__main__":
    main()
