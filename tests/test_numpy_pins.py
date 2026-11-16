# SPDX-License-Identifier: Apache-2.0
"""Numpy stays forked by Python version.

Main's lock resolves numpy 2.4.6 on Python < 3.12 and 2.5.4 on Python >= 3.12.
A single exact pin collapses that fork and downgrades the 3.12 path CI runs.
"""

import re
import sys
import tomllib

import numpy

from lpl_t1 import ROOT

# Versions main's uv.lock resolves. The Python >= 3.12 path must stay on 2.5.4.
BELOW_3_12 = "2.4.6"
FROM_3_12 = "2.5.4"

_NUMPY = re.compile(
    r"^numpy==(?P<version>\d+\.\d+\.\d+)"
    r"(?:\s*;\s*python_version\s*(?P<op><|>=)\s*['\"]3\.12['\"])?\s*$"
)


def _pyproject_numpy() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    found: dict[str, str] = {}
    for dep in project["project"]["dependencies"]:
        if not dep.startswith("numpy"):
            continue
        match = _NUMPY.fullmatch(dep)
        assert match, dep
        op = match.group("op")
        assert op, f"numpy requirement has no Python marker: {dep}"
        assert op not in found, dep
        found[op] = match.group("version")
    return found


def _lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text())


def test_pyproject_pins_numpy_per_python_without_downgrading_3_12():
    assert _pyproject_numpy() == {"<": BELOW_3_12, ">=": FROM_3_12}


def test_lock_numpy_fork_agrees_with_pyproject_and_main():
    pins = _pyproject_numpy()
    packages = [package for package in _lock()["package"] if package["name"] == "numpy"]
    by_version = {package["version"]: package for package in packages}
    assert set(by_version) == {BELOW_3_12, FROM_3_12}

    def covers(version: str, relation: str) -> None:
        markers = by_version[version].get("resolution-markers") or []
        needle = f"python_full_version {relation} '3.12'"
        assert markers, f"numpy {version} has no resolution markers"
        assert all(needle in marker for marker in markers), markers
        other = ">=" if relation == "<" else "<"
        assert all(f"python_full_version {other} '3.12'" not in marker for marker in markers)

    covers(pins["<"], "<")
    covers(pins[">="], ">=")

    root = next(
        package for package in _lock()["package"] if package["name"] == "livingproof-booster-t1"
    )
    locked = {
        req["marker"]: req["specifier"]
        for req in root["metadata"]["requires-dist"]
        if req["name"] == "numpy"
    }
    assert locked == {
        "python_full_version < '3.12'": f"=={pins['<']}",
        "python_full_version >= '3.12'": f"=={pins['>=']}",
    }
    assert pins[">="] == FROM_3_12

    for package in _lock()["package"]:
        for dep in package.get("dependencies") or []:
            if dep.get("name") != "numpy" or "version" not in dep:
                continue
            marker = dep.get("marker", "")
            if "python_full_version >= '3.12'" in marker:
                assert dep["version"] == FROM_3_12, dep
            elif "python_full_version < '3.12'" in marker:
                assert dep["version"] == BELOW_3_12, dep
            else:
                raise AssertionError(dep)


def test_installed_numpy_is_the_lock_version_for_this_python():
    expected = FROM_3_12 if sys.version_info >= (3, 12) else BELOW_3_12
    assert numpy.__version__ == expected
