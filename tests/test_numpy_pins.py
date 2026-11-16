# SPDX-License-Identifier: Apache-2.0
"""Numpy stays forked by Python version.

Python below 3.12 keeps a numpy pin below 2.5. Python 3.12 and newer keep a
pin of at least 2.5.4. pyproject, the lock, and the installed package name the
same version on each side of that fork. A single unmarked pin of the older
lock version would move the 3.12 path down.
"""

import json
import re
import sys
import tomllib

import numpy
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version

from lpl_t1 import ROOT

# Reviewed bounds. Exact current releases stay in the lock, not in this test.
_BELOW_3_12 = Version("2.5")
_FROM_3_12 = Version("2.5.4")

_FORK = re.compile(r"""python_full_version\s*(<|>=)\s*['"]3\.12['"]""")
_WORKFLOW_PYTHON = re.compile(r"""(?m)^[ \t]*python-version:\s*['"]?([^'"#\s]+)""")

_NUMPY_2_4_RULE = {
    "description": (
        "numpy 2.5+ requires Python >=3.12; keep the python_version < '3.12' line on 2.4.x."
    ),
    "matchManagers": ["pep621"],
    "matchPackageNames": ["numpy"],
    "matchCurrentValue": "/^==2\\.4\\./",
    "allowedVersions": "<2.5",
}


def _lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text())


def _project_dependencies() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return list(project["project"]["dependencies"])


def _numpy_dependency_strings() -> list[str]:
    return [dep for dep in _project_dependencies() if Requirement(dep).name == "numpy"]


def _exact_version(req: Requirement) -> str:
    specs = list(req.specifier)
    assert len(specs) == 1 and specs[0].operator == "==", req
    return specs[0].version


def _matches(req: Requirement, python_version: str) -> bool:
    if req.marker is None:
        return True
    env = default_environment()
    env["python_version"] = python_version
    env["python_full_version"] = f"{python_version}.0"
    return req.marker.evaluate(env)


def _pyproject_forks() -> dict[str, str]:
    reqs = [Requirement(dep) for dep in _numpy_dependency_strings()]
    assert len(reqs) == 2
    found: dict[str, str] = {}
    chosen: dict[str, Requirement] = {}
    for op, python in (("<", "3.11"), (">=", "3.12")):
        matched = [req for req in reqs if _matches(req, python)]
        assert len(matched) == 1, (python, [str(req) for req in matched])
        chosen[op] = matched[0]
        found[op] = _exact_version(matched[0])
    assert chosen["<"] is not chosen[">="]
    assert [req for req in reqs if _matches(req, "3.13")] == [chosen[">="]]
    return found


def _fork_op(marker: str) -> str | None:
    ops = set(_FORK.findall(marker))
    if ops == {"<"}:
        return "<"
    if ops == {">="}:
        return ">="
    return None


def _versions_by_fork(entries: list[tuple[str, str]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for marker, version in entries:
        op = _fork_op(marker)
        assert op is not None, marker
        assert op not in found, marker
        found[op] = version
    assert set(found) == {"<", ">="}, found
    return found


def _requires_dist_forks() -> dict[str, str]:
    root = next(pkg for pkg in _lock()["package"] if pkg["name"] == "livingproof-booster-t1")
    entries = []
    for req in root["metadata"]["requires-dist"]:
        if req["name"] != "numpy":
            continue
        specifier = req["specifier"]
        assert specifier.startswith("=="), req
        entries.append((req["marker"], specifier.removeprefix("==")))
    return _versions_by_fork(entries)


def _locked_package_forks() -> dict[str, str]:
    entries = []
    for package in _lock()["package"]:
        if package["name"] != "numpy":
            continue
        markers = package.get("resolution-markers") or []
        assert markers, package["version"]
        ops = {_fork_op(marker) for marker in markers}
        assert ops == {"<"} or ops == {">="}, (package["version"], markers)
        entries.append((markers[0], package["version"]))
    return _versions_by_fork(entries)


def _dependency_forks_agree(pins: dict[str, str]) -> None:
    for package in _lock()["package"]:
        for dep in package.get("dependencies") or []:
            if dep.get("name") != "numpy" or "version" not in dep:
                continue
            op = _fork_op(dep.get("marker", ""))
            assert op is not None, dep
            assert dep["version"] == pins[op], dep


def _is_unmarked_single_pin(requirements: list[str], version: str) -> bool:
    if len(requirements) != 1:
        return False
    req = Requirement(requirements[0])
    return req.marker is None and _exact_version(req) == version


def test_each_python_version_has_one_numpy_pin():
    forks = _pyproject_forks()
    assert set(forks) == {"<", ">="}


def test_below_3_12_numpy_is_under_2_5_and_3_12_meets_the_floor():
    forks = _pyproject_forks()
    assert Version(forks["<"]) < _BELOW_3_12
    assert Version(forks[">="]) >= _FROM_3_12


def test_pyproject_lock_and_installed_numpy_agree():
    forks = _pyproject_forks()
    assert _requires_dist_forks() == forks
    assert _locked_package_forks() == forks
    _dependency_forks_agree(forks)
    expected = forks[">="] if sys.version_info >= (3, 12) else forks["<"]
    assert numpy.__version__ == expected


def test_renovate_keeps_the_numpy_2_4_line_below_2_5():
    config = json.loads((ROOT / "renovate.json").read_text())
    assert _NUMPY_2_4_RULE in (config.get("packageRules") or [])


def test_single_pin_of_the_older_lock_fork_downgrades_python_3_12():
    """One unmarked pin of the older lock version is below the 3.12 lock version."""
    forks = _locked_package_forks()
    older, newer = forks["<"], forks[">="]
    assert Version(older) < Version(newer)
    rejected = [f"numpy=={older}"]
    assert _is_unmarked_single_pin(rejected, older)
    rejected_req = Requirement(rejected[0])
    assert _matches(rejected_req, "3.11") and _matches(rejected_req, "3.12")
    assert not _is_unmarked_single_pin(_numpy_dependency_strings(), older)


def test_workflow_python_versions_agree():
    """Every workflow python-version is the same value."""
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert workflows
    versions: list[str] = []
    for path in workflows:
        code = "\n".join(
            line.split(" #", 1)[0].rstrip()
            for line in path.read_text(encoding="utf-8").splitlines()
        )
        versions.extend(_WORKFLOW_PYTHON.findall(code))
    assert len(set(versions)) <= 1
