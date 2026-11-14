# SPDX-License-Identifier: Apache-2.0
"""This robot package does not select a world or a scenario.

Names of scenarios and worlds may appear only in legacy evidence readers.
This repository stores no recorded evidence, so that allowlist is empty and
no reader is kept. The search pattern itself lives in this file.
"""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# This file names the strings it forbids. Every other path is scanned.
GUARD = Path(__file__).resolve()
# Readers of evidence already recorded in this repo would be listed here.
LEGACY_EVIDENCE = ()
SKIP_DIRS = {
    ".git",
    ".cache",
    ".venv",
    "artifacts",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".zensical-cache",
    "downloads",
}
# Split so this file does not itself contain the forbidden names.
NAMES = re.compile("hall" + "way" + "|" + "cor" + "ner", re.IGNORECASE)
SELECTION_KEYS = {"world", "scenario"}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def _text_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or _skipped(path) or path.resolve() == GUARD:
            continue
        if path.suffix.lower() in {".stl", ".png", ".jpg", ".pyc", ".whl"}:
            continue
        yield path


def _keys(value, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                found.add(key)
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)


def test_scenario_and_world_names_stay_in_legacy_evidence():
    assert LEGACY_EVIDENCE == ()
    offenders = []
    for path in _text_files():
        rel = _rel(path)
        if rel in LEGACY_EVIDENCE:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if NAMES.search(path.name) or NAMES.search(text):
            offenders.append(rel)
    assert offenders == []


def test_pins_locks_and_runner_do_not_select_a_world_or_scenario():
    pins = json.loads((ROOT / "pins.json").read_text())
    keys = set()
    _keys(pins, keys)
    assert SELECTION_KEYS.isdisjoint(keys)
    assert "worlds" not in pins["groups"]
    for group in pins["groups"].values():
        for path in group.get("files", {}):
            assert not path.startswith("worlds/")
            assert "/scenarios/" not in path
    assert not (ROOT / "verification.lock.json").exists()
    runner = json.loads((ROOT / "runner.json").read_text())
    runner_keys = set()
    _keys(runner, runner_keys)
    assert SELECTION_KEYS.isdisjoint(runner_keys)
    for path in _text_files():
        if "lock" not in path.name or path.suffix != ".json":
            continue
        lock_keys = set()
        _keys(json.loads(path.read_text(encoding="utf-8")), lock_keys)
        assert SELECTION_KEYS.isdisjoint(lock_keys), _rel(path)


def test_workflows_have_no_world_or_scenario_inputs():
    workflow_dir = ROOT / ".github" / "workflows"
    assert sorted(p.name for p in workflow_dir.iterdir()) == ["docs.yml", "release.yml"]
    for path in sorted(workflow_dir.iterdir()):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        keys = set()
        _keys(document, keys)
        assert SELECTION_KEYS.isdisjoint(keys), path.name
        for job in (document.get("jobs") or {}).values():
            assert "with" not in job or SELECTION_KEYS.isdisjoint(job["with"])
            uses = job.get("uses", "")
            assert isinstance(uses, str)
            assert not NAMES.search(uses)


def test_package_does_not_load_a_world_or_wrap_a_scenario():
    package = ROOT / "src" / "lpl_t1"
    names = sorted(p.name for p in package.glob("*.py"))
    assert "scenario.py" not in names
    assert "evidence.py" not in names
    assert "walk.py" not in names
    text = "\n".join((package / name).read_text(encoding="utf-8") for name in names)
    for call in ("load_world", "add_world"):
        assert call not in text
    taskfile = (ROOT / "Taskfile.yml").read_text(encoding="utf-8")
    assert "\n  repin:" not in taskfile
    assert "\n  repin:check:" not in taskfile
    assert not (ROOT / "tools" / "repin.py").exists()
