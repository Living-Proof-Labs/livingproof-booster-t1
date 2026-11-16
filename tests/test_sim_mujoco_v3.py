# SPDX-License-Identifier: Apache-2.0
"""livingproof-sim-mujoco v3 compatibility that a reviewer would otherwise check by hand.

v3 requires ``--lock`` on ``evidence`` and ``walk stand``, owns the world, and
requires every pin to record ``bytes``. This package does not call those
commands and does not add a world. The published pins already record the
artifact sizes; this file checks those sizes against the files ``fetch`` wrote.
"""

import json

from lpl_t1 import ROOT
from lpl_t1.adapter import ROBOT
from lpl_t1.scene import build
from livingproof_sim_mujoco.pins import local

# Commands v3 will not run without a caller lock. media's --lock is optional.
SIM_COMMANDS = ("livingproof_sim_mujoco.evidence", "livingproof_sim_mujoco.walk")
SCAN_ROOTS = (
    ROOT / "src",
    ROOT / "docs",
    ROOT / ".github" / "workflows",
)
SCAN_FILES = (ROOT / "Taskfile.yml", ROOT / "README.md", ROOT / "pyproject.toml")


def _text_files():
    for path in SCAN_FILES:
        if path.is_file():
            yield path
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() not in {".png", ".jpg", ".whl", ".pyc"}:
                yield path


def _command_missing_lock(line: str) -> bool:
    """True for an evidence or walk-stand command that does not pass ``--lock``.

    Importing helpers from ``livingproof_sim_mujoco.walk`` is not that command.
    """
    stripped = line.strip()
    if stripped.startswith(("from ", "import ")):
        return False
    if not any(command in line for command in SIM_COMMANDS):
        return False
    return "--lock" not in line


def test_evidence_and_stand_calls_pass_lock():
    """A sim-mujoco evidence or stand invocation includes ``--lock`` on that line."""
    assert _command_missing_lock("uv run python -m livingproof_sim_mujoco.evidence adapter cfg out")
    assert _command_missing_lock(
        "uv run python -m livingproof_sim_mujoco.walk stand adapter --configuration stock"
    )
    assert not _command_missing_lock(
        "uv run python -m livingproof_sim_mujoco.walk stand adapter --lock lock.json"
    )
    assert not _command_missing_lock("from livingproof_sim_mujoco.walk import initial_state")
    offenders = []
    for path in _text_files():
        rel = path.relative_to(ROOT).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _command_missing_lock(line):
                offenders.append(f"{rel}:{number}: {line.strip()}")
    assert offenders == []


def test_the_package_does_not_add_the_world():
    """v3 supplies the world. This package compiles the pinned model only."""
    package = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "src" / "lpl_t1").glob("*.py")
    )
    assert "add_world" not in package and "load_world" not in package
    assert not hasattr(ROBOT, "world")
    for name, model in (("stock", "booster_gym"), ("playground", "playground")):
        scene = build(model=model)
        assert scene.world == {} and scene.walls == [], name


def test_published_pins_record_each_artifact_byte_count():
    """Every pin ``bytes`` is the size of the file fetch verified, not a guess."""
    groups = json.loads((ROOT / "pins.json").read_text())["groups"]
    missing = []
    mismatched = []
    for name, group in groups.items():
        for path, entry in group["files"].items():
            label = f"{name}:{path}"
            count = entry.get("bytes")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                missing.append(label)
                continue
            artifact = local(name, path, ROOT)
            size = artifact.stat().st_size
            if size != count:
                mismatched.append(f"{label}: pin {count}, artifact {size}")
    assert missing == []
    assert mismatched == []
