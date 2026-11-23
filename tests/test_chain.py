# SPDX-License-Identifier: Apache-2.0
"""Adapter entry point, configuration profiles, and the released simulator pin.

Integration pins (scenario, world, verifier, reporter) live in
livingproof-verification. This package does not publish them and does not grade.
"""

import importlib
import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

import lpl_t1
from lpl_t1 import ROOT
from lpl_t1.adapter import ROBOT


def test_configurations_match_the_published_profiles():
    robot = json.loads((ROOT / "robot.json").read_text())
    assert robot["id"] == "booster-t1"
    assert [c["name"] for c in robot["configurations"]] == ["stock", "playground"]
    assert robot["default_configuration"] == "stock"
    for entry in robot["configurations"]:
        profile = json.loads((ROOT / entry["path"]).read_text())
        assert profile["name"] == entry["name"]
        assert profile["robot"] == robot["id"]
        assert profile["version"] == "1.0.0"
        assert profile["format"] == "livingproof-configuration-profile/0.1"


def test_runner_pin_is_a_released_tag():
    """The simulator dependency is a release tag. It is not a branch, a pull request, or a raw commit."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    source = pyproject["tool"]["uv"]["sources"]["livingproof-sim-mujoco"]
    tag = source["tag"]
    assert re.fullmatch(r"v\d+\.\d+\.\d+", tag)
    assert "branch" not in source and "rev" not in source
    uv_lock = (ROOT / "uv.lock").read_text()
    assert re.search(rf"livingproof-sim-mujoco\.git\?tag={tag}#[0-9a-f]{{40}}", uv_lock)


def test_adapter_scene_is_the_published_model():
    """The adapter compiles the pinned model and does not attach a world."""
    for name in ("stock", "playground"):
        scene = ROBOT.scene(name)
        assert not hasattr(scene, "walls") and not hasattr(scene, "world")
        assert scene.model.nbody > 1
    assert not hasattr(ROBOT, "world")
    settings = ROBOT.runner_settings()
    assert settings["sample_hz"] == 50
    assert "scenario" not in settings and "world" not in settings


def _assert_workflow_inputs_are_documentation_only(workflows):
    assert sorted(workflows) == ["docs.yml", "release.yml", "test.yml"]
    for name, text in workflows.items():
        document = yaml.safe_load(text)
        triggers = document.get("on", document.get(True))
        assert "pull-requests:" not in text
        if name != "docs.yml":
            assert "workflow_call" not in triggers
            assert "workflow_dispatch" not in triggers
            continue
        # Reuse is confined to exact release identity, never a scenario/world
        # selection or a downstream integration/verification input.
        assert triggers["workflow_call"] == {
            "inputs": {
                field: {"required": True, "type": "string"}
                for field in ("release_tag", "release_version", "release_commit")
            }
        }
        dispatch = triggers["workflow_dispatch"]
        assert set(dispatch) == {"inputs"}
        assert set(dispatch["inputs"]) == {"release_tag"}
        replay = dispatch["inputs"]["release_tag"]
        assert replay["type"] == "string" and replay["default"] == ""
        assert set(replay) <= {"description", "type", "default", "required"}
        assert replay.get("required", False) is False


def test_workflows_do_not_select_a_scenario():
    workflows = {path.name: path.read_text() for path in (ROOT / ".github/workflows").iterdir()}
    _assert_workflow_inputs_are_documentation_only(workflows)
    assert not (ROOT / "verification.lock.json").exists()
    assert not (ROOT / "tools/repin.py").exists() and not (ROOT / "tools/pr_comment.py").exists()
    assert not (ROOT / "src/lpl_t1/fetch.py").exists()


def test_robot_repository_does_no_grading():
    package = Path(lpl_t1.__file__).parent
    assert not (package / "clearance.py").exists() and not (package / "stats.py").exists()
    assert not (package / "scenario.py").exists() and not (package / "evidence.py").exists()
    for name in ("evaluate", "grade", "compare", "interval", "manifest"):
        assert not hasattr(lpl_t1, name), name
        assert not hasattr(ROBOT, name), name


def test_adapter_is_the_published_entry_point():
    ref = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["livingproof"]["adapter"]
    module, name = ref.split(":")
    adapter = getattr(importlib.import_module(module), name)
    assert ref == "lpl_t1.adapter:ROBOT" and adapter is ROBOT and adapter.root == ROOT
    assert adapter.id == "booster-t1"
    taskfile = (ROOT / "Taskfile.yml").read_text()
    for removed in ("repin", "repin:check", "setup", "fetch", "check", "test", "evidence"):
        assert f"\n  {removed}:\n" not in taskfile, removed


@pytest.mark.parametrize(
    ("filename", "trigger", "field"),
    [
        ("docs.yml", "workflow_call", "scenario"),
        ("docs.yml", "workflow_call", "world"),
        ("docs.yml", "workflow_call", "verifier"),
        ("docs.yml", "workflow_dispatch", "scenario"),
        ("docs.yml", "workflow_dispatch", "world"),
        ("release.yml", "workflow_call", "scenario"),
        ("test.yml", "workflow_call", "scenario"),
        ("test.yml", "workflow_dispatch", "scenario"),
    ],
)
def test_workflow_boundary_rejects_integration_inputs(filename, trigger, field):
    workflows = {path.name: path.read_text() for path in (ROOT / ".github/workflows").iterdir()}
    document = yaml.safe_load(workflows[filename])
    triggers = document.get("on", document.get(True))
    inputs = triggers.setdefault(trigger, {}).setdefault("inputs", {})
    inputs[field] = {"required": True, "type": "string"}
    workflows[filename] = yaml.safe_dump(document)
    with pytest.raises(AssertionError):
        _assert_workflow_inputs_are_documentation_only(workflows)
