# SPDX-License-Identifier: Apache-2.0
"""The robot manifest declares the scenarios this adapter supports.

robot-capability-specs v2.3.0: ``scenarios`` on livingproof-robot-manifest/0.1
names the scenario ids the adapter supports. This package declares walk-hallway
and walk-corner, which the adapter drives itself (``drive``, robot-capability-specs#40
step 11b).
"""

import json

import pytest

from lpl_t1 import ROOT
from lpl_t1.profiles import validate

SPECS_COMMIT = "57317e617014a6df97bb7fba9d92a890af8d42e5"
SCHEMA = "specs/robots/robot-manifest.schema.json"
DECLARED = ["walk-hallway", "walk-corner"]


def test_manifest_declares_walk_hallway_and_walk_corner_and_validates_against_the_spec():
    """walk-hallway and walk-corner, validated by the pinned v2.3.0 robot-manifest schema."""
    document = json.loads((ROOT / "manifest.json").read_text())
    assert document["format"] == "livingproof-robot-manifest/0.1"
    assert document["scenarios"] == DECLARED

    specs = json.loads((ROOT / "pins.json").read_text())["groups"]["specs"]
    assert specs["tag"] == "v2.3.0"
    assert specs["commit"] == SPECS_COMMIT
    assert SCHEMA in specs["files"]

    validate("robot_manifest", document)
    illegal = {**document, "scenarios": ["Walk-Hallway"]}
    with pytest.raises(ValueError, match="robot_manifest"):
        validate("robot_manifest", illegal)
