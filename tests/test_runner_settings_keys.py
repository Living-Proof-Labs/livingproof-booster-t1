# SPDX-License-Identifier: Apache-2.0
"""runner.json holds robot timing only, the same keys as the G1 and Go1 packages.

wall_contact_model described a contact check against world walls. This package
adds no walls (Scene.walls is empty) and no code reads the key.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runner_settings_are_robot_timing_only():
    runner = json.loads((ROOT / "runner.json").read_text(encoding="utf-8"))
    assert set(runner) == {"$comment", "sample_hz", "stand_seconds"}
    assert "wall" not in json.dumps(runner).lower()
