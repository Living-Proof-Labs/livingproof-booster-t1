# SPDX-License-Identifier: Apache-2.0
"""Exact match against booster_gym's play_mujoco.py loop (lpl_t1.reference).

Checkpoint swap (documented in lpl_t1.publisher): play_mujoco.py evaluates
``ActorCritic.act(obs).loc`` from a .pth training checkpoint, which booster_gym does not
publish. Its export_model.py saves ``torch.jit.script(model.actor)`` as T1.pt, and
``act(obs).loc`` is ``actor(obs)``, so T1.pt is the same function. The reference loop
uses T1.pt for that reason; nothing else in the loop changes.
"""

import numpy as np
import pytest
import torch

from lpl_t1.fetch import local
from lpl_t1.reference import run

TICKS = 100  # policy steps, 2 s of walking


@pytest.fixture(scope="module")
def walk():
    return run((0.5, 0.0, 0.0), TICKS)


def test_checkpoint_swap_t1_pt_is_the_actor_mean():
    """T1.pt is a scripted actor MLP (47 -> 12) with no sampling: deterministic, and
    export_model.py scripts exactly model.actor."""
    pt = torch.jit.load(str(local("booster_gym", "deploy/models/T1.pt")), map_location="cpu")
    src = local("booster_gym", "export_model.py").read_text()
    assert "torch.jit.script(model.actor)" in src
    assert "dist.loc" in local("booster_gym", "play_mujoco.py").read_text()
    x = torch.randn(4, 47)
    with torch.no_grad():
        assert torch.equal(pt(x), pt(x)) and pt(x).shape == (4, 12)
    names = [n for n, _ in pt.named_parameters()]
    assert names == [f"{i}.{p}" for i in (0, 2, 4, 6) for p in ("weight", "bias")]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "play_mujoco.py reads sensors after mj_step, i.e. values computed before that "
        "step's integration (one 0.002 s physics step stale); runner v1.4.0 reads them "
        "after mj_step1 of the current step. Gyro and gravity differ from the 2nd policy "
        "step on. Reported as a FAIL; the runner is not changed here."
    ),
)
def test_runner_matches_play_mujoco_exactly(walk):
    assert walk["variants"]["float64"]["first_policy_step"] == {"reference": 0, "runner": 0}
    assert walk["passed"], walk["variants"]["float64"]["first_divergence"]


def test_the_only_difference_is_the_stale_sensor_read(walk):
    v = walk["variants"]
    f64 = v["float64"]["first_divergence"]
    assert f64["tick"] == 1
    assert set(k for k, d in f64["max_obs_diff_by_term"].items() if d > 0) == {
        "gravity",
        "ang_vel",
    }
    fresh = v["float64_fresh_sensors"]
    assert fresh["max_abs_diff"]["action"] == 0.0 and fresh["max_abs_diff"]["obs"] < 1e-12
    assert fresh["passes_rule"]


def test_action_clip_is_never_reached(walk):
    """play_mujoco.py clips to +/-1 and the runner does not; on this walk no action
    reaches the limit, so the clip changes nothing."""
    v = walk["variants"]
    assert v["float64_fresh_sensors"]["max_abs_action_reference"] < 1.0
    assert (
        v["float64_fresh_sensors_no_clip"]["max_abs_diff"]
        == (v["float64_fresh_sensors"]["max_abs_diff"])
    )
    assert np.isfinite(v["float32_verbatim"]["max_abs_diff"]["obs"])
