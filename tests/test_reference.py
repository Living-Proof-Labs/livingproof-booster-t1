# SPDX-License-Identifier: Apache-2.0
"""Exact match against booster_gym's play_mujoco.py loop (lpl_t1.reference).

The reference is play_mujoco.py with three documented deviations:

1. Fresh sensor reads: play_mujoco.py reads the orientation and gyro sensors right after
   mj_step, i.e. values computed before that step's integration (one 0.002 s physics step
   stale). The reference calls mj_forward first, so they describe the current state, as
   the runner reads them. Runner v1.4.0 is not changed.
2. Checkpoint swap: play_mujoco.py evaluates ``ActorCritic.act(obs).loc`` from a .pth
   training checkpoint, which booster_gym does not publish. Its export_model.py saves
   ``torch.jit.script(model.actor)`` as T1.pt, and ``act(obs).loc`` is ``actor(obs)``, so
   T1.pt is the same function.
3. Phase arithmetic: the reference advances the gait phase as the runner does (radians,
   once per policy step) instead of play_mujoco.py's per-physics-step fraction; the same
   phase rounded differently. With it, observations and actions are exactly equal.
"""

import numpy as np
import pytest
import torch

from livingproof_sim_mujoco import pins

from lpl_t1 import ROOT
from lpl_t1.publisher import play
from lpl_t1.reference import _initial_qpos, compare, ours, run


def local(group: str, path: str):
    return pins.local(group, path, ROOT)


GRADED = {"dtype": np.float64, "fresh_sensors": True, "runner_phase": True}

TICKS = 100  # policy steps, 2 s of walking


@pytest.fixture(scope="module")
def walk():
    return run((0.5, 0.0, 0.0), TICKS)


def test_checkpoint_swap_t1_pt_is_the_actor_mean():
    pt = torch.jit.load(str(local("booster_gym", "deploy/models/T1.pt")), map_location="cpu")
    assert "torch.jit.script(model.actor)" in local("booster_gym", "export_model.py").read_text()
    assert "dist.loc" in local("booster_gym", "play_mujoco.py").read_text()
    x = torch.randn(4, 47)
    with torch.no_grad():
        assert torch.equal(pt(x), pt(x)) and pt(x).shape == (4, 12)
    names = [n for n, _ in pt.named_parameters()]
    assert names == [f"{i}.{p}" for i in (0, 2, 4, 6) for p in ("weight", "bias")]


def test_runner_matches_play_mujoco_with_fresh_sensor_reads(walk):
    v = walk["variants"]["float64"]
    assert v["first_policy_step"] == {"reference": 0, "runner": 0}
    assert v["bit_exact"] and walk["passed"]
    assert v["max_abs_diff"] == {"obs": 0.0, "action": 0.0}  # exact ==, no tolerance


def test_publisher_phase_arithmetic_is_the_only_rounding_difference(walk):
    v = walk["variants"]["float64_publisher_phase"]
    assert v["max_abs_diff"]["action"] == 0.0
    assert 0.0 < v["max_abs_diff"]["obs"] < 1e-12
    assert {k for k, d in v["max_obs_diff_by_term"].items() if d > 0} == {"phase"}


def _logs(steps=100, q0=None):
    cmd = (0.5, 0.0, 0.0)
    ref, _, rq = play(cmd, steps, **GRADED)
    q0 = _initial_qpos(np.float64) if q0 is None else q0
    got, gq = ours(cmd, steps, q0)
    return ref, got, rq, gq


def test_a_one_ulp_perturbation_is_detected():
    """The comparison is sharp: 1 ULP on one joint angle of the runner's start fails it."""
    q0 = _initial_qpos(np.float64).copy()
    q0[7] = np.nextafter(q0[7], np.inf)
    d = compare(*_logs(200, q0))
    assert not d["bit_exact"] and d["first_divergence"] is not None


def test_three_runs_are_bit_identical():
    runs = [_logs(200) for _ in range(3)]
    key = [
        b"".join(o.tobytes() + np.asarray(a).tobytes() for _, o, a in log) + q.tobytes()
        for ref, got, rq, gq in runs
        for log, q in ((ref, rq), (got, gq))
    ]
    assert key[0::2] == [key[0]] * 3 and key[1::2] == [key[1]] * 3
    assert compare(*runs[0])["bit_exact"]


def test_the_stale_read_is_the_only_deviation_from_the_verbatim_loop(walk):
    f64 = walk["variants"]["verbatim_stale_float64"]["first_divergence"]
    assert f64["tick"] == 1
    assert {k for k, d in f64["max_obs_diff_by_term"].items() if d > 0} == {"gravity", "ang_vel"}


def test_action_clip_is_never_reached(walk):
    """play_mujoco.py clips to +/-1 and the runner does not; no action reaches the limit."""
    v = walk["variants"]
    assert v["float64"]["max_abs_action_reference"] < 1.0
    assert v["float64_no_clip"]["max_abs_diff"] == v["float64"]["max_abs_diff"]
