# SPDX-License-Identifier: Apache-2.0
"""Shared simulation setup: runner settings and the seeded initial state. Walk-hallway
trials are recorded by ``lpl_t1.evidence`` and graded by the capability verifier, never
here."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from lpl_t1 import ROOT
from lpl_t1.profiles import joint_dofs
from lpl_t1.scene import place_pose


def runner_settings(root: Path = ROOT) -> dict:
    doc = json.loads((root / "runner.json").read_text())
    return {k: v["value"] if isinstance(v, dict) and "value" in v else v for k, v in doc.items()}


def initial_state(model, data, ctrl, seed: int, noise: dict, spawn: str) -> None:
    """The published initial pose at the world's named place plus the seed's noise."""
    rng = np.random.default_rng(seed)
    qadr, vadr = joint_dofs(model)
    # fixed draw order, the same as the G1 package
    dq = rng.uniform(-1, 1, len(qadr)) * noise["joint_pos_rad"]
    dv = rng.uniform(-1, 1, len(vadr)) * noise["joint_vel_rad_s"]
    dxy = rng.uniform(-1, 1, 2) * noise["base_xy_m"]
    dyaw = rng.uniform(-1, 1) * noise["base_yaw_rad"]
    dbase = rng.uniform(-1, 1, 6) * noise["base_vel_m_s"]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ctrl.initial_qpos
    position, yaw = place_pose(model, spawn)
    data.qpos[0:2] = position[0:2] + dxy
    yaw += dyaw
    data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qpos[qadr] += dq
    data.qvel[vadr] = dv
    data.qvel[0:6] = dbase
    if ctrl.spec.actuation == "position":  # hold the default pose until the first tick,
        for name, q in zip(ctrl.spec.actuators, ctrl.spec.default_pose):  # as keyframe ctrl
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)
