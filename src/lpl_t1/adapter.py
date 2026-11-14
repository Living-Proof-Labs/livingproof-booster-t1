# SPDX-License-Identifier: Apache-2.0
"""The Booster T1 adapter for livingproof-sim-mujoco.

Only robot-specific code and data live here: the per-configuration model
(booster_gym or playground), the published controllers, the nominal initial
pose and the profiles. The adapter does not select a world or a scenario.
A caller that has already compiled a named place into the model still gets
that place as the start pose.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from lpl_t1 import ROOT


def _start_offset(start_offset) -> tuple[np.ndarray, float]:
    """One declared xy/yaw change. Absent means the exact nominal pose."""
    if start_offset is None:
        return np.zeros(2), 0.0
    if not isinstance(start_offset, dict):
        raise TypeError(
            "start_offset must be a dict with base_xy_m and/or base_yaw_rad, "
            f"got {type(start_offset).__name__}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    unknown = set(start_offset) - {"base_xy_m", "base_yaw_rad"}
    if unknown:
        raise ValueError(
            "start_offset only accepts base_xy_m and base_yaw_rad "
            f"(one named comparison change), got {sorted(unknown)}. "
            "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
        )
    dxy = np.zeros(2)
    if "base_xy_m" in start_offset:
        raw = start_offset["base_xy_m"]
        if len(raw) != 2:
            raise ValueError(
                "start_offset base_xy_m must be [x, y] in metres. "
                "Example: start_offset={'base_xy_m': [0.0, 0.05]}"
            )
        dxy[:] = raw
    dyaw = float(start_offset.get("base_yaw_rad", 0.0))
    return dxy, dyaw


def _nominal_state(model, data, ctrl, start_offset) -> None:
    """Published nominal pose, plus one declared start offset. Velocities are zero."""
    dxy, dyaw = _start_offset(start_offset)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = ctrl.initial_qpos
    data.qpos[0:2] = data.qpos[0:2] + dxy
    if dyaw:
        w, _x, _y, z = (float(v) for v in data.qpos[3:7])
        yaw = 2 * math.atan2(z, w) + dyaw
        data.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data.qvel[:] = 0.0
    if ctrl.spec.actuation == "position":
        for name, q in zip(ctrl.spec.actuators, ctrl.spec.default_pose):
            data.ctrl[model.actuator(name).id] = q
    mujoco.mj_forward(model, data)


class T1:
    id = "booster-t1"
    root: Path = ROOT
    width_decimals = 3

    def base_body(self, name: str) -> str:
        from lpl_t1.scene import BASE_BODY

        return BASE_BODY

    def scene(self, name: str):
        from lpl_t1.profiles import model_of
        from lpl_t1.scene import build

        return build(self.root, model_of(name, self.root))

    def controller(self, name: str, scene):
        from lpl_t1.profiles import controller

        return controller(name, scene, self.root)

    def policy(self, model, ctrl):
        from livingproof_sim_mujoco.policy import OnnxPolicy

        return OnnxPolicy(model, ctrl.policy_path, ctrl.spec)

    def initial_state(self, model, data, ctrl, spawn: str, *, start_offset=None) -> None:
        """Exact nominal pose. A place site already on the model is applied.

        Signature matches livingproof-sim-mujoco ``Robot.initial_state``:
        ``(model, data, ctrl, spawn, start_offset=None)``. There is no seed
        and no noise. This method does not load a world. When the compiled
        model has no site for ``spawn``, the published nominal pose is used.
        """
        from livingproof_sim_mujoco.scene import PLACE
        from livingproof_sim_mujoco.walk import initial_state

        site = f"{PLACE}{spawn}"
        names = {model.site(i).name for i in range(model.nsite)}
        if site in names:
            initial_state(model, data, ctrl, spawn, start_offset=start_offset)
            return
        _nominal_state(model, data, ctrl, start_offset)

    def runner_settings(self) -> dict:
        from livingproof_sim_mujoco.walk import runner_settings

        return runner_settings(self.root)

    def robot(self) -> dict:
        from lpl_t1.profiles import robot

        return robot(self.root)

    def configuration(self, name: str) -> dict:
        from lpl_t1.profiles import configuration

        return configuration(name, self.root)[0]

    def environment(self, name: str) -> dict:
        from lpl_t1.profiles import model_of

        return {"model": model_of(name, self.root)}


ROBOT = T1()
