# SPDX-License-Identifier: Apache-2.0
"""The Booster T1 adapter for livingproof-sim-mujoco's generic walk-hallway evidence loop.
Only robot-specific code and data live here: the per-configuration model (booster_gym or
playground), the published controllers, the seeded initial pose and the profiles."""

from __future__ import annotations

from pathlib import Path

from lpl_t1 import ROOT


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

    def initial_state(self, model, data, ctrl, seed, noise, spawn) -> None:
        from lpl_t1.walk import initial_state

        initial_state(model, data, ctrl, seed, noise, spawn)

    def runner_settings(self) -> dict:
        from lpl_t1.walk import runner_settings

        return runner_settings(self.root)

    def robot(self) -> dict:
        from lpl_t1.profiles import robot

        return robot(self.root)

    def configuration(self, name: str) -> dict:
        from lpl_t1.profiles import configuration

        return configuration(name, self.root)[0]

    def world(self) -> dict:
        from lpl_t1.scene import load_world

        return load_world(self.root)

    def environment(self, name: str) -> dict:
        from lpl_t1.profiles import model_of

        return {"model": model_of(name, self.root)}


ROBOT = T1()
