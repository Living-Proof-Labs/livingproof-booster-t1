# SPDX-License-Identifier: Apache-2.0
"""The Booster T1 MuJoCo models.

booster_gym's T1_locomotion.xml (stock) and MuJoCo Playground's flat-terrain T1
scene (playground) are loaded unchanged: timestep, solver settings, collision
geoms, ground plane and actuators stay as published. This module does not load
a world.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
from livingproof_sim_mujoco.pins import load_pins, local
from livingproof_sim_mujoco.scene import actuator_table

from lpl_t1 import ROOT

SCENE = "resources/T1/T1_locomotion.xml"
PLAYGROUND_SCENE = "mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml"
MODELS = {"booster_gym": ("booster_gym", SCENE), "playground": ("playground", PLAYGROUND_SCENE)}
BASE_BODY = "Trunk"


@dataclass
class Scene:
    model: mujoco.MjModel
    world: dict
    ctrlrange: dict[str, tuple[float, float]]  # motor torque limits, by actuator name
    walls: list[str]


def model_path(model: str = "booster_gym", root: Path = ROOT) -> Path:
    """The pinned MJCF of a configuration's ``settings.model``."""
    group, path = MODELS[model]
    return local(group, path, root)


def raw_model(root: Path = ROOT, model: str = "booster_gym") -> mujoco.MjModel:
    load_pins(root)
    return mujoco.MjModel.from_xml_path(str(model_path(model, root)))


def build(root: Path = ROOT, model: str = "booster_gym") -> Scene:
    """Compile a pinned model unchanged. No world geometry is added."""
    load_pins(root)
    spec = mujoco.MjSpec.from_file(str(model_path(model, root)))
    compiled = spec.compile()
    ctrlrange = actuator_table(compiled)[1]
    return Scene(compiled, {}, ctrlrange, [])
