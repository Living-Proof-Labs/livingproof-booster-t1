# SPDX-License-Identifier: Apache-2.0
"""The Booster T1 MuJoCo models plus the pinned hallway world.

booster_gym's T1_locomotion.xml (stock) and MuJoCo Playground's flat-terrain T1 scene
(playground) are loaded unchanged. For booster_gym that means: its timestep, solver settings,
collision geoms, ground plane and <motor> actuators (ctrlrange = published torque limits)
are not touched. The world's walls are added as visual-only boxes (contype 0,
conaffinity 0), so they create no contacts; wall clearance is checked geometrically.
Named places become sites, read back from the compiled model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from livingproof_sim_mujoco import scene as generic
from livingproof_sim_mujoco.scene import PLACE, WORLD, place_pose  # noqa: F401
import mujoco

from lpl_t1 import ROOT
from livingproof_sim_mujoco.pins import load_pins, local

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


def load_world(root: Path = ROOT) -> dict:
    return generic.load_world(root)


def model_path(model: str = "booster_gym", root: Path = ROOT) -> Path:
    """The pinned MJCF of a configuration's ``settings.model``."""
    group, path = MODELS[model]
    return local(group, path, root)


def raw_model(root: Path = ROOT, model: str = "booster_gym") -> mujoco.MjModel:
    load_pins(root)
    return mujoco.MjModel.from_xml_path(str(model_path(model, root)))


def build(root: Path = ROOT, model: str = "booster_gym") -> Scene:
    """Compile a pinned model unchanged plus the hallway's visual walls and places."""
    load_pins(root)
    spec = mujoco.MjSpec.from_file(str(model_path(model, root)))
    world = load_world(root)
    walls = generic.add_world(spec, world)
    m = spec.compile()
    ctrlrange = generic.actuator_table(m)[1]
    return Scene(m, world, ctrlrange, walls)
