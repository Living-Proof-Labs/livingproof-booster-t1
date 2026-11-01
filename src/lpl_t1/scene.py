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
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from lpl_t1 import ROOT
from lpl_t1.fetch import load_pins, local

SCENE = "resources/T1/T1_locomotion.xml"
PLAYGROUND_SCENE = "mujoco_playground/_src/locomotion/t1/xmls/scene_mjx_feetonly_flat_terrain.xml"
MODELS = {"booster_gym": ("booster_gym", SCENE), "playground": ("playground", PLAYGROUND_SCENE)}
WORLD = "worlds/hallway-open/0.1.0/world.json"
PLACE = "place:"
BASE_BODY = "Trunk"


@dataclass
class Scene:
    model: mujoco.MjModel
    world: dict
    ctrlrange: dict[str, tuple[float, float]]  # motor torque limits, by actuator name
    walls: list[str]


def load_world(root: Path = ROOT) -> dict:
    return json.loads(local("worlds", WORLD, root).read_text())


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
    walls = []
    for geom in world["geometry"]:
        if geom["role"] != "wall":
            continue
        g = spec.worldbody.add_geom()
        g.name, g.type = geom["name"], mujoco.mjtGeom.mjGEOM_BOX
        g.size = [s / 2 for s in geom["size"]]
        g.pos = geom["centre"]
        g.contype = g.conaffinity = 0
        g.rgba = [0.55, 0.6, 0.7, 0.45]
        walls.append(geom["name"])
    for place in world["places"]:
        site = spec.worldbody.add_site()
        site.name = PLACE + place["name"]
        site.pos = place.get("position") or place.get("reference") or [0.0, 0.0, 0.0]
        yaw = math.radians(place.get("yaw_deg", 0.0))
        site.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        site.size = [0.03, 0.03, 0.03]
        site.rgba = [0.1, 0.6, 0.2, 0.5]
    m = spec.compile()
    ctrlrange = {
        m.actuator(i).name: tuple(float(v) for v in m.actuator_ctrlrange[i]) for i in range(m.nu)
    }
    return Scene(m, world, ctrlrange, walls)


def place_pose(model: mujoco.MjModel, name: str) -> tuple[np.ndarray, float]:
    """Position and yaw of a named world place, read from the compiled model."""
    site = model.site(PLACE + name)
    w, _, _, z = model.site_quat[site.id]
    return model.site_pos[site.id].copy(), 2 * math.atan2(z, w)
