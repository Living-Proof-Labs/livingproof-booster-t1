# SPDX-License-Identifier: Apache-2.0
"""Robot dimensions and mass measured from the pinned model (robot.json source: measured).

Extents of every mesh vertex in world coordinates at the published initial pose
(play_mujoco.py: base at 0.72 m, upright, default joint angles); mass is the sum of
body masses. The arms are fixed to the trunk in this model, so the pose has one width.
"""

from __future__ import annotations

import json

import sys

from livingproof_sim_mujoco.measure import extents, mesh_points
import mujoco

from lpl_t1.profiles import controller, model_of
from lpl_t1.scene import build


def measure(name: str = "stock") -> dict:
    """Mesh extents in a configuration's initial pose on its own pinned model."""
    scene = build(model=model_of(name))
    m = scene.model
    d = mujoco.MjData(m)
    d.qpos[:] = controller(name, scene).initial_qpos
    mujoco.mj_forward(m, d)
    p = mesh_points(m, d)
    ext = extents(p)
    return {
        "width_m": ext[1],
        "length_m": ext[0],
        "height_m": ext[2],
        "lowest_point_m": round(float(p[:, 2].min()), 3),
        "mass_kg": round(float(sum(m.body_mass)), 2),
    }


if __name__ == "__main__":
    print(json.dumps(measure(*sys.argv[1:2]), indent=2))
