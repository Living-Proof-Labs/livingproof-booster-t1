# SPDX-License-Identifier: Apache-2.0
"""Robot dimensions and mass measured from the pinned model (robot.json source: measured).

Extents of every mesh vertex in world coordinates at the published initial pose
(play_mujoco.py: base at 0.72 m, upright, default joint angles); mass is the sum of
body masses. The arms are fixed to the trunk in this model, so the pose has one width.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from lpl_t1.profiles import controller
from lpl_t1.scene import build


def measure() -> dict:
    scene = build()
    m = scene.model
    d = mujoco.MjData(m)
    d.qpos[:] = controller("stock", scene).initial_qpos
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    p = np.vstack(pts)
    ext = (p.max(0) - p.min(0)).round(3).tolist()
    return {
        "width_m": ext[1],
        "length_m": ext[0],
        "height_m": ext[2],
        "lowest_point_m": round(float(p[:, 2].min()), 3),
        "mass_kg": round(float(sum(m.body_mass)), 2),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
