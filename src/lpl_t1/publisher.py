# SPDX-License-Identifier: Apache-2.0
"""booster_gym's play_mujoco.py control loop (da396a0), without the viewer.

Verbatim except: the stdin command prompt is a constant ``command`` (gait frequency set
exactly as its prompt handler does); the viewer is removed; and the policy is the
published TorchScript ``deploy/models/T1.pt`` instead of a training checkpoint (.pth).

Checkpoint swap: play_mujoco.py loads ``ActorCritic`` weights from a .pth training
checkpoint and uses ``model.act(obs).loc``, the mean of ``Normal(actor(obs), std)``, which
is exactly ``actor(obs)``. booster_gym publishes no .pth; its export_model.py writes
``torch.jit.script(model.actor)``, which is T1.pt. So ``T1.pt(obs)`` is the same function
play_mujoco.py evaluates.

Options for isolating differences, all off by default (verbatim): ``dtype=np.float64``
replaces its float32 casts; ``fresh_sensors`` calls mj_forward before reading sensors
(after mj_step they hold values computed before integration, one physics step stale);
``clip=False`` drops its action clip.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

from lpl_t1 import ROOT
from lpl_t1.fetch import load_pins, local
from lpl_t1.scene import SCENE

TRAINING_CONFIG = "envs/T1.yaml"


def quat_rotate_inverse(q, v):  # play_mujoco.py, verbatim
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def training_config(root: Path = ROOT) -> dict:
    load_pins(root)
    return yaml.load(
        local("booster_gym", TRAINING_CONFIG, root).read_text(), Loader=yaml.FullLoader
    )


def play(
    command,
    steps: int,
    root: Path = ROOT,
    *,
    dtype=np.float32,
    fresh_sensors: bool = False,
    clip: bool = True,
    qpos0=None,
    qvel0=None,
):
    """Run ``steps`` physics steps. Returns (policy log [(it, obs, action)], base heights
    per step, final qpos). ``qpos0``/``qvel0`` override the initial state after the loop's
    own setup (used for seeded starts)."""
    import torch

    cfg = training_config(root)
    policy = torch.jit.load(
        str(local("booster_gym", "deploy/models/T1.pt", root)), map_location="cpu"
    )
    policy.eval()

    mj_model = mujoco.MjModel.from_xml_path(str(local("booster_gym", SCENE, root)))
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=dtype)
    dof_stiffness = np.zeros(mj_model.nu, dtype=dtype)
    dof_damping = np.zeros(mj_model.nu, dtype=dtype)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]
        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {i} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=dtype),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=dtype),
            default_dof_pos,
        ]
    )
    if qpos0 is not None:
        mj_data.qpos[:] = qpos0
    if qvel0 is not None:
        mj_data.qvel[:] = qvel0
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=dtype)
    lin_vel_x, lin_vel_y, ang_vel_yaw = (float(v) for v in command)
    if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:  # the prompt handler
        gait_frequency = 0
    else:
        gait_frequency = np.average(cfg["commands"]["gait_frequency"])
    gait_process = 0.0
    it = 0
    log, heights = [], []
    trunk = mj_model.body("Trunk").id
    while it < steps:
        if fresh_sensors:
            mujoco.mj_forward(mj_model, mj_data)
        dof_pos = mj_data.qpos.astype(dtype)[7:]
        dof_vel = mj_data.qvel.astype(dtype)[6:]
        quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(dtype)
        base_ang_vel = mj_data.sensor("angular-velocity").data.astype(dtype)
        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        if it % cfg["control"]["decimation"] == 0:
            obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
            obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
            obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
            obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
            obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
            obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
            obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
            obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
            obs[35:47] = actions
            with torch.no_grad():  # T1.pt = actor, i.e. model.act(obs).loc
                actions[:] = policy(torch.tensor(obs).unsqueeze(0)).detach().numpy()
            raw = actions.copy()
            if clip:
                lim = cfg["normalization"]["clip_actions"]
                actions[:] = np.clip(actions, -lim, lim)
            dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            log.append((it, obs.copy(), raw))
        mj_data.ctrl = np.clip(
            dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[:, 0],
            mj_model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(mj_model, mj_data)
        heights.append(float(mj_data.xpos[trunk][2]))
        it += 1
        gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)
    return log, heights, mj_data.qpos.copy()


def stand(seconds: float = 10.0, root: Path = ROOT) -> dict:
    """Zero-command stand in the publisher's loop (gait gated off, as play_mujoco.py)."""
    dt = training_config(root)["sim"]["dt"]
    log, z, q = play((0.0, 0.0, 0.0), int(round(seconds / dt)), root)
    last = z[-int(round(1.0 / dt)) :]
    return {
        "check": "zero-command-stand (publisher loop)",
        "seconds": seconds,
        "standing": min(z) > 0.3,
        "min_trunk_height": round(min(z), 4),
        "final_trunk_height_mean_1s": round(sum(last) / len(last), 4),
        "drift_xy_m": [round(float(v), 4) for v in q[:2]],
        "max_abs_action": round(float(max(np.abs(a).max() for _, _, a in log)), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["stand"])
    ap.parse_args(argv)
    print(json.dumps(stand(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
