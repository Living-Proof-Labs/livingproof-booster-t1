# SPDX-License-Identifier: Apache-2.0
"""Our runner against play_t1_joystick.py's own loop (mujoco_playground 12ad7f3).

The reference is its ``OnnxController``, verbatim except the gamepad is a constant command
and the viewer is a plain ``mj_step`` loop. The control callback runs inside ``mj_step`` on
the Playground MJCF's <position> actuators from ``mj_resetDataKeyframe(model, data, 0)``,
as the script does. The scene is the flat-terrain one (see configurations/playground.json
``scene_note``).

``get_obs`` does ``joint_velocities = data.qvel[6:]`` and then
``joint_velocities[:2] *= 0.0``: ``data.qvel[6:]`` is a view, so every policy step the
publisher's loop also sets the two head joint velocities of the simulation state to zero.
The runner zeroes them in the observation only (ObsTerm scale) and cannot write state.
``verbatim`` keeps that side effect; ``copy`` takes ``data.qvel[6:].copy()`` and is
otherwise identical. That is the one deviation from the script, as the fresh sensor read is
for stock: the graded reference is ``copy``; ``verbatim`` is reported, not graded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from lpl_t1 import ROOT
from lpl_t1.profiles import configuration, controller
from lpl_t1.scene import Scene, raw_model

TERMS = [
    ("linvel", 3),
    ("gyro", 3),
    ("gravity", 3),
    ("command", 3),
    ("joint_pos", 23),
    ("joint_vel", 23),
    ("last_action", 23),
    ("phase", 4),
]


class OnnxController:
    """play_t1_joystick.py OnnxController; the Gamepad is replaced by ``command``."""

    def __init__(
        self,
        policy_path,
        default_angles,
        ctrl_dt,
        n_substeps,
        command,
        action_scale=0.5,
        qvel_view=True,
    ):
        import onnxruntime as rt

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)
        self._counter = 0
        self._n_substeps = n_substeps
        self._ctrl_dt = ctrl_dt
        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt
        self._command = np.asarray(command, dtype=np.float64)
        self._qvel_view = qvel_view
        self.log = []

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:] if self._qvel_view else data.qvel[6:].copy()
        command = self._command
        ph = self._phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
        phase = np.concatenate([np.cos(ph), np.sin(ph)])
        joint_angles[:2] *= 0.0
        joint_velocities[:2] *= 0.0
        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity,
                command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        )
        return obs.astype(np.float32)

    def get_control(self, model, data) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
            self.log.append((self._counter - 1, obs, onnx_pred.copy()))
            phase_tp1 = self._phase + self._phase_dt
            self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def _setup(root):
    model = raw_model(root, "playground")
    model.opt.timestep = 0.002  # play_t1_joystick.py: sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    c = controller("playground", Scene(model, {}, {}, []), root)
    return model, data, c


def reference(command, ticks, root=ROOT, qvel_view=True):
    model, data, c = _setup(root)
    ctrl_dt, sim_dt = 0.02, 0.002
    n_substeps = int(round(ctrl_dt / sim_dt))
    pol = OnnxController(
        str(c.policy_path),
        np.array(model.keyframe("home").qpos[7:]),
        ctrl_dt,
        n_substeps,
        command,
        action_scale=1.0,
        qvel_view=qvel_view,
    )
    mujoco.set_mjcb_control(pol.get_control)
    try:
        while len(pol.log) < ticks:
            mujoco.mj_step(model, data)
    finally:
        mujoco.set_mjcb_control(None)
    return pol.log[:ticks], data.qpos.copy()


def ours(command, ticks, root=ROOT):
    from livingproof_sim_mujoco.policy import OnnxPolicy

    model, data, c = _setup(root)
    policy = OnnxPolicy(model, c.policy_path, c.spec)
    log, backend = [], policy.backend

    def recording(obs):
        out = backend(obs)
        log.append((policy.counter, np.asarray(obs).copy(), np.asarray(out).copy()))
        return out

    policy.backend = recording
    while len(log) < ticks:
        policy.step(data, command)
    return log[:ticks], data.qpos.copy()


def compare(ref, got, ref_q, got_q) -> dict:
    first, worst = None, {"obs": 0.0, "action": 0.0}
    per_term = {n: 0.0 for n, _ in TERMS}
    for k, ((ri, ro, ra), (gi, go, ga)) in enumerate(zip(ref, got)):
        d_obs, d_act = np.abs(ro - go), np.abs(np.ravel(ra) - np.ravel(ga))
        worst["obs"] = max(worst["obs"], float(d_obs.max()))
        worst["action"] = max(worst["action"], float(d_act.max()))
        o, by = 0, {}
        for name, n in TERMS:
            by[name] = float(d_obs[o : o + n].max())
            per_term[name] = max(per_term[name], by[name])
            o += n
        if first is None and (ri != gi or d_obs.max() > 0 or d_act.max() > 0):
            first = {
                "tick": k,
                "reference_step": int(ri),
                "runner_step": int(gi),
                "max_obs_diff_by_term": by,
                "max_action_diff": float(d_act.max()),
            }
    return {
        "ticks": len(ref),
        "first_policy_step": {"reference": int(ref[0][0]), "runner": int(got[0][0])},
        "bit_exact": first is None,
        "first_divergence": first,
        "max_abs_diff": worst,
        "max_obs_diff_by_term": per_term,
        "final_base_xy": {
            "reference": [round(float(v), 4) for v in ref_q[:2]],
            "runner": [round(float(v), 4) for v in got_q[:2]],
        },
    }


def run(command, ticks: int, root=ROOT) -> dict:
    """Pass rule, as the G1 playground reference: bit-exact observations and actions, here
    against the ``copy`` loop. The ``verbatim`` loop (head qvel written to 0) is reported."""
    got, gq = ours(command, ticks, root)
    ref, rq = reference(command, ticks, root, qvel_view=False)
    out = compare(ref, got, rq, gq)
    vref, vq = reference(command, ticks, root, qvel_view=True)
    out["verbatim_reference"] = compare(vref, got, vq, gq)
    out["passed"] = out["bit_exact"]
    out["pass_rule"] = "bit-exact against play_t1_joystick.py with data.qvel[6:].copy()"
    out["reference"] = "play_t1_joystick.py OnnxController @ mujoco_playground 12ad7f3"
    out["configuration_sha256"] = configuration("playground", root)[1]
    out["command"] = list(command)
    return out


def stand(seconds: float = 10.0, root=ROOT) -> dict:
    """Zero-command stand in the play_t1_joystick.py loop (phase held at pi)."""
    model, data, c = _setup(root)
    pol = OnnxController(
        str(c.policy_path),
        np.array(model.keyframe("home").qpos[7:]),
        0.02,
        10,
        (0.0, 0.0, 0.0),
        action_scale=1.0,
    )
    trunk, z = model.body("Trunk").id, []
    mujoco.set_mjcb_control(pol.get_control)
    try:
        for _ in range(round(seconds / 0.002)):
            mujoco.mj_step(model, data)
            z.append(float(data.xpos[trunk][2]))
    finally:
        mujoco.set_mjcb_control(None)
    return {
        "check": "zero-command-stand (play_t1_joystick.py loop)",
        "seconds": seconds,
        "trunk_z_min": round(min(z), 4),
        "trunk_z_last_1s": round(float(np.mean(z[-500:])), 4),
        "standing": min(z) > 0.3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "reference-playground.json")
    ap.add_argument("--stand", action="store_true")
    a = ap.parse_args(argv)
    if a.stand:
        print(json.dumps(stand(), indent=2))
        return 0
    res = run((0.5, 0.0, 0.0), a.ticks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in (("copy (graded)", res), ("verbatim", res["verbatim_reference"])):
        print(
            f"{k}: bit_exact={v['bit_exact']} first={v['first_policy_step']} "
            f"max={v['max_abs_diff']} div={v['first_divergence'] and v['first_divergence']['tick']}"
        )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
