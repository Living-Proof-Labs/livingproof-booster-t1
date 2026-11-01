# SPDX-License-Identifier: Apache-2.0
"""Robot and configuration profiles (robot-capability-specs v1.5.0), validated against
the pinned schemas, and the runner spec built from the selected configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from livingproof_sim_mujoco.walk import joint_dofs  # noqa: F401
import numpy as np
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from livingproof_sim_mujoco.policy import ObsTerm, PolicySpec

from lpl_t1 import ROOT
from livingproof_sim_mujoco.pins import load_pins, local, sha256
from lpl_t1.scene import Scene

SCHEMAS = {
    "robot": "specs/robots/robot-profile.schema.json",
    "configuration": "specs/robots/configuration-profile.schema.json",
    "manifest": "specs/robots/run-manifest.schema.json",
}


class Unsupported(Exception):
    """A scenario precondition is not met; ``precondition`` names it."""

    def __init__(self, precondition: str, detail: str):
        super().__init__(f"{precondition}: {detail}")
        self.precondition, self.detail = precondition, detail


def validate(kind: str, document: dict, root: Path = ROOT) -> None:
    loaded = {k: json.loads(local("specs", v, root).read_text()) for k, v in SCHEMAS.items()}
    registry = Registry().with_resources(
        (s["$id"], Resource.from_contents(s)) for s in loaded.values()
    )  # resolve cross-schema $refs from the pinned files, never the network
    validator = Draft202012Validator(loaded[kind], registry=registry)
    errors = sorted(validator.iter_errors(document), key=str)
    if errors:
        raise ValueError(
            f"{kind} does not match {SCHEMAS[kind]}: "
            + "; ".join(
                f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]
            )
        )


def robot(root: Path = ROOT) -> dict:
    doc = json.loads((root / "robot.json").read_text())
    validate("robot", doc, root)
    return doc


def configuration_path(name: str, root: Path = ROOT) -> Path:
    offered = {c["name"]: c["path"] for c in robot(root)["configurations"]}
    if name not in offered:
        raise Unsupported(
            "configuration_offered", f"{name!r} is not in robot.json {sorted(offered)}"
        )
    return root / offered[name]


def configuration(name: str, root: Path = ROOT) -> tuple[dict, str]:
    """The configuration profile and the sha256 of its file bytes."""
    path = configuration_path(name, root)
    doc = json.loads(path.read_text())
    validate("configuration", doc, root)
    if doc["name"] != name or doc["robot"] != robot(root)["id"]:
        raise ValueError(f"{path} names {doc['robot']}/{doc['name']}, expected {name}")
    return doc, sha256(path)


def artifact_path(pin: dict, root: Path = ROOT) -> Path:
    """The pins.json file matching a profile's pinned artifact; checks commit and sha256."""
    for group, g in load_pins(root)["groups"].items():
        if pin["repo"].endswith(g["repository"]) and pin["path"] in g["files"]:
            if g["commit"] != pin["commit"] or g["files"][pin["path"]]["sha256"] != pin["sha256"]:
                raise ValueError(f"{pin['path']}: profile pin differs from pins.json")
            path = local(group, pin["path"], root)
            if sha256(path) != pin["sha256"]:
                raise RuntimeError(
                    f"{path} sha256 mismatch; run uv run pytest (its session fixture fetches the pins)"
                )
            return path
    raise KeyError(f"{pin['repo']} {pin['path']} is not in pins.json")


@dataclass
class Controller:
    configuration: dict
    sha256: str
    policy_path: Path
    spec: PolicySpec
    initial_qpos: np.ndarray  # full qpos, base at the origin; spawn pose applied later


def by_substring(table: dict, name: str, default=None):
    """play_mujoco.py's lookup: every key contained in the actuator name, last one wins."""
    hit = default
    for key, value in table.items():
        if key in name:
            hit = value
    if hit is None:
        raise ValueError(f"no entry of {sorted(table)} matches {name!r}")
    return hit


def controller(name: str, scene: Scene, root: Path = ROOT) -> Controller:
    """Runner spec for one configuration, from the published values in its profile.

    The booster_gym layout (play_mujoco.py, 47 values): projected gravity (imu frame),
    gyro, command (x, y, yaw), cos/sin of the gait phase, joint positions minus the
    default pose, joint velocities x 0.1, last action. Every actuator is a policy joint,
    in actuator order."""
    conf, digest = configuration(name, root)
    s = conf["settings"]
    if s["adapter"] == "playground_joystick":
        return _playground(conf, digest, scene, root)
    if s["adapter"] != "booster_gym":
        raise ValueError(f"unknown adapter {s['adapter']!r}")
    path = artifact_path(conf["controller"]["artifact"], root)
    model = scene.model
    t, nrm, obs = s["training"], s["training"]["normalization"], s["observation"]
    actuators = [model.actuator(i).name for i in range(model.nu)]
    joints = [model.joint(model.actuator_trnid[i][0]).name for i in range(model.nu)]
    angles = dict(t["init_state"]["default_joint_angles"])
    fallback = angles.pop("default")
    default = [float(by_substring(angles, a, fallback)) for a in actuators]
    ctl = t["control"]
    spec = PolicySpec(
        joints=joints,
        actuators=actuators,
        default_pose=default,
        action_scale=ctl["action_scale"],
        decimation=ctl["decimation"],
        obs=[
            ObsTerm("gravity", obs["imu_site"], nrm["gravity"]),
            ObsTerm("sensor", obs["gyro_sensor"], nrm["ang_vel"]),
            ObsTerm("command", scale=[nrm["lin_vel"], nrm["lin_vel"], nrm["ang_vel"]]),
            ObsTerm("phase"),
            ObsTerm("joint_pos", scale=nrm["dof_pos"]),
            ObsTerm("joint_vel", scale=nrm["dof_vel"]),
            ObsTerm("last_action"),
        ],
        gait_hz=float(np.average(t["commands"]["gait_frequency"])),
        gait_offsets=[0.0],
        phase_order="cos_sin",
        actuation="pd_torque",
        kp=[float(by_substring(ctl["stiffness"], a)) for a in actuators],
        kd=[float(by_substring(ctl["damping"], a)) for a in actuators],
        first_tick=0,
    )
    rot = t["init_state"]["rot"]  # xyzw
    qpos = np.zeros(model.nq)
    qpos[0:3] = [0.0, 0.0, t["init_state"]["pos"][2]]
    qpos[3:7] = [rot[3], *rot[0:3]]
    for j, q in zip(joints, default):
        qpos[model.joint(j).qposadr[0]] = q
    return Controller(conf, digest, path, spec, qpos)


def model_of(name: str, root: Path = ROOT) -> str:
    """Which pinned model a configuration runs on (scene.MODELS key)."""
    return configuration(name, root)[0]["settings"].get("model", "booster_gym")


def _playground(conf: dict, digest: str, scene: Scene, root: Path) -> Controller:
    """play_t1_joystick.py's OnnxController as a runner spec, on the MJCF <position>
    actuators. Observation (85): local linvel, gyro, gravity (imu frame), command,
    q - default and dq with the two head joints zeroed, last action, cos/sin of the two
    foot phases. The first action is in physics step 9 (counter += 1 before the check)."""
    s = conf["settings"]
    path = artifact_path(conf["controller"]["artifact"], root)
    model = scene.model
    key = model.key(s["initial_keyframe"]).id
    qpos = model.key_qpos[key].copy()
    actuators = [model.actuator(i).name for i in range(model.nu)]
    joints = [model.joint(model.actuator_trnid[i][0]).name for i in range(model.nu)]
    adr = [int(model.joint(j).qposadr[0]) for j in joints]
    if adr != list(range(7, model.nq)):
        raise ValueError("actuator order is not qpos[7:] order; play_t1_joystick indexes qpos[7:]")
    mask = [0.0 if j in s["observation"]["zeroed_joints"] else 1.0 for j in joints]
    o = s["observation"]
    decimation = round(s["ctrl_dt_s"] / float(model.opt.timestep))
    spec = PolicySpec(
        joints=joints,
        actuators=actuators,
        default_pose=[float(qpos[a]) for a in adr],
        action_scale=s["action_scale"],
        decimation=decimation,
        first_tick=decimation - 1,
        obs=[
            ObsTerm("sensor", o["linvel_sensor"]),
            ObsTerm("sensor", o["gyro_sensor"]),
            ObsTerm("gravity", o["imu_site"]),
            ObsTerm("command"),
            ObsTerm("joint_pos", scale=mask),
            ObsTerm("joint_vel", scale=mask),
            ObsTerm("last_action"),
            ObsTerm("phase"),
        ],
        gait_hz=s["gait_hz"],
        gait_offsets=s["gait_offsets_rad"],
        phase_order="cos_sin",
        actuation="position",
    )
    qpos[0:2] = 0.0
    return Controller(conf, digest, path, spec, qpos)
