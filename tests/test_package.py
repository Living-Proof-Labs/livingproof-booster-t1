# SPDX-License-Identifier: Apache-2.0
import io
import json

import mujoco
import numpy as np
import pytest

from lpl_t1 import ROOT
from lpl_t1.fetch import ChecksumError, fetch, local, sha256
from lpl_t1.measure import measure
from lpl_t1.profiles import Unsupported, configuration, controller, robot, validate
from lpl_t1.publisher import training_config
from lpl_t1.scene import SCENE, build, place_pose, raw_model

PIN = "da396a06d6eed99e2de72d7749c48ee8748950f9"


def test_profiles_match_the_pinned_v1_5_0_schemas():
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]
    assert pins["specs"]["tag"] == "v1.5.0" and pins["specs"]["commit"].startswith("1df6eaa")
    assert pins["booster_gym"]["commit"] == PIN
    rob = robot()
    assert rob["id"] == "booster-t1" and rob["class"] == "humanoid"
    assert [c["name"] for c in rob["configurations"]] == ["stock"]
    conf, digest = configuration("stock")
    assert digest == sha256(ROOT / "configurations/stock.json")
    assert conf["version"] == "1.0.0"
    with pytest.raises(ValueError, match="robot does not match"):
        validate("robot", {**rob, "extra": 1})
    with pytest.raises(Unsupported):
        configuration("vendor-x")


def test_every_profile_pin_is_a_checksummed_download():
    rob, conf = robot(), configuration("stock")[0]
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]["booster_gym"]["files"]
    s = conf["settings"]
    for p in (
        rob["model_source"],
        conf["controller"]["artifact"],
        *(s[k] for k in ("training_config", "reference", "export")),
    ):
        assert p["commit"] == PIN and p["license"].startswith("Apache-2.0")
        assert pins[p["path"]]["sha256"] == p["sha256"] == sha256(local("booster_gym", p["path"]))
    assert "Apache License" in local("booster_gym", "LICENSE").read_text()


def test_settings_equal_the_published_training_config():
    cfg = training_config()
    t = configuration("stock")[0]["settings"]["training"]
    assert t["control"] == cfg["control"] and t["sim"]["dt"] == cfg["sim"]["dt"]
    for section in ("init_state", "normalization", "commands"):
        assert t[section] == {k: cfg[section][k] for k in t[section]}, section


def test_runner_spec_uses_the_published_layout_gains_pose_and_rates():
    scene = build()
    m = scene.model
    c = controller("stock", scene)
    names = [m.actuator(i).name for i in range(m.nu)]
    assert list(c.spec.actuators) == names and len(names) == 12
    kp = {"Hip": 200, "Knee": 200, "Ankle": 50}
    kd = {"Hip": 5, "Knee": 5, "Ankle": 1}
    part = [next(k for k in kp if k in n) for n in names]
    assert list(c.spec.kp) == [kp[p] for p in part] and list(c.spec.kd) == [kd[p] for p in part]
    pose = {"Hip_Pitch": -0.2, "Knee_Pitch": 0.4, "Ankle_Pitch": -0.25}
    assert list(c.spec.default_pose) == [
        next((v for k, v in pose.items() if k in n), 0.0) for n in names
    ]
    assert (c.spec.decimation, c.spec.action_scale, c.spec.gait_hz) == (10, 1.0, 1.5)
    assert m.opt.timestep == 0.002 and c.spec.actuation == "pd_torque"
    assert [t.kind for t in c.spec.obs] == [
        "gravity",
        "sensor",
        "command",
        "phase",
        "joint_pos",
        "joint_vel",
        "last_action",
    ]
    np.testing.assert_array_equal(c.initial_qpos[:7], [0, 0, 0.72, 1, 0, 0, 0])


def test_runner_accepts_the_published_policy():
    from livingproof_sim_mujoco.policy import OnnxPolicy

    scene = build()
    c = controller("stock", scene)
    p = OnnxPolicy(scene.model, c.policy_path, c.spec)
    assert p.frame_size == 47


def test_scene_keeps_the_published_model():
    raw, m = raw_model(), build().model
    for attr in ("timestep", "iterations", "integrator", "disableflags", "cone"):
        assert getattr(m.opt, attr) == getattr(raw.opt, attr), attr

    def collisions(model):  # by body: the added world geoms shift geom ids
        return [
            (
                model.body(model.geom_bodyid[g]).name,
                model.geom_contype[g],
                model.geom_conaffinity[g],
            )
            for g in range(model.ngeom)
            if model.geom_bodyid[g] != 0
        ]

    assert collisions(m) == collisions(raw)
    np.testing.assert_array_equal(m.actuator_ctrlrange, raw.actuator_ctrlrange)
    walls = [m.geom(n) for n in build().walls]
    assert [w.name for w in walls] == ["wall-left", "wall-right"]
    assert all(m.geom_contype[w.id] == 0 == m.geom_conaffinity[w.id] for w in walls)
    assert str(local("booster_gym", SCENE)).endswith("T1_locomotion.xml")


def test_spawn_comes_from_the_world_place():
    scene = build()
    world = {p["name"]: p for p in scene.world["places"]}
    pos, yaw = place_pose(scene.model, "spawn-entrance")
    np.testing.assert_allclose(pos, world["spawn-entrance"]["position"])
    assert yaw == pytest.approx(np.radians(world["spawn-entrance"]["yaw_deg"]))


def test_robot_dimensions_are_the_measured_ones():
    got, dims = measure(), robot()["dimensions"]
    for k in ("width", "length", "height"):
        assert dims[k]["value"] == got[f"{k}_m"], k
    assert dims["mass"]["value"] == got["mass_kg"]


def test_fetch_rejects_a_download_with_the_wrong_checksum(tmp_path):
    pins = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": "d",
                "files": {"f.txt": {"sha256": "0" * 64}},
            }
        }
    }
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    with pytest.raises(ChecksumError, match="f.txt"):
        fetch(tmp_path, opener=lambda url: io.BytesIO(b"tampered"))
    assert not (tmp_path / "d" / "f.txt").exists()


def test_scenario_parameters_follow_the_pinned_acceptance_table():
    from lpl_t1 import scenario as sc

    text = local("specs", "specs/scenarios/walk-hallway/README.md").read_text()
    for row in ("| `command_speed` | 0.5 m/s", "Route length 5.5 m", "| `required_passes` | 18"):
        assert row in text, row
    p = sc.parameters("stock")
    assert (p["time_limit"], p["exit_x"], p["fall_height"]) == (27.0, 3.0, 0.44)
    assert "0.05 m humanoid" in text and p["wall_clearance"] == 0.05
    width = robot()["dimensions"]["width"]["value"]
    assert p["lateral_limit"] == pytest.approx(1.5 / 2 - width / 2) == pytest.approx(0.4795)
    pre = {c["name"]: c for c in sc.preconditions("stock")}
    assert all(c["met"] for c in pre.values()), pre
    assert width + 2 * 0.05 <= 1.5 and "0.641 m vs 1.5 m" in pre["fits_width"]["detail"]
    pre = {c["name"]: c["met"] for c in sc.preconditions("vendor-x")}
    assert pre["configuration_offered"] is False


def test_a_robot_too_wide_for_the_hallway_is_unsupported(tmp_path, monkeypatch):
    from lpl_t1 import scenario as sc

    wide = robot()
    wide["dimensions"]["width"]["value"] = 1.41
    monkeypatch.setattr(sc, "robot", lambda root=ROOT: wide)
    pre = {c["name"]: c["met"] for c in sc.preconditions("stock")}
    assert pre["fits_width"] is False


def test_seed_noise_is_deterministic_and_starts_at_the_spawn():
    from lpl_t1.walk import initial_state, runner_settings

    scene = build()
    m, c = scene.model, controller("stock", scene)
    states = []
    for _ in range(2):
        d = mujoco.MjData(m)
        initial_state(m, d, c, 7, runner_settings()["initial_noise"], "spawn-entrance")
        states.append(d.qpos.copy())
    np.testing.assert_array_equal(*states)
    pos, _ = place_pose(m, "spawn-entrance")
    assert abs(states[0][0] - pos[0]) <= 0.02 and states[0][2] == 0.72
