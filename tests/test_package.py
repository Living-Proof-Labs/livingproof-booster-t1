# SPDX-License-Identifier: Apache-2.0
import io
import json

import mujoco
import numpy as np
import pytest

from lpl_t1 import ROOT
from livingproof_sim_mujoco import pins
from livingproof_sim_mujoco.pins import ChecksumError, sha256
from lpl_t1.measure import measure
from lpl_t1.profiles import Unsupported, configuration, controller, robot, validate
from lpl_t1.publisher import training_config
from lpl_t1.adapter import ROBOT
from lpl_t1.scene import SCENE, build, raw_model


def local(group: str, path: str):
    return pins.local(group, path, ROOT)


def fetch(root=ROOT, opener=None):
    return pins.fetch(root, opener) if opener else pins.fetch(root)


PIN = "da396a06d6eed99e2de72d7749c48ee8748950f9"


# configuration-profile 1.0.0 as published on main. Any byte change needs a new version.
PUBLISHED_CONFIGURATION_1_0_0 = {
    "configurations/stock.json": "e1626385968e82997bebce743fab19fb2f2661da3def41de2a594051911c324a",
    "configurations/playground.json": "beb97983a415b43f21e4cad0a715692e7d2b76bf1e55b58240871b754e0d4407",
}


def test_configuration_profile_1_0_0_keeps_its_published_digest():
    """One profile version is one file. A comment edit would change the sha256."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    cited = {entry["path"]: entry for entry in manifest["configurations"]}
    for path, digest in PUBLISHED_CONFIGURATION_1_0_0.items():
        assert sha256(ROOT / path) == digest
        assert cited[path]["version"] == "1.0.0"
        assert cited[path]["sha256"] == digest


def test_profiles_match_the_pinned_v2_3_0_schemas():
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]
    assert pins["specs"]["tag"] == "v2.3.0" and pins["specs"]["commit"].startswith("57317e6")
    assert pins["booster_gym"]["commit"] == PIN
    rob = robot()
    assert rob["id"] == "booster-t1" and rob["class"] == "humanoid"
    assert [c["name"] for c in rob["configurations"]] == ["stock", "playground"]
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
    raw, built = raw_model(), build()
    m = built.model
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
    assert m.ngeom == raw.ngeom and built.walls == [] and built.world == {}
    assert str(local("booster_gym", SCENE)).endswith("T1_locomotion.xml")


def test_nominal_base_is_the_published_pose_without_a_world_place():
    """The start pose is the controller's published pose. No world place is compiled in."""
    scene = build()
    names = {scene.model.site(i).name for i in range(scene.model.nsite)}
    assert not any(name.startswith("place:") for name in names)
    c = controller("stock", scene)
    data = mujoco.MjData(scene.model)
    ROBOT.initial_state(scene.model, data, c, "spawn-entrance")
    np.testing.assert_allclose(data.qpos[0:3], [0.0, 0.0, 0.72])
    np.testing.assert_allclose(data.qvel, 0.0)


def test_robot_dimensions_are_the_measured_ones():
    got, dims = measure(), robot()["dimensions"]
    for k in ("width", "length", "height"):
        assert dims[k]["value"] == got[f"{k}_m"], k
    assert dims["mass"]["value"] == got["mass_kg"]


def test_fetch_rejects_a_download_with_the_wrong_checksum(tmp_path):
    """A body of the pinned length with the wrong sha256 is a ChecksumError.

    v3 refuses a pin that omits ``bytes`` with ValueError before any download.
    ``bytes`` here is the length of the body this test supplies, so the runner
    checksums that body instead of stopping on the missing-field error.
    """
    body = b"tampered"
    downloaded = []

    def opener(url):
        downloaded.append(url)
        return io.BytesIO(body)

    pins = {
        "groups": {
            "g": {
                "repository": "o/r",
                "commit": "c",
                "dest": "d",
                "files": {"f.txt": {"sha256": "0" * 64, "bytes": len(body)}},
            }
        }
    }
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    with pytest.raises(ChecksumError, match="f.txt") as caught:
        fetch(tmp_path, opener=opener)
    assert downloaded, "the body was not downloaded, so the checksum was never checked"
    assert "missing bytes" not in str(caught.value)
    assert not (tmp_path / "d" / "f.txt").exists()


def test_fetch_rejects_a_pin_that_omits_bytes(tmp_path):
    """A pin without ``bytes`` is a ValueError and downloads nothing."""
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
    downloaded = []
    with pytest.raises(ValueError, match="g:f.txt: pin is missing bytes"):
        fetch(tmp_path, opener=lambda url: downloaded.append(url) or io.BytesIO(b"tampered"))
    assert downloaded == []
    assert not (tmp_path / "d" / "f.txt").exists()


def test_profiles_record_robot_facts_without_grading_them():
    """Width, fall height and the command interface are profile data. This package does not grade them."""
    rob = robot()
    conf = configuration("stock")[0]
    assert rob["dimensions"]["width"]["value"] == 0.541
    assert conf["locomotion"]["fall_base_height"]["value"] == 0.44
    assert conf["controller"]["interface"] == ["velocity_command"]
    assert conf["controller"]["rate_hz"] == 50
    pinned = json.loads((ROOT / "pins.json").read_text())["groups"]
    assert "worlds" not in pinned
    for path in pinned["specs"]["files"]:
        assert "/scenarios/" not in path
    with pytest.raises(Unsupported, match="configuration_offered"):
        configuration("vendor-x")


def test_an_unknown_configuration_is_unsupported_and_width_is_not_graded():
    """A missing configuration is unsupported. A recorded width is not accepted or rejected here."""
    assert robot()["dimensions"]["width"]["value"] == measure()["width_m"]
    with pytest.raises(Unsupported) as caught:
        configuration("vendor-x")
    assert caught.value.precondition == "configuration_offered"
    assert not hasattr(ROBOT, "preconditions")
    assert not hasattr(ROBOT, "world")


def test_nominal_pose_is_exact_and_start_offset_is_one_declared_change():
    for name in ("stock", "playground"):
        scene = build(model="playground" if name == "playground" else "booster_gym")
        m, c = scene.model, controller(name, scene)
        d = mujoco.MjData(m)
        ROBOT.initial_state(m, d, c, "spawn-entrance")
        nominal = d.qpos.copy()
        np.testing.assert_allclose(d.qvel, 0.0)
        np.testing.assert_allclose(nominal[0:2], c.initial_qpos[0:2])
        if name == "stock":
            assert nominal[2] == 0.72
        else:
            assert nominal[2] == pytest.approx(0.665)
        again = mujoco.MjData(m)
        ROBOT.initial_state(m, again, c, "spawn-entrance")
        np.testing.assert_array_equal(again.qpos, nominal)
        shifted = mujoco.MjData(m)
        ROBOT.initial_state(
            m, shifted, c, "spawn-entrance", start_offset={"base_xy_m": [0.0, 0.05]}
        )
        np.testing.assert_allclose(shifted.qpos[0], nominal[0])
        np.testing.assert_allclose(shifted.qpos[1], nominal[1] + 0.05)
        np.testing.assert_allclose(shifted.qpos[2:], nominal[2:])
        np.testing.assert_allclose(shifted.qvel, 0.0)
