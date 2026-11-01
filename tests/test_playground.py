# SPDX-License-Identifier: Apache-2.0
"""configurations/playground.json: pinned files, published values, exact reference."""

import json
import math
import re

import mujoco
import numpy as np
import pytest

from lpl_t1 import ROOT
from livingproof_sim_mujoco import pins
from livingproof_sim_mujoco.pins import sha256
from lpl_t1.playground_reference import run, stand
from lpl_t1.profiles import configuration, controller
from lpl_t1.scene import build, raw_model


def local(group: str, path: str):
    return pins.local(group, path, ROOT)


PG = "mujoco_playground/experimental/sim2sim/play_t1_joystick.py"


def test_every_playground_file_is_pinned_and_apache_2():
    conf, digest = configuration("playground")
    assert conf["version"] == "1.0.0"
    lic = local("playground", "LICENSE").read_text()
    assert "Apache License" in lic and "Version 2.0" in lic
    assert "Apache License" in local("menagerie", "booster_t1/LICENSE").read_text()
    s = conf["settings"]
    for a in (conf["controller"]["artifact"], s["reference"], s["scene"], s["robot_mjcf"]):
        assert a["license"] == "Apache-2.0" and a["commit"].startswith("12ad7f3")
        assert sha256(local("playground", a["path"])) == a["sha256"]
    lock = json.loads((ROOT / "verification.lock.json").read_text())
    entry = {c["name"]: c for c in lock["configurations"]}["playground"]
    assert entry["sha256"] == digest and entry["version"] == "1.0.0"


def test_settings_equal_play_t1_joystick():
    src = local("playground", PG).read_text()
    s = configuration("playground")[0]["settings"]
    assert "ctrl_dt = 0.02" in src and s["ctrl_dt_s"] == 0.02
    assert "sim_dt = 0.002" in src and s["sim_dt_s"] == 0.002
    assert "action_scale=1.0" in src and s["action_scale"] == 1.0
    assert "self._gait_freq = 1.5" in src and s["gait_hz"] == 1.5
    assert "np.array([0.0, np.pi])" in src and s["gait_offsets_rad"] == [0.0, math.pi]
    assert 'keyframe("home").qpos[7:]' in src and s["initial_keyframe"] == "home"
    for name in ("local_linvel", "gyro"):
        assert f'data.sensor("{name}")' in src
    assert 'model.site("imu")' in src
    assert "joint_angles[:2] *= 0.0" in src and "joint_velocities[:2] *= 0.0" in src
    assert re.search(r"FEET_ONLY_ROUGH_TERRAIN_XML", src)  # documented: flat scene used


def test_spec_is_the_publishers_layout_on_the_unchanged_mjcf():
    scene = build(model="playground")
    m = scene.model
    c = controller("playground", scene)
    home = m.key("home")
    assert c.spec.actuation == "position" and c.spec.decimation == 10 and c.spec.first_tick == 9
    assert c.spec.default_pose == list(home.qpos[7:])
    assert np.allclose(home.qpos[7:], home.ctrl)
    assert c.spec.joints[:2] == ["AAHead_yaw", "Head_pitch"]
    raw = raw_model(model="playground")
    assert raw.nu == m.nu == 23
    assert np.array_equal(raw.actuator_gainprm, m.actuator_gainprm)  # kp untouched
    assert np.array_equal(raw.dof_damping, m.dof_damping)
    assert np.array_equal(raw.actuator_forcerange, m.actuator_forcerange)
    kp = {m.actuator(i).name: float(m.actuator_gainprm[i, 0]) for i in range(m.nu)}
    assert kp["AAHead_yaw"] == 20 and kp["Waist"] == 50 and kp["Left_Knee_Pitch"] == 30


@pytest.fixture(scope="module")
def walk():
    return run((0.5, 0.0, 0.0), 500)


def test_runner_matches_play_t1_joystick_bit_exact(walk):
    assert walk["passed"] and walk["bit_exact"]
    assert walk["first_policy_step"] == {"reference": 9, "runner": 9}
    assert walk["max_abs_diff"] == {"obs": 0.0, "action": 0.0}


def test_the_head_qvel_write_is_the_only_deviation_from_the_verbatim_loop(walk):
    v = walk["verbatim_reference"]
    assert not v["bit_exact"] and v["first_divergence"]["tick"] >= 1
    assert v["first_policy_step"] == {"reference": 9, "runner": 9}


def test_stand_height_backs_the_locomotion_heights():
    conf = configuration("playground")[0]
    s = stand(2.0)
    assert s["standing"] and abs(s["trunk_z_min"] - 0.665) < 0.01
    assert conf["locomotion"]["nominal_base_height"]["value"] == 0.67


def test_initial_state_holds_the_default_pose_in_ctrl():
    from lpl_t1.walk import initial_state, runner_settings

    scene = build(model="playground")
    m, d = scene.model, mujoco.MjData(scene.model)
    c = controller("playground", scene)
    zero = {k: 0.0 for k in runner_settings()["initial_noise"]}
    initial_state(m, d, c, 1, zero, "spawn-entrance")
    assert np.allclose(d.ctrl, m.key("home").ctrl)
    assert math.isclose(d.qpos[2], 0.665)
