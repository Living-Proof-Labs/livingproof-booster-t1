# livingproof-booster-t1

Living Proof Labs robot package: Booster Robotics T1 humanoid. It pins an official model and a published, ready-made walking controller so the robot can be validated in the livingproof simulators (MuJoCo first) and in the [livingproof-worlds](https://github.com/Living-Proof-Labs/livingproof-worlds) test worlds.

Living Proof Labs does not train controllers. A robot is validated only with a walking controller or policy its maker or a maintained project has already published.

## Sources (pinned in `pins.json`, downloaded and sha256-verified into `.cache/`, never vendored)

| What | Upstream | License |
|---|---|---|
| Model `resources/T1/T1_locomotion.xml` + meshes | [BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym) @ `da396a0` | Apache-2.0 |
| Policy `deploy/models/T1.pt` (TorchScript actor) | same commit | Apache-2.0 |
| Training config `envs/T1.yaml`, loop `play_mujoco.py`, `export_model.py` | same commit | Apache-2.0 |
| Profile schemas | robot-capability-specs v1.5.0 | Apache-2.0 |
| World `hallway-open` 0.1.0 | livingproof-worlds v1.1.0 | Apache-2.0 |

## Layout

- `robot.json`: robot profile (dimensions measured from the pinned model: `task profile:measure`).
- `configurations/stock.json`: the published controller with the obs/action layout, PD gains, default pose and rates of `envs/T1.yaml`, exactly as `play_mujoco.py` uses them. No tuning.
- `src/lpl_t1/publisher.py`: `play_mujoco.py`'s loop without the viewer, for reference checks.

## Tasks

`task setup`, `task check`, `task test`, `task profile:measure`, `task profile:stand`.

## Rules

- No generated files are committed: no downloaded models or policies, no run output, no evidence.
- Upstream licenses are recorded in the profiles and honoured. This repository's own code is Apache-2.0.
