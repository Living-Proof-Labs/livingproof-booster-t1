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

## System dependencies

Python 3.12 with [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev). Tasks run MuJoCo with `MUJOCO_GL=osmesa`, so the OSMesa software renderer is needed:

    sudo apt-get install -y libosmesa6

## Reference check

`task reference` compares runner v1.4.0 with `play_mujoco.py`'s loop tick by tick and requires observations and actions to be exactly equal (`==`). The reference reads sensors fresh, uses T1.pt for the unpublished .pth, and advances the gait phase with the runner's arithmetic; each deviation is documented in `lpl_t1/reference.py`. Tests also check that a 1-ULP change is detected and that three runs are bit-identical.

## Tasks

`task setup`, `task check`, `task test`, `task profile:measure`, `task profile:stand`.

## Rules

- No generated files are committed: no downloaded models or policies, no run output, no evidence.
- Upstream licenses are recorded in the profiles and honoured. This repository's own code is Apache-2.0.

## walk-hallway@0.1.0 (hallway-open 0.1.0, clear width 1.5 m)

| Parameter | T1 | How |
|---|---|---|
| `lateral_limit` | 0.4795 m | 1.5 / 2 - 0.541 / 2, the G1 rule (G1: 0.49 m) |
| `wall_clearance` | 0.05 m | humanoid |
| `fits_width` | met | 0.541 + 2 x 0.05 = 0.641 m <= 1.5 m, so the T1 is supported |
| `fall_height` | 0.44 m | about 2/3 of the 0.667 m standing trunk height |

`task evidence` records 20 seeded trials (seeds 1-20, 0.5 m/s) for the capability verifier; grading is done there and by the capability report, never here. `task reference` compares the runner with `play_mujoco.py`.
