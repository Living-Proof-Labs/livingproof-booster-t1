# livingproof-booster-t1

Living Proof Labs robot package: Booster Robotics T1 humanoid. It pins an official model and a published, ready-made walking controller so the robot can be validated in the livingproof simulators (MuJoCo first) and in the [livingproof-worlds](https://github.com/Living-Proof-Labs/livingproof-worlds) test worlds.

Living Proof Labs does not train controllers. A robot is validated only with a walking controller or policy its maker or a maintained project has already published.

## Sources (pinned in `pins.json`, downloaded and sha256-verified into `.cache/`, never vendored)

| What | Upstream | License |
|---|---|---|
| Model `resources/T1/T1_locomotion.xml` + meshes | [BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym) @ `da396a0` | Apache-2.0 |
| Policy `deploy/models/T1.pt` (TorchScript actor) | same commit | Apache-2.0 |
| Training config `envs/T1.yaml`, loop `play_mujoco.py`, `export_model.py` | same commit | Apache-2.0 |
| Profile schemas and walk-hallway scenario | robot-capability-specs v2.0.2 (`16524cb`) | Apache-2.0 |
| World `hallway-open` 0.1.0 | livingproof-worlds v1.1.0 | Apache-2.0 |

## Layout

- `robot.json`: robot profile (dimensions measured from the pinned model: `task profile:measure`).
- `configurations/stock.json`: the published controller with the obs/action layout, PD gains, default pose and rates of `envs/T1.yaml`, exactly as `play_mujoco.py` uses them. No tuning.
- `src/lpl_t1/publisher.py`: `play_mujoco.py`'s loop without the viewer, for reference checks.

## System dependencies

Python 3.12 with [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev). Tasks run MuJoCo with `MUJOCO_GL=osmesa`, so the OSMesa software renderer is needed:

    sudo apt-get install -y libosmesa6

## Reference check

`task reference` compares runner v2.0.1 (the tag pinned in `verification.lock.json`) with `play_mujoco.py`'s loop tick by tick and requires observations and actions to be exactly equal (`==`). The reference reads sensors fresh, uses T1.pt for the unpublished .pth, and advances the gait phase with the runner's arithmetic; each deviation is documented in `src/lpl_t1/reference.py`. Tests also check that a 1-ULP change is detected and that three runs are bit-identical. Trials start at the exact nominal pose. There is no seed and no noise.

## Tasks

`uv sync`, `uv run pytest tests` (CI: capability-verifier's walk-hallway `check` job), `task profile:measure`, `task profile:stand`.

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
| trials | `trials.count` 3 | identical trials from the exact nominal pose |
| start | exact nominal pose at `spawn-entrance` | a `start_offset` would be one declared comparison change |

The specification's default count is 1. This package uses 3 so a published run can show that the identical trials agree. Evidence 2.0.0 has no seeds and no `required_passes`. The verifier's k-of-N grade for that format requires every trial to pass. Both `stock` and `playground` are in the lock, so this repository's verification runs both.

`uv run python -m livingproof_sim_mujoco.evidence lpl_t1.adapter:ROBOT CONFIGURATION OUT` records `trials.count` identical trials (0.5 m/s) for the capability verifier, for example `uv run python -m livingproof_sim_mujoco.evidence lpl_t1.adapter:ROBOT stock artifacts/evidence/stock`. CONFIGURATION is `stock` or `playground`. It writes `trials/trial-NN.json` and a `livingproof-hallway-evidence/2.0.0` manifest, plus a `livingproof-run-manifest/0.1` before the first trial. Exit 4 means the identical trials disagreed (a test-system reproducibility finding, not a robot grade). `--allow-pending` is for local runs before the pinned releases exist. The capability verifier grades the trials and the capability report only renders that stored grade; nothing is graded here. `task reference` compares the runner with `play_mujoco.py`.

Every release ref in `verification.lock.json` is currently pinned (specs v2.0.2, runner v2.0.1, verifier v1.16.1, reporter v3.3.0). Evidence goes to `artifacts/evidence/<configuration>/` (`manifest.json`, `run-manifest.json`, `trials/trial-NN.json`). Artifacts are never committed.

The runner pin is sim-mujoco v2.0.1 (`5f6b9d10962eb6108dc55b079806b7a3cf794de7`): the pyproject tag on the `livingproof-sim-mujoco` source (`tag = "v2.0.1"`), `verification.lock.json` `releases.runner.tag` and `releases.runner.commit`. `task repin -- runner=vX.Y.Z` rewrites that tag and the lock, then `uv lock`. The verifier pin is capability-verifier v1.16.1 (`67bb6bbc330645f9fee66b9ab16f6c055cd313f8`), including `releases.verifier` and the `walk-hallway.yml` ref.

## Configuration `playground` (1.0.0)

MuJoCo Playground's T1 joystick policy (`experimental/sim2sim/onnx/t1_policy.onnx`,
mujoco_playground 12ad7f3, Apache-2.0; meshes from mujoco_menagerie 1b86ece, Apache-2.0)
on Playground's own T1 MJCF and `<position>` actuators, with play_t1_joystick.py's
observation, keyframe `home` default pose, 0.02 s / 0.002 s rates and action scale 1.0.
Nothing is tuned. `task reference:playground` compares the runner with the script's
`OnnxController` loop tick by tick.

Deviations from the script, both documented in `configurations/playground.json` and
`src/lpl_t1/playground_reference.py`:

1. Flat-terrain scene instead of the rough heightfield the script loads (the hallway is flat).
2. The script's `joint_velocities = data.qvel[6:]` is a view, so `joint_velocities[:2] *= 0`
   also zeroes the head joint velocities of the simulation every policy step. The runner
   zeroes them in the observation only. With `.copy()` the runner is bit-exact over 500
   ticks; the verbatim loop is reported (it diverges from tick 1, final xy 4.44, 0.99 m vs
   4.45, 1.04 m).

robot.json keeps the stock model's width (0.541 m); the Playground model in its home pose
is narrower (0.505 m, `python -m lpl_t1.measure playground`), so the hallway check stays
conservative. Standing trunk height in the script's loop is 0.667 m, so nominal 0.67 m and
fall 0.44 m are the same as stock.
