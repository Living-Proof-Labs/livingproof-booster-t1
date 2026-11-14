# livingproof-booster-t1

**Start here: documentation** — [manual source and build instructions](docs/index.md).
The rendered manual is pending deployment verification.

Living Proof Labs robot package: Booster Robotics T1 humanoid. It pins an official model and a published, ready-made walking controller, and it exposes them through a simulator adapter. Scenario runs are published by [livingproof-verification](https://github.com/Living-Proof-Labs/livingproof-verification). This repository does not select a world or a scenario.

Living Proof Labs does not train controllers. A robot is validated only with a walking controller or policy its maker or a maintained project has already published.

## Sources (pinned in `pins.json`, downloaded and sha256-verified into `.cache/`, never vendored)

| What | Upstream | License |
|---|---|---|
| Model `resources/T1/T1_locomotion.xml` + meshes | [BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym) @ `da396a0` | Apache-2.0 |
| Policy `deploy/models/T1.pt` (TorchScript actor) | same commit | Apache-2.0 |
| Training config `envs/T1.yaml`, loop `play_mujoco.py`, `export_model.py` | same commit | Apache-2.0 |
| Robot and configuration profile schemas | robot-capability-specs v2.0.2 (`16524cb`) | Apache-2.0 |

## Layout

- `robot.json`: robot profile (dimensions measured from the pinned model: `task profile:measure`).
- `configurations/stock.json`: the published controller with the obs/action layout, PD gains, default pose and rates of `envs/T1.yaml`, exactly as `play_mujoco.py` uses them. No tuning.
- `configurations/playground.json`: MuJoCo Playground's published T1 joystick policy on Playground's own T1 MJCF.
- `src/lpl_t1/adapter.py`: the simulator adapter (`lpl_t1.adapter:ROBOT`). It loads a configuration's model and controller. It does not load a world.
- `src/lpl_t1/publisher.py`: `play_mujoco.py`'s loop without the viewer, for reference checks.

## System dependencies

Python 3.12 with [uv](https://docs.astral.sh/uv/) and [Task](https://taskfile.dev). Tasks run MuJoCo with `MUJOCO_GL=osmesa`, so the OSMesa software renderer is needed:

    sudo apt-get install -y libosmesa6

## Reference check

`task reference` compares the simulator library pinned in `pyproject.toml` (livingproof-sim-mujoco v2.0.3) with `play_mujoco.py`'s loop tick by tick and requires observations and actions to be exactly equal (`==`). The reference reads sensors fresh, uses T1.pt for the unpublished .pth, and advances the gait phase with the runner's arithmetic; each deviation is documented in `src/lpl_t1/reference.py`. Tests also check that a 1-ULP change is detected and that three runs are bit-identical. Trials start at the exact nominal pose. There is no seed and no noise.

## Tasks

`uv sync`, `uv run pytest tests`, `task profile:measure`, `task profile:stand`, `task reference`, `task reference:playground`.

## Rules

- No generated files are committed: no downloaded models or policies, no run output, no evidence.
- Upstream licenses are recorded in the profiles and honoured. This repository's own code is Apache-2.0.
- The simulator dependency is the released tag in `pyproject.toml`. Integration pins for a scenario run live in livingproof-verification, not in this repository.

The published adapter is `lpl_t1.adapter:ROBOT`. A caller supplies the configuration name (`stock` or `playground`). This package records the controller and the robot profile; it does not grade a run.

## Configuration `playground` (1.0.0)

MuJoCo Playground's T1 joystick policy (`experimental/sim2sim/onnx/t1_policy.onnx`,
mujoco_playground 12ad7f3, Apache-2.0; meshes from mujoco_menagerie 1b86ece, Apache-2.0)
on Playground's own T1 MJCF and `<position>` actuators, with play_t1_joystick.py's
observation, keyframe `home` default pose, 0.02 s / 0.002 s rates and action scale 1.0.
Nothing is tuned. `task reference:playground` compares the runner with the script's
`OnnxController` loop tick by tick.

Deviations from the script, both documented in `configurations/playground.json` and
`src/lpl_t1/playground_reference.py`:

1. Flat-terrain scene instead of the rough heightfield the script loads.
2. The script's `joint_velocities = data.qvel[6:]` is a view, so `joint_velocities[:2] *= 0`
   also zeroes the head joint velocities of the simulation every policy step. The runner
   zeroes them in the observation only. With `.copy()` the runner is bit-exact over 500
   ticks; the verbatim loop is reported (it diverges from tick 1, final xy 4.44, 0.99 m vs
   4.45, 1.04 m).

`robot.json` keeps the stock model's width (0.541 m). The Playground model in its home pose
is narrower (0.505 m, `python -m lpl_t1.measure playground`). Standing trunk height in the
script's loop is 0.667 m, so nominal 0.67 m and fall 0.44 m are the same as stock.
