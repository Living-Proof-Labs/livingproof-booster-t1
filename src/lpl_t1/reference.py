# SPDX-License-Identifier: Apache-2.0
"""Our runner against booster_gym's own control loop (play_mujoco.py), tick by tick.

The reference is ``lpl_t1.publisher.play`` with ``fresh_sensors=True`` and
``runner_phase=True``: play_mujoco.py's loop with three documented deviations, and
otherwise verbatim (constant command, no viewer):

1. Fresh sensor reads. play_mujoco.py reads ``orientation`` and ``angular-velocity`` right
   after ``mj_step``, when they still hold values computed before that step's integration
   (one 0.002 s physics step stale). The reference calls ``mj_forward`` before reading, so
   gyro and gravity describe the current state, as the runner (and training) see them.
2. T1.pt in place of the .pth checkpoint. play_mujoco.py evaluates
   ``ActorCritic.act(obs).loc`` from a .pth that booster_gym does not publish; its
   export_model.py saves ``torch.jit.script(model.actor)`` as T1.pt and ``act(obs).loc``
   is ``actor(obs)``, so T1.pt is the same function.
3. Phase arithmetic: play_mujoco.py accumulates the gait phase as a fraction every physics
   step (``fmod(gp + dt f, 1)``, read as cos/sin of ``2 pi gp``); the runner advances it in
   radians once per policy step (``fmod(phase + 2 pi f 10 dt + pi, 2 pi) - pi``). The same
   phase, rounded differently (~2e-14 over 100 ticks). The graded reference uses the runner's
   arithmetic, so no tolerance is needed; the publisher's is reported.

Both sides run the same pinned MJCF from the same initial state with the same command.

Pass rule: the same first policy step, and against the float64 reference observations and
actions exactly equal (``==``). The verbatim loop
(stale reads, float32 and float64) is reported, not graded; ``no_clip`` shows the +/-1
action clip (absent from runner v1.4.0) is never reached.

The stand (zero command) is not compared: play_mujoco.py gates the phase off for a zero
command and the runner's phase term cannot be gated, so only walking commands are run.
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
from lpl_t1.publisher import play
from lpl_t1.scene import Scene, raw_model

TERMS = [
    ("gravity", 3),
    ("ang_vel", 3),
    ("command", 3),
    ("phase", 2),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("last_action", 12),
]


def ours(command, steps, qpos0, root=ROOT):
    """Runner v1.4.0 OnnxPolicy; each tick is logged with the physics step it acts in."""
    from livingproof_sim_mujoco.policy import OnnxPolicy

    model = raw_model(root)
    c = controller("stock", Scene(model, {}, {}, []), root)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)
    policy = OnnxPolicy(model, c.policy_path, c.spec)
    log, backend = [], policy.backend

    def recording(obs):
        out = backend(obs)
        log.append((policy.counter, np.asarray(obs).copy(), np.asarray(out).copy()))
        return out

    policy.backend = recording
    for _ in range(steps):
        policy.step(data, command)
    return log, data.qpos.copy()


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
        "max_abs_action_reference": float(max(np.abs(a).max() for _, _, a in ref)),
        "final_base_xy": {
            "reference": [round(float(v), 4) for v in ref_q[:2]],
            "runner": [round(float(v), 4) for v in got_q[:2]],
        },
    }


def passes(d: dict) -> bool:
    return (
        d["first_policy_step"]["reference"] == d["first_policy_step"]["runner"]
        and d["max_abs_diff"]["action"] == 0.0
        and d["max_abs_diff"]["obs"] == 0.0
    )


VARIANTS = {
    "float64": {"dtype": np.float64, "fresh_sensors": True, "runner_phase": True},  # graded
    "float64_publisher_phase": {"dtype": np.float64, "fresh_sensors": True},
    "float32": {"fresh_sensors": True, "runner_phase": True},
    "float64_no_clip": {
        "dtype": np.float64,
        "fresh_sensors": True,
        "runner_phase": True,
        "clip": False,
    },
    "verbatim_stale_float32": {},
    "verbatim_stale_float64": {"dtype": np.float64},
}


def run(command, ticks: int, root=ROOT) -> dict:
    steps = 10 * ticks
    keep = ("first_policy_step", "bit_exact", "first_divergence", "max_abs_diff")
    keep += ("max_obs_diff_by_term", "max_abs_action_reference", "final_base_xy")
    out = {}
    for name, kw in VARIANTS.items():
        ref, _, rq = play(command, steps, root, **kw)
        q0 = _initial_qpos(kw.get("dtype", np.float32), root)
        got, gq = ours(command, steps, q0, root)
        d = compare(ref, got, rq, gq)
        out[name] = {k: d[k] for k in keep} | {"ticks": d["ticks"], "passes_rule": passes(d)}
    return {
        "reference": (
            "play_mujoco.py loop + envs/T1.yaml @ booster_gym da396a0, fresh sensor reads, "
            "T1.pt as the actor"
        ),
        "configuration": "stock",
        "configuration_sha256": configuration("stock", root)[1],
        "command": list(command),
        "pass_rule": (
            "same first policy step, and against the float64 reference (fresh reads, runner "
            "phase arithmetic) observations and actions exactly equal; the publisher-phase and "
            "verbatim (stale-read) loops are reported, not graded"
        ),
        "passed": out["float64"]["passes_rule"],
        "variants": out,
    }


def _initial_qpos(dtype, root=ROOT):
    """The publisher loop's own initial qpos, cast as it casts it."""
    _, _, q = play((0.0, 0.0, 0.0), 0, root, dtype=dtype)
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=50, help="policy steps (x10 physics steps)")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "reference-compare.json")
    a = ap.parse_args(argv)
    res = run((0.5, 0.0, 0.0), a.ticks)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"walk_0.5: {'PASS' if res['passed'] else 'FAIL'} ({res['pass_rule']})")
    for k, v in res["variants"].items():
        f = v["first_divergence"]
        print(
            f"  {k}: {'bit-exact' if v['bit_exact'] else 'differs'} over {v['ticks']} ticks; "
            f"first step ref {v['first_policy_step']['reference']} runner "
            f"{v['first_policy_step']['runner']}; max diff obs {v['max_abs_diff']['obs']:.3g} "
            f"action {v['max_abs_diff']['action']:.3g}; max |action| ref "
            f"{v['max_abs_action_reference']:.3g}"
            + (f"; first divergence tick {f['tick']} {f['max_obs_diff_by_term']}" if f else "")
        )
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
