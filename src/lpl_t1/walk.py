# SPDX-License-Identifier: Apache-2.0
"""Shared simulation setup: runner settings and the nominal initial state. Walk-hallway
trials are recorded by ``lpl_t1.evidence`` and graded by the capability verifier, never
here. Trials start at the exact nominal pose. There is no seed and no noise."""

from __future__ import annotations

from pathlib import Path

from livingproof_sim_mujoco import walk as generic
from livingproof_sim_mujoco.walk import initial_state  # noqa: F401

from lpl_t1 import ROOT


def runner_settings(root: Path = ROOT) -> dict:
    return generic.runner_settings(root)
