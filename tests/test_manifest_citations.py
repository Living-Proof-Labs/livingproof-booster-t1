# SPDX-License-Identifier: Apache-2.0
"""Manifest citations must match the committed tree and pins.json."""

import json

from livingproof_sim_mujoco.pins import sha256

from lpl_t1 import ROOT


def test_manifest_citations_match_committed_bytes_and_pins():
    """Profile and configuration sha256s match the files; assets equal pins.json."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    profile = ROOT / manifest["profile"]["path"]
    assert manifest["profile"]["sha256"] == sha256(profile)
    for item in manifest["configurations"]:
        path = ROOT / item["path"]
        document = json.loads(path.read_text())
        assert item["version"] == document["version"]
        assert item["sha256"] == sha256(path)
    pins = json.loads((ROOT / "pins.json").read_text())["groups"]
    expected = {
        (
            name,
            group["repository"],
            group["commit"],
            path,
            entry["sha256"],
            entry["bytes"],
        )
        for name, group in pins.items()
        for path, entry in group["files"].items()
    }
    cited = {
        (
            asset["group"],
            asset["repository"],
            asset["commit"],
            asset["path"],
            asset["sha256"],
            asset["bytes"],
        )
        for asset in manifest["assets"]
    }
    assert cited == expected
    assert len(manifest["assets"]) == len(expected)
