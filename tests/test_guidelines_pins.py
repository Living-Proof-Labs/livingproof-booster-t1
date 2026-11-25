# SPDX-License-Identifier: Apache-2.0
"""collaboration-guidelines bumps move every consumer pin together and keep the release interface.

Proposed by Vera in the capability-verifier#72 review (guidelines bump checklist items 2, 3 and 6),
adapted for this repository. SHARED_RELEASE_* is the floor the pinned semantic-release.yml must
still declare. LIVINGPROOF_NETWORK=1 reads that file at the pinned tag and parses its full input
and output interface; callers are checked against the parsed interface. Offline, callers are
checked against the floor. CI does not set LIVINGPROOF_NETWORK, so an offline test drives that
same fetch branch with a stand-in workflow. Adding an input or output upstream does not require
editing SHARED_RELEASE_*. Removing a recorded key does.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = "Living-Proof-Labs/collaboration-guidelines"
TAG = r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
USES = re.compile(
    rf"^\s+uses:\s*{re.escape(REPO)}/\.github/workflows/"
    rf"(?P<file>[\w.-]+\.ya?ml)@(?P<ref>\S+)\s*$",
    re.MULTILINE,
)
PRESET = re.compile(rf"^github>{REPO}:renovate-preset#(?P<ref>.+)$")
# Floor the pinned semantic-release.yml must keep declaring. The full interface is
# parsed from that file at the pinned tag (see shared_release_interface).
SHARED_RELEASE_INPUTS = {"dry-run", "draft-release"}
SHARED_RELEASE_OUTPUTS = {
    "draft-created",
    "published",
    "version",
    "git-tag",
    "commit",
    "release-url",
}


def _root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "renovate.json").is_file():
            return candidate
    raise AssertionError("could not find the repository root (renovate.json)")


ROOT = _root()


def workflow_path() -> Path | None:
    for name in (".github/workflows/verify.yml", ".github/workflows/release.yml"):
        path = ROOT / name
        if path.is_file() and USES.search(path.read_text()):
            return path
    return None


def guideline_calls(text: str | None = None) -> dict[str, re.Match[str]]:
    path = workflow_path()
    if text is None:
        if path is None:
            return {}
        text = path.read_text()
    calls: dict[str, re.Match[str]] = {}
    for match in USES.finditer(text):
        headers = re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", text[: match.start()], re.MULTILINE)
        assert headers, "expected a job name above the guidelines workflow call"
        calls[headers[-1]] = match
    return calls


def preset_refs(config: dict | None = None) -> list[str]:
    config = config or json.loads((ROOT / "renovate.json").read_text())
    return [m.group("ref") for m in map(PRESET.match, config.get("extends", [])) if m]


def pinned_versions(text: str | None = None, config: dict | None = None) -> set[str]:
    refs = set(preset_refs(config))
    if text is not None or workflow_path() is not None:
        refs |= {match.group("ref") for match in guideline_calls(text).values()}
    return refs


def _job_block(text: str, job: str) -> str:
    match = re.search(rf"^  {re.escape(job)}:\n", text, re.MULTILINE)
    assert match, f"job {job} not found"
    start = match.start()
    end = len(text)
    for later in re.finditer(r"^  [A-Za-z0-9_-]+:\s*$", text[match.end() :], re.MULTILINE):
        end = match.end() + later.start()
        break
    return text[start:end]


def _mapping(block: str, key: str, indent: int = 4) -> dict[str, str]:
    pad = " " * indent
    inner = " " * (indent + 2)
    found = re.search(rf"^{pad}{re.escape(key)}:\n((?:{inner}.*\n)*)", block, re.MULTILINE)
    if not found:
        return {}
    return dict(re.findall(rf"^{inner}([A-Za-z0-9_-]+):\s*(\S+)", found.group(1), re.MULTILINE))


def _workflow_call_keys(shared: str, section: str) -> set[str]:
    call = re.search(r"^  workflow_call:\n((?:    .*\n)+)", shared, re.MULTILINE)
    assert call, "expected a workflow_call interface"
    body = call.group(1)
    found = re.search(rf"^    {re.escape(section)}:\n((?:      .*\n)+)", body, re.MULTILINE)
    if not found:
        return set()
    return set(re.findall(r"^      ([A-Za-z0-9_-]+):", found.group(1), re.MULTILINE))


def pinned_semantic_release_text() -> str:
    refs = {match.group("ref") for match in guideline_calls().values()}
    (ref,) = refs
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/.github/workflows/semantic-release.yml"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def shared_release_interface(shared: str | None = None) -> tuple[set[str], set[str]]:
    """Inputs and outputs callers may use.

    `shared` is the pinned workflow text. The recorded floor must still be present,
    and every other key that file declares is part of the interface. Without the
    file, the floor itself is the interface.
    """
    if shared is None:
        return set(SHARED_RELEASE_INPUTS), set(SHARED_RELEASE_OUTPUTS)
    inputs = _workflow_call_keys(shared, "inputs")
    outputs = _workflow_call_keys(shared, "outputs")
    missing_inputs = SHARED_RELEASE_INPUTS - inputs
    missing_outputs = SHARED_RELEASE_OUTPUTS - outputs
    assert not missing_inputs, sorted(missing_inputs)
    assert not missing_outputs, sorted(missing_outputs)
    return inputs, outputs


def release_interface() -> tuple[set[str], set[str]]:
    if os.environ.get("LIVINGPROOF_NETWORK"):
        return shared_release_interface(pinned_semantic_release_text())
    return shared_release_interface(None)


def assert_callers_use_interface(text: str, inputs: set[str], outputs: set[str]) -> None:
    calls = guideline_calls(text)
    assert calls, "expected a job that uses the shared semantic-release workflow"
    for name, match in calls.items():
        assert match.group("file") == "semantic-release.yml"
        block = _job_block(text, name)
        used_inputs = set(_mapping(block, "with"))
        assert used_inputs <= inputs, sorted(used_inputs - inputs)
        assert _mapping(block, "permissions").get("contents") == "write"
        used_outputs = set(re.findall(rf"needs\.{re.escape(name)}\.outputs\.([\w-]+)", text))
        assert used_outputs <= outputs, sorted(used_outputs - outputs)


def _shared_workflow(inputs: set[str], outputs: set[str]) -> str:
    def keys(names: set[str]) -> str:
        return "".join(f"      {name}:\n        description: recorded\n" for name in sorted(names))

    return (
        "name: Shared semantic release\n"
        "on:\n"
        "  workflow_call:\n"
        f"    inputs:\n{keys(inputs)}"
        f"    outputs:\n{keys(outputs)}"
    )


def test_every_guidelines_pin_is_a_full_release_tag():
    refs = pinned_versions()
    assert refs, "expected a Renovate preset pin and/or a release workflow call"
    for ref in refs:
        assert re.fullmatch(TAG, ref), f"{ref} is not a full vX.Y.Z release tag"


def test_workflow_call_and_renovate_preset_name_one_guidelines_release():
    # A partial bump (one pin moved, the other left behind) fails here.
    # Repos that only pin the preset still need exactly one release tag.
    assert len(pinned_versions()) == 1, sorted(pinned_versions())


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_mismatched_pins_are_reported():
    text = workflow_path().read_text()
    mutated = USES.sub(lambda m: m.group(0).replace(m.group("ref"), "v2.2.0", 1), text, count=1)
    preset = {"extends": [f"github>{REPO}:renovate-preset#v9.2.0"]}
    assert len(pinned_versions(mutated, preset)) == 2


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_release_call_uses_only_the_shared_interface():
    text = workflow_path().read_text()
    inputs, outputs = release_interface()
    assert_callers_use_interface(text, inputs, outputs)


@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_an_unknown_release_output_is_reported():
    text = workflow_path().read_text()
    mutated = (
        text + "\n# needs.release.outputs.tag\n"
        if "needs.release.outputs" in text
        else text.replace(
            "permissions:", "needs.release.outputs.tag: placeholder\n  permissions:", 1
        )
    )
    found = set(re.findall(r"needs\.release\.outputs\.([\w-]+)", mutated))
    assert "tag" in found
    inputs, outputs = release_interface()
    with pytest.raises(AssertionError, match="'tag'"):
        assert_callers_use_interface(mutated, inputs, outputs)


def test_shared_interface_is_parsed_from_the_workflow_text():
    extra = {"expected-version", "expected-commit"}
    shared = _shared_workflow(SHARED_RELEASE_INPUTS | extra, set(SHARED_RELEASE_OUTPUTS))
    inputs, outputs = shared_release_interface(shared)
    assert inputs == _workflow_call_keys(shared, "inputs")
    assert outputs == _workflow_call_keys(shared, "outputs")
    assert extra <= inputs - SHARED_RELEASE_INPUTS
    assert outputs == SHARED_RELEASE_OUTPUTS


def test_callers_are_checked_against_the_parsed_interface():
    """A caller may use a key the pinned workflow declares and the floor does not record."""
    extra_inputs = {"expected-version", "expected-commit"}
    extra_output = "future-output"
    shared = _shared_workflow(
        SHARED_RELEASE_INPUTS | extra_inputs,
        set(SHARED_RELEASE_OUTPUTS) | {extra_output},
    )
    inputs, outputs = shared_release_interface(shared)
    caller = (
        "jobs:\n"
        "  release:\n"
        "    permissions:\n"
        "      contents: write\n"
        "    uses: Living-Proof-Labs/collaboration-guidelines/.github/workflows/"
        "semantic-release.yml@v9.9.10\n"
        "    with:\n"
        "      expected-version: v1.2.3\n"
        "      expected-commit: abcdef\n"
        "  deploy:\n"
        "    needs: release\n"
        "    if: needs.release.outputs.future-output == 'true'\n"
    )
    assert extra_inputs - SHARED_RELEASE_INPUTS
    assert extra_output not in SHARED_RELEASE_OUTPUTS
    assert_callers_use_interface(caller, inputs, outputs)
    unknown_input = caller.replace("expected-commit: abcdef", "not-a-real-input: abcdef")
    with pytest.raises(AssertionError, match="'not-a-real-input'"):
        assert_callers_use_interface(unknown_input, inputs, outputs)
    unknown_output = caller.replace("future-output", "not-an-output")
    with pytest.raises(AssertionError, match="'not-an-output'"):
        assert_callers_use_interface(unknown_output, inputs, outputs)


def test_parsed_interface_must_keep_the_recorded_floor():
    missing_input = _shared_workflow(
        SHARED_RELEASE_INPUTS - {"dry-run"}, set(SHARED_RELEASE_OUTPUTS)
    )
    with pytest.raises(AssertionError, match="'dry-run'"):
        shared_release_interface(missing_input)
    missing_output = _shared_workflow(
        set(SHARED_RELEASE_INPUTS), set(SHARED_RELEASE_OUTPUTS) - {"version"}
    )
    with pytest.raises(AssertionError, match="'version'"):
        shared_release_interface(missing_output)


def _patched_pin(monkeypatch: pytest.MonkeyPatch, shared: str) -> None:
    """Run the fetch branch of release_interface without contacting GitHub.

    CI never sets LIVINGPROOF_NETWORK. Setting it here, and replacing the fetch,
    is what makes a revert of that branch fail the offline suite.
    """
    monkeypatch.setenv("LIVINGPROOF_NETWORK", "1")
    monkeypatch.setattr(sys.modules[__name__], "pinned_semantic_release_text", lambda: shared)


def test_fetch_branch_returns_keys_parsed_from_the_patched_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra_inputs = {"expected-version", "expected-commit"}
    extra_outputs = {"future-output"}
    shared = _shared_workflow(
        SHARED_RELEASE_INPUTS | extra_inputs,
        set(SHARED_RELEASE_OUTPUTS) | extra_outputs,
    )
    _patched_pin(monkeypatch, shared)
    inputs, outputs = release_interface()
    assert inputs == _workflow_call_keys(shared, "inputs")
    assert outputs == _workflow_call_keys(shared, "outputs")
    assert extra_inputs <= inputs - SHARED_RELEASE_INPUTS
    assert extra_outputs <= outputs - SHARED_RELEASE_OUTPUTS


def test_fetch_branch_raises_when_the_patched_workflow_drops_a_floor_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_input = _shared_workflow(
        SHARED_RELEASE_INPUTS - {"dry-run"}, set(SHARED_RELEASE_OUTPUTS)
    )
    _patched_pin(monkeypatch, missing_input)
    with pytest.raises(AssertionError, match="'dry-run'"):
        release_interface()
    missing_output = _shared_workflow(
        set(SHARED_RELEASE_INPUTS), set(SHARED_RELEASE_OUTPUTS) - {"commit"}
    )
    _patched_pin(monkeypatch, missing_output)
    with pytest.raises(AssertionError, match="'commit'"):
        release_interface()


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
@pytest.mark.skipif(
    workflow_path() is None, reason="this repository has no guidelines workflow call"
)
def test_recorded_interface_matches_the_pinned_shared_workflow():
    shared = pinned_semantic_release_text()
    parsed_inputs = _workflow_call_keys(shared, "inputs")
    parsed_outputs = _workflow_call_keys(shared, "outputs")
    inputs, outputs = release_interface()
    assert inputs == parsed_inputs
    assert outputs == parsed_outputs
    assert SHARED_RELEASE_INPUTS <= inputs
    assert SHARED_RELEASE_OUTPUTS <= outputs
    # Writer inputs live on the pinned file. Recording only the floor keeps the next
    # added key from failing this test; exact equality with the floor would fail it.
    assert {"expected-version", "expected-commit"} <= inputs - SHARED_RELEASE_INPUTS
    assert_callers_use_interface(workflow_path().read_text(), inputs, outputs)


def test_this_repository_does_not_opt_in_to_docs_brand_updates():
    """v9.4.0's docs-brand manager stays off until a consumer opts in.

    Proposed by Vera in the guidelines v9.4.0 bump review. This repository
    enables custom.regex and has a docs-brand.lock.json, so the new manager
    would match, but the pinned preset disables that dep type by default.
    A later consumer packageRule that sets enabled: true fails here.
    """
    config = json.loads((ROOT / "renovate.json").read_text())
    assert "custom.regex" in config.get("enabledManagers", [])
    assert (ROOT / "docs-brand.lock.json").is_file()
    for rule in config.get("packageRules") or []:
        dumped = json.dumps(rule)
        if "docs-brand" in dumped or rule.get("matchDepTypes") == ["docs-brand"]:
            assert rule.get("enabled") is False, rule


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
def test_pinned_preset_keeps_docs_brand_disabled_by_default():
    # Proposed by Vera in the guidelines v9.4.0 bump review.
    refs = pinned_versions()
    assert refs, "expected a Renovate preset pin"
    (ref,) = refs
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/renovate-preset.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        preset = json.loads(response.read().decode())
    brand_rules = [
        rule
        for rule in preset.get("packageRules", [])
        if "docs-brand" in json.dumps(rule) or rule.get("matchDepTypes") == ["docs-brand"]
    ]
    assert brand_rules, f"pinned {ref} preset has no docs-brand packageRule"
    for rule in brand_rules:
        assert rule.get("enabled") is False, rule
        assert rule.get("automerge") is False
    managers = [
        manager
        for manager in preset.get("customManagers", [])
        if manager.get("depTypeTemplate") == "docs-brand"
        or "docs-brand.lock.json" in json.dumps(manager)
    ]
    assert managers, f"pinned {ref} preset has no docs-brand.lock.json manager"
