# SPDX-License-Identifier: Apache-2.0
"""Offline replay and publication regressions; no GitHub writes or robot execution."""

import copy
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import release
import yaml

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
RELEASE = {
    "tag_name": "v1.2.3",
    "draft": False,
    "prerelease": False,
    "immutable": True,
    "published_at": "2026-11-22T00:00:00Z",
}


class ReleaseTests(unittest.TestCase):
    def validate(self, record=None, tag="v1.2.3", version="1.2.3", tagged=SHA, expected=SHA):
        return release.validate_release(
            RELEASE if record is None else record, tag, version, SHA, tagged, expected
        )

    def test_published_identity_is_stable_across_delivery_replays(self):
        self.assertEqual(self.validate(), self.validate())
        self.assertEqual(self.validate()["version"], "1.2.3")

    def test_missing_failed_draft_mutable_and_prerelease_fail_closed(self):
        for field, value in [
            ("draft", True),
            ("prerelease", True),
            ("immutable", False),
            ("immutable", None),
            ("published_at", None),
            ("tag_name", "v9.9.9"),
        ]:
            with self.subTest(field=field, value=value):
                record = copy.deepcopy(RELEASE)
                record[field] = value
                with self.assertRaises(ValueError):
                    self.validate(record)
        with self.assertRaises(ValueError):
            self.validate({})

    def test_tag_version_commit_and_current_main_must_agree(self):
        for change in [
            dict(tag="main"),
            dict(version="1.2.4"),
            dict(tagged="b" * 40),
            dict(expected="b" * 40),
            dict(tag="v01.2.3"),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(**change)

    def test_api_errors_never_become_a_release(self):
        with patch.dict(
            os.environ,
            RELEASE_TAG="v1.2.3",
            RELEASE_VERSION="1.2.3",
            RELEASE_COMMIT=SHA,
            GITHUB_SHA=SHA,
            GITHUB_REPOSITORY="owner/repo",
        ):
            with patch.object(
                subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "gh")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    release.published()

    def test_manual_has_no_writer_and_only_explicit_releases_can_deploy(self):
        docs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())
        triggers = docs.get("on", docs.get(True))
        self.assertNotIn("push", triggers)
        self.assertEqual(docs["permissions"], {"contents": "read"})
        self.assertIn("inputs.release_tag != ''", docs["jobs"]["deploy"]["if"])
        text = str(docs)
        self.assertNotIn("contents: write", text)
        self.assertNotIn("semantic-release", text)
        self.assertEqual(docs["jobs"]["deploy"]["concurrency"]["cancel-in-progress"], False)

    def test_existing_writer_recovery_is_preserved(self):
        import sys

        sys.path.insert(0, str(ROOT / "tests"))
        from test_release_gate import _context, _eval_expr

        jobs = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
        for attempt, triggering_attempt in [("1", "1"), ("2", "1"), ("1", "2"), ("2", "2")]:
            context = _context()
            context["github"]["run_attempt"] = attempt
            context["github"]["event"]["workflow_run"]["run_attempt"] = triggering_attempt
            self.assertTrue(_eval_expr(jobs["release"]["if"], context))
        self.assertNotIn("needs", jobs["release"])
        self.assertEqual(jobs["deploy-manual"]["if"], "needs.release.outputs.published == 'true'")

    def test_deploy_retry_selects_the_retained_build_not_the_current_attempt(self):
        jobs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())["jobs"]
        name = "${{ format('github-pages-{0}-{1}', github.run_id, github.run_attempt) }}"
        self.assertEqual(jobs["build"]["outputs"]["pages-name"], name)
        upload = next(
            s for s in jobs["build"]["steps"] if "upload-pages-artifact@" in s.get("uses", "")
        )
        deploy = jobs["deploy"]["steps"][-1]
        self.assertEqual(upload["with"]["name"], name)
        self.assertEqual(upload["with"]["retention-days"], 14)
        self.assertEqual(deploy["with"]["artifact_name"], "${{ needs.build.outputs.pages-name }}")
        self.assertEqual(jobs["deploy"]["needs"], "build")
        # A failed-job-only rerun retains the successful build's output; a new
        # complete build gets a different name and cannot collide or overwrite it.
        builds = {1: "github-pages-123-1", 2: "github-pages-123-2"}
        retained = builds[1]
        self.assertNotEqual(retained, builds[2])
        self.assertNotIn("continue-on-error", deploy)

    def test_native_runtime_readback_has_bounded_id_and_name_retries(self):
        jobs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())["jobs"]
        deploy = jobs["deploy"]
        self.assertEqual(
            deploy["permissions"], {"contents": "read", "pages": "write", "id-token": "write"}
        )
        steps = deploy["steps"]
        self.assertEqual(steps[-2]["name"], "Require current main inside deployment lock")
        self.assertEqual(
            steps[-3]["name"], "Require matching ID and name readback of the exact tar"
        )
        self.assertFalse(any("checkout@" in s.get("uses", "") for s in steps))
        self.assertNotIn("/actions/artifacts/", str(steps))
        for route, selector, expected in [
            ("id", "artifact-ids", "${{ needs.build.outputs.pages-id }}"),
            ("name", "name", "${{ needs.build.outputs.pages-name }}"),
        ]:
            first = next(s for s in steps if s.get("id") == f"pages-{route}-first")
            retry = next(s for s in steps if s.get("id") == f"pages-{route}-retry")
            self.assertTrue(first["continue-on-error"])
            self.assertNotIn("continue-on-error", retry)
            self.assertEqual(retry["if"], f"steps.pages-{route}-first.outcome == 'failure'")
            for step in [first, retry]:
                self.assertEqual(
                    step["uses"],
                    "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
                )
                self.assertEqual(step["timeout-minutes"], 2)
                self.assertEqual(set(step["with"]), {selector, "path"})
                self.assertEqual(step["with"][selector], expected)
            self.assertNotEqual(first["with"]["path"], retry["with"]["path"])
            self.assertIn(
                f"steps.pages-{route}-retry.outputs.download-path",
                steps[-3]["env"][f"{route.upper()}_READBACK"],
            )

    def test_native_readback_rejects_wrong_or_missing_bytes_on_either_route(self):
        import hashlib
        import tempfile

        jobs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())["jobs"]
        step = jobs["deploy"]["steps"][-3]
        for mode in [
            "valid",
            "missing-id",
            "missing-name",
            "wrong-id",
            "wrong-name",
            "name-mismatch",
            "bad-digest",
        ]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                identifier, named = root / "id", root / "name"
                name = "github-pages-123-1"
                located = identifier / ("github-pages-123-2" if mode == "name-mismatch" else name)
                located.mkdir(parents=True)
                named.mkdir()
                if mode != "missing-id":
                    (located / "artifact.tar").write_bytes(
                        b"wrong" if mode == "wrong-id" else b"sealed"
                    )
                if mode != "missing-name":
                    (named / "artifact.tar").write_bytes(
                        b"wrong" if mode == "wrong-name" else b"sealed"
                    )
                env = {
                    **os.environ,
                    "ID_READBACK": str(identifier),
                    "NAME_READBACK": str(named),
                    "EXPECTED_NAME": name,
                    "EXPECTED_TAR": "a" * 64
                    if mode == "bad-digest"
                    else hashlib.sha256(b"sealed").hexdigest(),
                }
                result = subprocess.run(
                    ["bash", "-c", step["run"]], env=env, capture_output=True, text=True
                )
                self.assertEqual(result.returncode == 0, mode == "valid", result.stderr)

    def test_natural_delivery_requires_all_writer_outputs_before_remote_reads(self):
        valid = {
            "RELEASE_TAG": "v1.2.3",
            "RELEASE_VERSION": "1.2.3",
            "RELEASE_COMMIT": SHA,
            "GITHUB_SHA": SHA,
            "GITHUB_EVENT_NAME": "workflow_run",
            "GITHUB_REPOSITORY": "owner/repo",
        }
        for field, value in [
            ("RELEASE_TAG", ""),
            ("RELEASE_VERSION", ""),
            ("RELEASE_COMMIT", ""),
            ("RELEASE_COMMIT", "b" * 40),
            ("RELEASE_VERSION", "1.2.4"),
        ]:
            with (
                self.subTest(field=field, value=value),
                patch.dict(os.environ, {**valid, field: value}, clear=True),
            ):
                with patch.object(subprocess, "check_output") as read:
                    with self.assertRaises(ValueError):
                        release.published()
                    read.assert_not_called()

    def test_only_explicit_named_replay_may_derive_version(self):
        import json

        for event, version, commit, succeeds in [
            ("workflow_run", "1.2.3", SHA, True),
            ("workflow_run", "", SHA, False),
            ("workflow_dispatch", "", "", True),
            ("pull_request", "", "", False),
        ]:
            env = {
                "GITHUB_EVENT_NAME": event,
                "RELEASE_TAG": "v1.2.3",
                "RELEASE_VERSION": version,
                "RELEASE_COMMIT": commit,
                "GITHUB_SHA": SHA,
                "GITHUB_REPOSITORY": "owner/repo",
            }
            with (
                self.subTest(event=event, version=version),
                patch.dict(os.environ, env, clear=True),
            ):
                with (
                    patch.object(subprocess, "check_output", return_value=json.dumps(RELEASE)),
                    patch.object(release, "git", return_value=SHA),
                ):
                    if succeeds:
                        self.assertEqual(
                            release.published(),
                            {"tag": "v1.2.3", "version": "1.2.3", "commit": SHA},
                        )
                    else:
                        with self.assertRaises(ValueError):
                            release.published()

    def test_workflow_passes_writer_commit_and_validates_even_when_tag_is_missing(self):
        docs = yaml.safe_load((ROOT / ".github/workflows/docs.yml").read_text())
        natural = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"][
            "deploy-manual"
        ]
        self.assertEqual(
            natural["with"],
            {
                "release_tag": "${{ needs.release.outputs.git-tag }}",
                "release_version": "${{ needs.release.outputs.version }}",
                "release_commit": "${{ needs.release.outputs.commit }}",
            },
        )
        guard = next(
            s
            for s in docs["jobs"]["build"]["steps"]
            if s.get("name") == "Validate the exact published release"
        )
        self.assertEqual(
            guard["if"], "github.event_name == 'workflow_run' || inputs.release_tag != ''"
        )
        self.assertEqual(
            docs["jobs"]["build"]["env"]["RELEASE_COMMIT"], "${{ inputs.release_commit }}"
        )
