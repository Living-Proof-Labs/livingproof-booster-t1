# SPDX-License-Identifier: Apache-2.0
"""Fail-closed release identity checks; this module never writes to GitHub."""

import json
import os
import re
import subprocess
import sys

STABLE = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_release(release, tag, version, commit, tagged_commit, expected):
    if not STABLE.fullmatch(tag) or version != tag[1:]:
        raise ValueError("Expected an exact stable release tag/version")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or commit != tagged_commit or commit != expected:
        raise ValueError("Release tag, checkout and verified commit must match")
    if (
        release.get("tag_name") != tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
        or not release.get("published_at")
    ):
        raise ValueError("An immutable, published stable GitHub Release is required")
    return {"tag": tag, "version": version, "commit": commit}


def published():
    tag = os.environ.get("RELEASE_TAG", "")
    if not STABLE.fullmatch(tag):
        raise ValueError("Expected vX.Y.Z")
    version = os.environ.get("RELEASE_VERSION", "")
    expected = os.environ["GITHUB_SHA"]
    if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":
        # Only an explicit named, read-only replay may derive this identity.
        version = version or tag[1:]
    else:
        writer_commit = os.environ.get("RELEASE_COMMIT", "")
        if not version or not re.fullmatch(r"[0-9a-f]{40}", writer_commit):
            raise ValueError("Natural delivery requires writer version, git-tag and commit")
        if writer_commit != expected:
            raise ValueError("Writer commit differs from the verified source")
    if version != tag[1:]:
        raise ValueError("Writer version differs from the release tag")
    release = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{os.environ['GITHUB_REPOSITORY']}/releases/tags/{tag}"],
            text=True,
        )
    )
    return validate_release(
        release,
        tag,
        version,
        git("rev-parse", "HEAD"),
        git("rev-parse", f"refs/tags/{tag}^{{commit}}"),
        expected,
    )


if __name__ == "__main__":
    {"published": published}[sys.argv[1]]()
