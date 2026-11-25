# SPDX-License-Identifier: Apache-2.0
"""Release gate, race model and workflow contract for the workflow_run release.

Proposed by Vera in the booster-t1#63 review (issue #59). Under `workflow_run`,
GITHUB_SHA is the *default-branch tip* when the Test run completes, not the commit
Test ran on (GitHub docs, "Events that trigger workflows", workflow_run: GITHUB_SHA =
"Last commit on default branch"). The release `if:` compares the two, so a Test run for
a commit that is no longer the tip is skipped, and the Test run for the newer tip releases
everything since the last tag in one version.

Offline:
- every workflow parses and the Test workflow keeps the push-on-main run that the gate needs;
- the privileged `workflow_run` workflow stays a single call to the shared release workflow
  (no steps, no secrets, no cancel-in-progress, minimal permissions);
- the real `if:` is evaluated over a timeline of pushes and Test completions.
LIVINGPROOF_NETWORK=1 also checks the pinned shared workflow still checks out GITHUB_SHA
and still refuses a checkout that is not the triggering commit. That refusal lives in
plan.mjs. run.mjs passes GITHUB_SHA as expected for a non-candidate run by calling
planVersion. A later git fetch may update only the release tag, and must throw unless
that tag still peels to the planned commit. Offline fixtures exercise those three checks
with no network, so CI runs them. Together with the gate, that releases the commit Test
passed on, with no explicit checkout `ref:`.
"""

from __future__ import annotations

import os
import re
import shlex
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE = WORKFLOWS / "release.yml"
TEST = WORKFLOWS / "test.yml"
GUIDELINES = "Living-Proof-Labs/collaboration-guidelines"
REPO = "Living-Proof-Labs/livingproof-booster-t1"


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name} did not parse as a mapping"
    return doc


def _triggers(doc: dict) -> dict:
    # PyYAML (YAML 1.1) parses the bare key `on` as boolean True.
    triggers = doc["on"] if "on" in doc else doc[True]
    assert isinstance(triggers, dict)
    return triggers


# --- a second, independent evaluator for the release job's `if:` -----------------------

_TOKEN = re.compile(r"'(?:[^']|'')*'|[A-Za-z_][\w.-]*|&&|\|\||==|!=|[()!]|\s+")


def gate_allows(expr: str, context: dict) -> bool:
    """Evaluate a restricted GitHub `if:` expression by translating it to Python."""
    text = expr.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()

    def lookup(path: str) -> object:
        cur: object = context
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return ""  # GitHub: a missing property is null, which == '' is true for
            cur = cur[part]
        return cur

    out: list[str] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        assert match, f"unsupported expression syntax near {text[pos:]!r}"
        tok = match.group(0)
        pos = match.end()
        if tok.isspace():
            continue
        if tok.startswith("'"):
            out.append(repr(tok[1:-1].replace("''", "'")))
        elif tok == "&&":
            out.append(" and ")
        elif tok == "||":
            out.append(" or ")
        elif tok == "!":
            out.append(" not ")
        elif tok in ("(", ")", "==", "!="):
            out.append(tok)
        elif tok in ("true", "false"):
            out.append(tok.capitalize())
        else:
            assert not text[pos:].lstrip().startswith("("), (
                f"function calls are not allowed: {tok}()"
            )
            out.append(f"lookup({tok!r})")
    return bool(eval("".join(out), {"__builtins__": {}}, {"lookup": lookup}))  # noqa: S307


def release_if() -> str:
    jobs = _load(RELEASE)["jobs"]
    job = jobs["release"]
    cond = job.get("if")
    assert isinstance(cond, str) and cond.strip(), "the release job needs an explicit if:"
    return cond


def completion(*, tip: str, tested: str, conclusion: str = "success", **overrides: str) -> dict:
    """The github context of the Release run that a completed Test run starts."""
    run = {
        "event": "push",
        "head_branch": "main",
        "head_repository": {"full_name": REPO},
        "conclusion": conclusion,
        "head_sha": tested,
    }
    run.update(overrides)
    return {
        "github": {
            "event_name": "workflow_run",
            "ref": "refs/heads/main",
            "sha": tip,  # workflow_run: last commit on the default branch
            "repository": REPO,
            "event": {"workflow_run": run},
        }
    }


# --- timeline model: which Test completions release, and what each release contains ---------


def simulate(events: list[tuple]) -> list[tuple[str, list[str]]]:
    """Replay pushes and Test completions. Returns (released commit, commits in the version)."""
    expr = release_if()
    pushed: list[str] = []
    released_upto = 0
    releases: list[tuple[str, list[str]]] = []
    for event in events:
        if event[0] == "push":
            pushed.append(event[1])
            continue
        _, tested, conclusion = event
        tip = pushed[-1]
        if gate_allows(expr, completion(tip=tip, tested=tested, conclusion=conclusion)):
            # The shared workflow checks out github.sha (== tip == tested) and
            # semantic-release releases everything since the last tag.
            assert tested == tip
            releases.append((tip, pushed[released_upto:]))
            released_upto = len(pushed)
    return releases


@pytest.mark.parametrize(
    ("name", "events", "expected"),
    [
        (
            "main moves while Test runs: the older run skips, the newer run batches both",
            [("push", "A"), ("push", "B"), ("done", "A", "success"), ("done", "B", "success")],
            [("B", ["A", "B"])],
        ),
        (
            "the newer Test finishes first: it releases both, the older one is then skipped",
            [("push", "A"), ("push", "B"), ("done", "B", "success"), ("done", "A", "success")],
            [("B", ["A", "B"])],
        ),
        (
            "Test finishes before the next push: two releases, one per commit",
            [("push", "A"), ("done", "A", "success"), ("push", "B"), ("done", "B", "success")],
            [("A", ["A"]), ("B", ["B"])],
        ),
        (
            "a failed Test on the tip never releases",
            [("push", "A"), ("done", "A", "failure")],
            [],
        ),
        (
            "the newer Test fails: nothing releases until a later green run batches all three",
            [
                ("push", "A"),
                ("push", "B"),
                ("done", "A", "success"),
                ("done", "B", "failure"),
                ("push", "C"),
                ("done", "C", "success"),
            ],
            [("C", ["A", "B", "C"])],
        ),
        (
            "a cancelled or skipped Test never releases",
            [("push", "A"), ("done", "A", "cancelled"), ("done", "A", "skipped")],
            [],
        ),
        (
            "a re-run of an old green Test cannot release once main moved on",
            [("push", "A"), ("done", "A", "success"), ("push", "B"), ("done", "A", "success")],
            [("A", ["A"])],
        ),
    ],
)
def test_release_race_model(name: str, events: list[tuple], expected: list) -> None:
    assert simulate(events) == expected, name


@pytest.mark.parametrize(
    "overrides",
    [
        {"event": "pull_request"},  # a fork or branch named main
        {"event": "merge_group", "head_branch": "gh-readonly-queue/main/pr-1-abc"},
        {"event": "workflow_dispatch"},
        {"head_branch": "feature"},
        {"head_repository": {"full_name": "someone-else/livingproof-booster-t1"}},
    ],
)
def test_only_same_repository_push_runs_on_main_can_release(overrides: dict) -> None:
    ctx = completion(tip="a" * 40, tested="a" * 40, **overrides)
    assert not gate_allows(release_if(), ctx)
    assert gate_allows(release_if(), completion(tip="a" * 40, tested="a" * 40))


# --- workflow contract ---------------------------------------------------------------------


def test_every_workflow_parses_and_declares_name_triggers_and_jobs() -> None:
    files = sorted(WORKFLOWS.glob("*.y*ml"))
    assert files
    names = []
    for path in files:
        doc = _load(path)
        assert isinstance(doc.get("name"), str) and doc["name"], path.name
        assert _triggers(doc), path.name
        assert isinstance(doc.get("jobs"), dict) and doc["jobs"], path.name
        names.append(doc["name"])
    assert len(names) == len(set(names)), f"duplicate workflow names: {names}"


def test_test_workflow_still_runs_on_push_to_main() -> None:
    """The gate needs a Test run with event == push on main, or nothing ever releases."""
    triggers = _triggers(_load(TEST))
    assert triggers["push"] == {"branches": ["main"]}
    assert "workflow_run" not in triggers


def test_release_workflow_is_one_privileged_call_and_runs_no_repository_code() -> None:
    doc = _load(RELEASE)
    assert doc["permissions"] == {"contents": "read"}
    jobs = doc["jobs"]
    assert set(jobs) == {"release", "deploy-manual"}
    name, job = "release", jobs["release"]
    assert jobs["deploy-manual"]["permissions"] == {
        "contents": "read",
        "pages": "write",
        "id-token": "write",
    }
    assert "steps" not in job, "no steps: nothing from the repository runs with contents: write"
    assert job["permissions"] == {"contents": "write"}
    assert job["uses"].startswith(f"{GUIDELINES}/.github/workflows/semantic-release.yml@v")
    assert not {"secrets", "with", "env", "continue-on-error", "needs"} & set(job), name
    # A new Test run must never cancel a release that is already tagging.
    assert "concurrency" not in doc
    assert "cancel-in-progress" not in RELEASE.read_text(encoding="utf-8")


def test_release_workflow_listens_only_to_the_test_workflow_by_its_exact_name() -> None:
    trigger = _triggers(_load(RELEASE))
    assert set(trigger) == {"workflow_run"}
    assert trigger["workflow_run"] == {
        "workflows": [_load(TEST)["name"]],
        "types": ["completed"],
        "branches": ["main"],
    }


# --- the shared workflow (network) ---------------------------------------------------------


def _pinned_ref() -> str:
    match = re.search(
        rf"uses:\s*{re.escape(GUIDELINES)}/\.github/workflows/semantic-release\.yml@(\S+)",
        RELEASE.read_text(encoding="utf-8"),
    )
    assert match
    return match.group(1)


def _raw(path: str) -> str:
    url = f"https://raw.githubusercontent.com/{GUIDELINES}/{_pinned_ref()}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


# The pinned scripts may wrap or requote this guard. The substance is fixed:
# plan.mjs reads HEAD, and run.mjs feeds it GITHUB_SHA on a non-candidate run.
_HEAD_MISMATCH = "Checked-out HEAD must match the triggering commit"
_JS_IDENT = r"[A-Za-z_$][\w$]*"
_GITHUB_SHA_EXPR = re.compile(r"""process\.env(?:\.GITHUB_SHA|\[\s*(['"])GITHUB_SHA\1\s*\])""")
_CANDIDATE_TRUE = frozenset({"candidate", "candidate===true", "candidate==true"})
_CANDIDATE_FALSE = frozenset({"!candidate", "candidate===false", "candidate==false"})


def _normalize_js(source: str) -> str:
    """Collapse whitespace and drop comments, leaving string literals intact."""
    out: list[str] = []
    i = 0
    n = len(source)
    pending_space = False

    def emit_space() -> None:
        nonlocal pending_space
        if pending_space and out and not out[-1].isspace():
            out.append(" ")
        pending_space = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and source[i] not in "\r\n":
                i += 1
            pending_space = True
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and source[i : i + 2] != "*/":
                i += 1
            i = min(i + 2, n)
            pending_space = True
            continue
        if ch in "'\"`":
            emit_space()
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                c = source[i]
                out.append(c)
                i += 1
                if c == "\\":
                    if i < n:
                        out.append(source[i])
                        i += 1
                    continue
                if c == quote:
                    break
            continue
        if ch.isspace():
            pending_space = True
            i += 1
            continue
        emit_space()
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _is_github_sha(expr: str) -> bool:
    return _GITHUB_SHA_EXPR.fullmatch(expr.strip()) is not None


def _unwrap(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        wraps = True
        for index, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and index != len(expr) - 1:
                    wraps = False
                    break
        if not wraps or depth != 0:
            break
        expr = expr[1:-1].strip()
    return expr


def _split_ternary(expr: str) -> tuple[str, str, str] | None:
    """Split one top-level ``cond ? when_true : when_false`` expression."""
    depth = 0
    question = None
    for index, ch in enumerate(expr):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and ch == "?" and question is None:
            question = index
        elif depth == 0 and ch == ":" and question is not None:
            return (
                expr[:question].strip(),
                expr[question + 1 : index].strip(),
                expr[index + 1 :].strip(),
            )
    return None


def _resolves_to_github_sha(text: str, expr: str, seen: frozenset[str] = frozenset()) -> bool:
    """True when expr is GITHUB_SHA, including through a local binding.

    `const sha = process.env.GITHUB_SHA` then using `sha` on the non-candidate
    arm is the same commit. The candidate arm must still be some other value.
    """
    expr = _unwrap(expr)
    if _is_github_sha(expr):
        return True
    if not re.fullmatch(_JS_IDENT, expr) or expr in seen:
        return False
    initial = _initializer(text, expr)
    if initial is None:
        return False
    return _resolves_to_github_sha(text, initial, seen | {expr})


def _selects_github_sha_for_non_candidate(text: str, expr: str) -> bool:
    """True when the non-candidate arm is GITHUB_SHA and the candidate arm is not."""
    parts = _split_ternary(_unwrap(expr))
    if parts is None:
        return False
    cond, when_true, when_false = (_unwrap(part) for part in parts)
    cond_flat = re.sub(r"\s+", "", cond)
    if cond_flat in _CANDIDATE_TRUE:
        non_candidate, candidate_arm = when_false, when_true
    elif cond_flat in _CANDIDATE_FALSE:
        non_candidate, candidate_arm = when_true, when_false
    else:
        return False
    return _resolves_to_github_sha(text, non_candidate) and not _resolves_to_github_sha(
        text, candidate_arm
    )


def _split_args(source: str) -> list[str]:
    args: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in source:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _call_arg_lists(text: str, callee: str) -> list[list[str]]:
    found: list[list[str]] = []
    start = 0
    while True:
        at = text.find(callee, start)
        if at < 0:
            break
        before = text[at - 1] if at else ""
        after = text[at + len(callee) : at + len(callee) + 1]
        if before.isalnum() or before in "_$" or after.isalnum() or after in "_$":
            start = at + len(callee)
            continue
        if text[:at].rstrip().endswith("function"):
            start = at + len(callee)
            continue
        cursor = at + len(callee)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            start = at + len(callee)
            continue
        depth = 1
        cursor += 1
        arg_at = cursor
        while cursor < len(text) and depth:
            if text[cursor] in "([{":
                depth += 1
            elif text[cursor] in ")]}":
                depth -= 1
            cursor += 1
        if depth != 0:
            break
        found.append(_split_args(text[arg_at : cursor - 1]))
        start = cursor
    return found


def _imported_plan_callees(text: str) -> list[str]:
    callees: list[str] = []
    named = re.compile(r"import\s*\{(?P<body>[^}]*)\}\s*from\s*(['\"])\./plan\.mjs\2")
    for match in named.finditer(text):
        for part in match.group("body").split(","):
            binding = re.fullmatch(
                rf"planVersion(?:\s+as\s+({_JS_IDENT}))?",
                part.strip(),
            )
            if binding:
                callees.append(binding.group(1) or "planVersion")
    namespace = re.compile(rf"import\s*\*\s*as\s+({_JS_IDENT})\s*from\s*(['\"])\./plan\.mjs\2")
    for match in namespace.finditer(text):
        callees.append(f"{match.group(1)}.planVersion")
    return callees


def _initializer(text: str, name: str) -> str | None:
    match = re.search(
        rf"\b(?:const|let|var)\s+{re.escape(name)}\s*=\s*(?P<expr>[^;]+)",
        text,
    )
    return match.group("expr").strip() if match else None


def _if_else_selects_github_sha(text: str, name: str) -> bool:
    """True when if/else assigns GITHUB_SHA on the non-candidate path only."""
    ident = re.escape(name)
    forms = (
        (
            rf"if\s*\(\s*candidate\s*\)\s*\{{?\s*{ident}\s*=\s*(?P<yes>[^;]+);\s*\}}?\s*"
            rf"else\s*\{{?\s*{ident}\s*=\s*(?P<no>[^;]+);",
            "no",
            "yes",
        ),
        (
            rf"if\s*\(\s*!\s*candidate\s*\)\s*\{{?\s*{ident}\s*=\s*(?P<no>[^;]+);\s*\}}?\s*"
            rf"else\s*\{{?\s*{ident}\s*=\s*(?P<yes>[^;]+);",
            "no",
            "yes",
        ),
    )
    for pattern, sha_side, other_side in forms:
        match = re.search(pattern, text)
        if match and _resolves_to_github_sha(text, match.group(sha_side)):
            return not _resolves_to_github_sha(text, match.group(other_side))
    return False


def _expr_passes_github_sha(text: str, expr: str) -> bool:
    expr = _unwrap(expr)
    if _selects_github_sha_for_non_candidate(text, expr):
        return True
    if re.fullmatch(_JS_IDENT, expr):
        initial = _initializer(text, expr)
        if initial is not None:
            return _selects_github_sha_for_non_candidate(text, initial)
        return _if_else_selects_github_sha(text, expr)
    return False


def _assert_plan_refuses_head_mismatch(source: str) -> None:
    """plan.mjs throws the HEAD-mismatch error when checked-out commit !== expected."""
    text = _normalize_js(source)
    reads_head = re.search(
        r"\b(?:const|let|var)\s+commit\s*=\s*[^;]*"
        r"""['"]rev-parse['"]\s*,\s*['"]HEAD['"]""",
        text,
    )
    assert reads_head, "plan.mjs must bind commit from git rev-parse HEAD"
    guard = re.search(
        r"if\s*\([^;]*?(?:\bcommit\s*!==\s*expected\b|\bexpected\s*!==\s*commit\b)"
        r"[^;]*?\)\s*\{?\s*throw\s+(?:new\s+)?Error\s*\(\s*"
        + r"""(['"`])"""
        + re.escape(_HEAD_MISMATCH)
        + r"\1",
        text,
        re.DOTALL,
    )
    assert guard, f"plan.mjs must throw {_HEAD_MISMATCH!r} when commit !== expected"


def _assert_run_passes_github_sha(source: str) -> None:
    """run.mjs imports planVersion and passes GITHUB_SHA for non-candidate runs."""
    text = _normalize_js(source)
    callees = _imported_plan_callees(text)
    assert callees, "run.mjs must import planVersion from ./plan.mjs"
    passed = any(
        len(args) >= 2 and _expr_passes_github_sha(text, args[1])
        for callee in callees
        for args in _call_arg_lists(text, callee)
    )
    assert passed, (
        "run.mjs must call planVersion with GITHUB_SHA as expected when the run is not a candidate"
    )


# git fetch here is a post-publish tag check, not a second checkout. These commands
# would move HEAD or the current branch; a tag-to-tag fetch does not.
_HEAD_MOVING_GIT = frozenset(
    {"pull", "checkout", "switch", "reset", "merge", "rebase", "update-ref"}
)
_FETCH_FLAGS = frozenset({"--force", "-f"})


def _match_closer(text: str, open_at: int, opener: str, closer: str) -> int | None:
    """Index of the bracket matching ``text[open_at]``, skipping string literals."""
    if open_at >= len(text) or text[open_at] != opener:
        return None
    depth = 0
    i = open_at
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"`":
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == ch:
                    i += 1
                    break
                i += 1
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _unquote_js(token: str) -> str | None:
    token = token.strip()
    if len(token) < 2 or token[0] not in "'\"`" or token[-1] != token[0]:
        return None
    return token[1:-1]


def _positional_params(param_src: str) -> tuple[str, ...]:
    names: list[str] = []
    for part in _split_args(param_src):
        part = part.strip()
        if not part or part[0] in "{[":
            break
        name = part.split("=", 1)[0].strip()
        if not re.fullmatch(_JS_IDENT, name):
            break
        names.append(name)
    return tuple(names)


def _js_functions(text: str) -> list[tuple[str, tuple[str, ...], str, int, int]]:
    found: list[tuple[str, tuple[str, ...], str, int, int]] = []
    for match in re.finditer(rf"\bfunction\s+({_JS_IDENT})\s*\(", text):
        paren = match.end() - 1
        close = _match_closer(text, paren, "(", ")")
        if close is None:
            continue
        brace = close + 1
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            continue
        end = _match_closer(text, brace, "{", "}")
        if end is None:
            continue
        found.append(
            (
                match.group(1),
                _positional_params(text[paren + 1 : close]),
                text[brace + 1 : end],
                match.start(),
                end,
            )
        )
    return found


def _js_string_values(text: str) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] not in "'\"`":
            i += 1
            continue
        start = i
        quote = text[i]
        i += 1
        chars: list[str] = []
        while i < n:
            if text[i] == "\\":
                if i + 1 < n:
                    chars.append(text[i + 1])
                i += 2
                continue
            if text[i] == quote:
                i += 1
                break
            chars.append(text[i])
            i += 1
        values.append((start, "".join(chars)))
    return values


def _git_arg_lists(text: str) -> list[tuple[int, list[str] | None]]:
    """Argument lists passed to git. None means an argument was not a literal."""
    found: list[tuple[int, list[str] | None]] = []
    for match in re.finditer(r"""(['"])git\1\s*,\s*\[""", text):
        bracket = match.end() - 1
        close = _match_closer(text, bracket, "[", "]")
        if close is None:
            found.append((match.start(), None))
            continue
        args: list[str] | None = []
        for raw in _split_args(text[bracket + 1 : close]):
            literal = _unquote_js(raw)
            if literal is None or args is None:
                args = None
                break
            args.append(literal)
        found.append((match.start(), args))
    for pos, value in _js_string_values(text):
        if not re.match(r"git\s+\S", value):
            continue
        try:
            parts = shlex.split(value)
        except ValueError:
            found.append((pos, None))
            continue
        if not parts or parts[0] != "git":
            found.append((pos, None))
            continue
        found.append((pos, parts[1:]))
    return found


def _is_plan_commit(text: str, expr: str) -> bool:
    expr = _unwrap(expr)
    if expr == "plan.commit":
        return True
    if re.fullmatch(_JS_IDENT, expr) is None:
        return False
    initial = _initializer(text, expr)
    return initial is not None and _unwrap(initial) == "plan.commit"


def _tag_to_tag_refspec(refspec: str, tag_param: str) -> bool:
    body = refspec.removeprefix("+")
    same = f"refs/tags/${{{tag_param}}}:refs/tags/${{{tag_param}}}"
    return re.sub(r"\s+", "", body) == same and "refs/heads" not in body


def _fetch_refspec(args: list[str], tag_param: str) -> str:
    flags = [arg for arg in args[1:] if arg.startswith("-")]
    positionals = [arg for arg in args[1:] if not arg.startswith("-")]
    assert set(flags) <= _FETCH_FLAGS, f"git fetch flags {flags} are not limited to a tag update"
    assert "--update-head-ok" not in args
    if len(positionals) == 2:
        remote, refspec = positionals
        assert ":" not in remote and not remote.startswith("refs/"), remote
    elif len(positionals) == 1:
        refspec = positionals[0]
    else:
        raise AssertionError(f"git fetch arguments {positionals} are not one tag refspec")
    assert _tag_to_tag_refspec(refspec, tag_param), (
        f"git fetch refspec {refspec!r} must copy only refs/tags/${{{tag_param}}} onto itself"
    )
    return refspec


def _peeled_tag_var(body: str, tag_param: str) -> str | None:
    needle = f"refs/tags/${{{tag_param}}}^{{commit}}"
    assign = re.compile(rf"\b(?:const|let|var)\s+(?P<var>{_JS_IDENT})\s*=\s*(?P<expr>[^;]*);")
    for match in assign.finditer(body):
        expr = re.sub(r"\s+", "", match.group("expr"))
        if "rev-parse" in expr and needle in expr:
            return match.group("var")
    return None


def _assert_fetches_stay_on_planned_commit(source: str) -> None:
    """A git fetch may update only the release tag, then must match plan.commit.

    planVersion has already refused a HEAD that is not the expected commit. The fetch
    runs after publish and must not select a different commit: its refspec writes
    refs/tags/<tag> and the function throws unless that tag peels to plan.commit.
    """
    text = _normalize_js(source)
    commands = _git_arg_lists(text)
    fetches: list[tuple[int, list[str]]] = []
    for pos, args in commands:
        assert args is not None, "run.mjs git invocation must use literal arguments"
        subcommand = next((arg for arg in args if not arg.startswith("-")), None)
        assert subcommand not in _HEAD_MOVING_GIT, (
            f"git {subcommand} can change the checked-out commit"
        )
        if subcommand == "fetch":
            fetches.append((pos, args))
    assert fetches, "run.mjs's git fetch must be visible to this check"
    functions = _js_functions(text)
    for pos, args in fetches:
        owners = [fn for fn in functions if fn[3] <= pos <= fn[4]]
        assert owners, "git fetch must be inside the function that verifies the tag"
        name, params, body, _start, _end = min(owners, key=lambda fn: fn[4] - fn[3])
        assert len(params) >= 2, "tag verification must take the tag and the expected commit"
        tag_param, expected_param = params[0], params[1]
        _fetch_refspec(args, tag_param)
        peeled = _peeled_tag_var(body, tag_param)
        assert peeled, "the fetched tag must be peeled with git rev-parse ^{commit}"
        guard = re.search(
            rf"if\s*\([^;]*?(?:\b{peeled}\s*!==\s*{re.escape(expected_param)}\b|"
            rf"\b{re.escape(expected_param)}\s*!==\s*{peeled}\b)"
            rf"[^;]*?\)\s*\{{?\s*throw\b",
            body,
        )
        assert guard, f"tag verification must throw when {peeled} !== {expected_param}"
        calls = _call_arg_lists(text, name)
        assert calls, f"{name} must be called with the planned commit"
        for call in calls:
            assert len(call) >= 2 and _is_plan_commit(text, call[1]), (
                "tag verification must compare the tag to plan.commit, "
                "the commit planVersion already checked"
            )


# Offline stand-ins for the pinned scripts. CI does not set LIVINGPROOF_NETWORK, so the
# network test never runs there. These fixtures call the same checks.
_PLAN_FIXTURE = """\
export async function planVersion(cwd, expected) {
  const commit = git('rev-parse', 'HEAD').trim();
  if (!shaPattern.test(expected ?? '') || commit !== expected) {
    throw new Error('Checked-out HEAD must match the triggering commit');
  }
  return { commit };
}
"""

_RUN_FIXTURE = """\
import { planVersion } from './plan.mjs';

export function verifyTagMatchesCommit(gitTag, expectedCommit) {
  execFileSync('git', ['fetch', 'origin', `refs/tags/${gitTag}:refs/tags/${gitTag}`, '--force']);
  const taggedCommit = execFileSync('git', ['rev-parse', '--verify', `refs/tags/${gitTag}^{commit}`]).trim();
  if (taggedCommit !== expectedCommit) {
    throw new Error('tag moved');
  }
}

export async function run() {
  const candidate = process.env.RELEASE_CANDIDATE === 'true';
  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : process.env.GITHUB_SHA;
  const plan = await planVersion(cwd, expected, { candidate });
  const head = plan.commit;
  verifyTagMatchesCommit(plan.gitTag, head);
}
"""

_RUN_FIXTURE_SHA_LOCAL = _RUN_FIXTURE.replace(
    "  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : process.env.GITHUB_SHA;",
    "  const sha = process.env.GITHUB_SHA;\n"
    "  const expected = candidate ? process.env.RELEASE_EXPECTED_COMMIT : sha;",
)

_PLAN_WITHOUT_COMPARISON = _PLAN_FIXTURE.replace(" || commit !== expected", "")
_PLAN_WITHOUT_MESSAGE = _PLAN_FIXTURE.replace(
    "Checked-out HEAD must match the triggering commit",
    "checkout accepted",
)
_PLAN_WITHOUT_HEAD = _PLAN_FIXTURE.replace(
    "const commit = git('rev-parse', 'HEAD').trim();",
    "const commit = expected;",
)
_RUN_WITHOUT_GITHUB_SHA = _RUN_FIXTURE.replace(
    "process.env.GITHUB_SHA",
    "process.env.RELEASE_EXPECTED_COMMIT",
)
_RUN_WITHOUT_FETCH = _RUN_FIXTURE.replace(
    "  execFileSync('git', ['fetch', 'origin', `refs/tags/${gitTag}:refs/tags/${gitTag}`, '--force']);\n",
    "",
)
_RUN_BRANCH_FETCH = _RUN_FIXTURE.replace(
    "`refs/tags/${gitTag}:refs/tags/${gitTag}`",
    "'refs/heads/main:refs/heads/main'",
)


def test_offline_fixture_accepts_head_guard_github_sha_and_tag_fetch() -> None:
    _assert_plan_refuses_head_mismatch(_PLAN_FIXTURE)
    _assert_run_passes_github_sha(_RUN_FIXTURE)
    _assert_fetches_stay_on_planned_commit(_RUN_FIXTURE)


def test_offline_fixture_accepts_github_sha_bound_before_the_ternary() -> None:
    _assert_run_passes_github_sha(_RUN_FIXTURE_SHA_LOCAL)


@pytest.mark.parametrize(
    "plan",
    [_PLAN_WITHOUT_COMPARISON, _PLAN_WITHOUT_MESSAGE, _PLAN_WITHOUT_HEAD],
)
def test_offline_fixture_refuses_a_plan_missing_the_head_guard(plan: str) -> None:
    with pytest.raises(AssertionError):
        _assert_plan_refuses_head_mismatch(plan)


def test_offline_fixture_refuses_a_run_missing_github_sha() -> None:
    with pytest.raises(AssertionError):
        _assert_run_passes_github_sha(_RUN_WITHOUT_GITHUB_SHA)


def test_offline_fixture_refuses_a_run_missing_the_tag_fetch() -> None:
    with pytest.raises(AssertionError):
        _assert_fetches_stay_on_planned_commit(_RUN_WITHOUT_FETCH)


def test_offline_fixture_refuses_a_branch_fetch() -> None:
    with pytest.raises(AssertionError):
        _assert_fetches_stay_on_planned_commit(_RUN_BRANCH_FETCH)


@pytest.mark.skipif(not os.environ.get("LIVINGPROOF_NETWORK"), reason="set LIVINGPROOF_NETWORK=1")
def test_pinned_shared_workflow_releases_exactly_github_sha() -> None:
    """Checkout is GITHUB_SHA, and a non-candidate run refuses any other HEAD.

    plan.mjs throws when the checked-out commit is not the expected commit. run.mjs
    imports planVersion and passes GITHUB_SHA as that expected commit when the run is
    not a candidate. Its git fetch only updates the published tag and throws unless
    that tag peels to plan.commit. The pin is the ref on this repo's release workflow.
    A guidelines bump that drops any of these has to be reviewed together with this gate.
    """
    workflow = yaml.safe_load(_raw(".github/workflows/semantic-release.yml"))
    steps = workflow["jobs"]["release"]["steps"]
    (checkout,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert "ref" not in checkout.get("with", {}), (
        "checkout must default to GITHUB_REF at GITHUB_SHA"
    )
    assert checkout["with"].get("persist-credentials") is False
    assert workflow["jobs"]["release"]["if"] == "github.ref == 'refs/heads/main'"
    runner = _raw(".github/actions/semantic-release/run.mjs")
    plan = _raw(".github/actions/semantic-release/plan.mjs")
    _assert_plan_refuses_head_mismatch(plan)
    _assert_run_passes_github_sha(runner)
    _assert_fetches_stay_on_planned_commit(runner)
