# SPDX-License-Identifier: Apache-2.0
"""Build the repository manual without importing or executing robot code."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/docs"
REPO = "https://github.com/Living-Proof-Labs/livingproof-booster-t1"


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def build():
    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    source = OUT / "source"
    if OUT.exists():
        shutil.rmtree(OUT)
    source.mkdir(parents=True)
    mapping = {
        "README.md": "guide.md",
        "docs/index.md": "index.md",
        "docs/contribute.md": "contribute.md",
        "docs/verification.md": "verification.md",
    }
    for original, target in mapping.items():
        text = (ROOT / original).read_text()
        text = text.replace("](docs/index.md)", "](index.md)").replace(
            "](../README.md)", "](guide.md)"
        )
        (source / target).write_text(text)
    (source / "license.md").write_text(
        "# License\n\n```text\n" + (ROOT / "LICENSE").read_text() + "\n```\n"
    )
    tasks = [
        "# Contributor tasks",
        "",
        "Generated from the reviewed root Taskfile with Task 3.54.0.",
        "Run these host commands from the checkout with the guide prerequisites installed.",
        "Summaries show commands; the docs build does not execute them.",
        "",
    ]
    for task in json.loads(run("task", "--list", "--json"))["tasks"]:
        tasks += [
            f"## {task['name']}",
            "",
            "```text",
            run("task", "--summary", task["name"]),
            "```",
            "",
        ]
    (source / "tasks.md").write_text("\n".join(tasks))
    lock = json.loads((ROOT / "docs-brand.lock.json").read_text())
    with urlopen(
        f"https://raw.githubusercontent.com/{lock['repository']}/{lock['commit']}/{lock['path']}",
        timeout=60,
    ) as response:
        brand = response.read()
    if hashlib.sha256(brand).hexdigest() != lock["sha256"]:
        raise ValueError("Docs brand checksum mismatch")
    (source / "assets").mkdir()
    (source / "assets/brand.css").write_bytes(brand)
    record = {
        "source_commit": commit,
        "source_dirty": dirty,
        "brand": lock,
        "sources": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [*mapping, "Taskfile.yml", "LICENSE"]
        },
    }
    (source / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    (source / "version.md").write_text(
        "# Version and source\n\nThis is development documentation, not a package release.\n\n"
        f"Source commit: [`{commit}`]({REPO}/tree/{commit}). Dirty preview: **{dirty}**.\n\n"
        "A dirty preview includes uncommitted changes; [build metadata](build.json) records source hashes.\n\n"
        "Supported context: the dependencies and verification lock in this source revision. "
        f"Historical rendered manuals are not maintained; use [source tags]({REPO}/tags).\n\n"
        f"Docs recipe and brand: {lock['version']} at `{lock['commit']}`.\n"
    )
    for page in source.glob("*.md"):
        with page.open("a") as handle:
            handle.write(
                f"\n\n---\n\n[Documentation home](https://docs.livingprooflabs.com/) · "
                f'[Source · development]({REPO}/tree/{commit} "{commit}") · '
                f"[Version and source](version.md) · [Help / report an issue]({REPO}/issues) · "
                "[License](license.md)\n"
            )
    config = yaml.safe_load((ROOT / "tools/docs/mkdocs.yml").read_text())
    config.update(docs_dir="source", site_dir="html")
    config_path = OUT / "mkdocs.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run(
        [sys.executable, "-m", "zensical", "build", "--clean", "-f", str(config_path)],
        cwd=ROOT,
        check=True,
    )
    verify()


class VisibleHTMLParser(HTMLParser):
    """Ignore non-rendered HTML; CSS visibility is checked in the browser tests."""

    VOID_TAGS = (
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    )

    def __init__(self):
        super().__init__()
        self.hidden_elements = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        style = re.sub(r"\s+", "", attrs.get("style", "").lower())
        hidden = (
            tag in ("script", "style", "template")
            or "hidden" in attrs
            or "display:none" in style
            or "visibility:hidden" in style
        )
        if tag not in self.VOID_TAGS:
            self.hidden_elements.append((tag, hidden))

    def handle_endtag(self, tag):
        for index in range(len(self.hidden_elements) - 1, -1, -1):
            if self.hidden_elements[index][0] == tag:
                del self.hidden_elements[index:]
                break


def commit_pattern(commit):
    """Match this build's full or abbreviated identity, not arbitrary hex words."""
    prefixes = "|".join(re.escape(commit[:length]) for length in range(7, 41))
    return re.compile(r"\b(?:" + prefixes + r")\b", re.IGNORECASE)


class Links(VisibleHTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.in_footer = 0
        self.in_link = False
        self.link_attrs = {}
        self.link_text = []
        self.source_links = []
        self.footer_text = []
        self.visible_text = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        attrs = dict(attrs)
        if tag not in self.VOID_TAGS and (tag == "footer" or self.in_footer):
            self.in_footer += 1
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "a":
            self.in_link = True
            self.link_attrs = attrs
            self.link_text = []

    def handle_endtag(self, tag):
        if tag in self.VOID_TAGS:
            return
        super().handle_endtag(tag)
        if self.in_footer:
            self.in_footer -= 1
        if tag == "a" and self.in_link:
            text = "".join(self.link_text).strip()
            if text.startswith("Source · development"):
                self.source_links.append(
                    (text, self.link_attrs.get("href", ""), self.link_attrs.get("title", ""))
                )
            self.in_link = False

    def handle_data(self, data):
        if any(hidden for _, hidden in self.hidden_elements):
            return
        self.visible_text.append(data)
        if self.in_footer:
            self.footer_text.append(data)
        if self.in_link:
            self.link_text.append(data)


def verify():
    html = OUT / "html"
    pages = {p.resolve(): Links(p.read_text()) for p in html.rglob("*.html")}
    commit = json.loads((html / "build.json").read_text())["source_commit"]
    assert re.fullmatch("[0-9a-f]{40}", commit)
    sha_pattern = commit_pattern(commit)
    for path, parsed in pages.items():
        text = path.read_text()
        assert "assets/brand.css" in text, path
        assert 'data-md-color-primary="indigo"' not in text, path
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or url.path.startswith("/"):
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), (path, link)
            if url.fragment and target in pages:
                assert unquote(url.fragment) in pages[target].ids, (path, link)
        footer_text = " ".join(parsed.footer_text)
        assert not sha_pattern.search(footer_text), (path, footer_text)
        for link_text, href, title in parsed.source_links:
            assert link_text == "Source · development", (path, link_text)
            assert not sha_pattern.search(link_text), (path, link_text)
            assert commit in href, (path, href)
            assert commit in title, (path, title)
        if path.name != "version.html":
            visible = " ".join(parsed.visible_text)
            assert not sha_pattern.search(visible), (path, visible)
    print(f"PASS: {len(pages)} HTML pages; local links/fragments, brand, and source identity")


if __name__ == "__main__":
    build()
