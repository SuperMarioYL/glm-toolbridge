"""Single-source-of-truth version test.

Asserts every user-facing version surface agrees. On the shipped v0.4.0 tag
this fails: web/site.json carried NO content_version field, so the
content_version == VERSION assertion raised a KeyError — proving the drift
was real. After the v0.5.0 bump all surfaces read 0.5.0.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import glm_toolbridge

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET = "0.5.0"


def _read_version_file() -> str:
    return (_REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _read_pyproject_version() -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "pyproject.toml has no version field"
    return match.group(1)


def _read_site_content_version() -> str:
    site = json.loads((_REPO_ROOT / "web" / "site.json").read_text(encoding="utf-8"))
    assert "content_version" in site, "web/site.json has no content_version field"
    return site["content_version"]


def _read_changelog_head_version() -> str:
    text = (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    # The first "## [x.y.z] -" heading is the head (latest) entry.
    match = re.search(r"^##\s+\[([0-9.]+)\]\s*-\s", text, re.MULTILINE)
    assert match is not None, "CHANGELOG.md has no version heading"
    return match.group(1)


def test_version_surfaces_agree():
    surfaces = {
        "VERSION file": _read_version_file(),
        "glm_toolbridge.__version__": glm_toolbridge.__version__,
        "pyproject.toml version": _read_pyproject_version(),
        "web/site.json content_version": _read_site_content_version(),
        "CHANGELOG head": _read_changelog_head_version(),
    }
    for name, value in surfaces.items():
        assert value == _TARGET, f"{name} is {value!r}, expected {_TARGET!r}"
    # Redundant belt-and-braces: every surface equals every other surface.
    values = set(surfaces.values())
    assert values == {_TARGET}, f"version surfaces disagree: {surfaces}"
