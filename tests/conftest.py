"""Shared fixtures: load the captured GLM-5.2 tool-call samples once."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "glm_tool_call_samples.json"


@pytest.fixture(scope="session")
def samples() -> dict:
    with _FIXTURE.open(encoding="utf-8") as fh:
        return json.load(fh)
