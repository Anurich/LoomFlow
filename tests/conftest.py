"""Pytest configuration: enable anyio's pytest plugin on the asyncio backend."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
