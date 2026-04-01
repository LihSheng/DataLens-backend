"""Pytest fixtures and configuration."""
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
