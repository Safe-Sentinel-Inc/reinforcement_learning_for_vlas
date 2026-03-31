"""Shared test fixtures."""
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

@pytest.fixture
def project_root():
    return PROJECT_ROOT
