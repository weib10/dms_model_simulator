"""Shared test fixtures: a made-up consumer layout with a class_map (tests/fixtures/mapped_layout.yaml)."""
from pathlib import Path
import pytest
from dms_model_simulator.config import load_yaml

MAPPED_LAYOUT = Path(__file__).parent / "fixtures" / "mapped_layout.yaml"
MAPPED_CLASSES = {"belt_strap", "lids_open", "jaw_open", "handset", "face_box", "smoke_stick", "lids_shut", "jaw_shut"}


@pytest.fixture
def mapped_layout_path() -> Path:
    return MAPPED_LAYOUT


@pytest.fixture
def mapped_spec() -> dict:
    """A fresh dict of the fixture layout (tests may change it)."""
    return load_yaml(MAPPED_LAYOUT)
