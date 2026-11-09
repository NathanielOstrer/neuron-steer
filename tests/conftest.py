import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption("--model", action="store_true", help="run tests that load the real model")


def pytest_configure(config):
    config.addinivalue_line("markers", "model: loads the real 8-bit model (about 4.5 GB)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--model"):
        return
    skip = pytest.mark.skip(reason="needs --model")
    for it in items:
        if "model" in it.keywords:
            it.add_marker(skip)
