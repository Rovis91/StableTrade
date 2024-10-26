import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Create a temporary directory that's properly cleaned up."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        try:
            shutil.rmtree(temp_path)
        except PermissionError:
            pass