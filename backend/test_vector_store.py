"""
Standalone test runner for Qdrant Vector Store.
Runs pytest on tests/test_vector_store.py or executes directly with python test_vector_store.py.
"""

from pathlib import Path
import sys
import pytest

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    test_path = BASE_DIR / "tests" / "test_vector_store.py"
    exit_code = pytest.main(["-v", str(test_path)])
    sys.exit(exit_code)
