"""
Standalone test runner for Hybrid Search Engine.
Runs pytest on tests/test_hybrid_retriever.py or executes directly with `python test_hybrid_retriever.py`.
"""

from pathlib import Path
import sys
import pytest

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    test_path = BASE_DIR / "tests" / "test_hybrid_retriever.py"
    exit_code = pytest.main(["-v", str(test_path)])
    sys.exit(exit_code)
