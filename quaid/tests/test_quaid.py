"""
Unit and regression test for the quaid package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import quaid


def test_quaid_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "quaid" in sys.modules
