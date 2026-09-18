"""Smoke test: the package imports and declares a version. Replaced by real tests in Phase 2."""

import clinical


def test_version_is_declared() -> None:
    assert clinical.__version__ == "0.1.0"
