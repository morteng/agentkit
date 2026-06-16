import re
from importlib.metadata import version


def test_package_imports():
    import agentkit

    # __version__ is derived from the installed package metadata (not a frozen
    # literal), so it always matches pyproject and never drifts on a release.
    assert agentkit.__version__ == version("agentkit")
    # Sanity: it resolved to a real semver, not the not-installed fallback.
    assert agentkit.__version__ != "0.0.0+unknown"
    assert re.match(r"^\d+\.\d+\.\d+", agentkit.__version__)
