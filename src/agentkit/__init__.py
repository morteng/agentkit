"""agentkit — domain-blind agent runtime.

Public API stabilises at v1.0. Symbols importable from the top-level package
are part of the API contract; everything else is internal.
"""

__version__ = "0.1.0"

# Public API re-exports land here as modules become available.
# Until v0.1.0 ships, `from agentkit import X` will raise ImportError for
# anything not yet implemented — by design.
