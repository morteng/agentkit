"""Optional transports (extras: agentkit[fastapi])."""

try:
    from agentkit.transports.websocket import (
        WSAuth,  # pyright: ignore[reportUnusedImport]
        mount_websocket_route,  # pyright: ignore[reportUnusedImport]
    )

    __all__ = ["WSAuth", "mount_websocket_route"]
except ImportError:  # FastAPI not installed
    __all__ = []
