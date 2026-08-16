"""Optional transports (extras: agentkit[fastapi])."""

try:
    # Redundant `X as X` aliases: PEP 484's explicit re-export form. Without
    # them these look like unused imports in a package __init__ — which is
    # exactly what they resemble and exactly what they are not. The per-name
    # suppression the file used to carry did the same job by silencing the
    # check rather than satisfying it.
    from agentkit.transports.websocket import (
        ANY_ORIGIN as ANY_ORIGIN,
    )
    from agentkit.transports.websocket import (
        WS_CLOSE_LIFETIME_EXCEEDED as WS_CLOSE_LIFETIME_EXCEEDED,
    )
    from agentkit.transports.websocket import (
        WS_CLOSE_LIFETIME_EXCEEDED_REASON as WS_CLOSE_LIFETIME_EXCEEDED_REASON,
    )
    from agentkit.transports.websocket import (
        InsecureAllowAllAuth as InsecureAllowAllAuth,
    )
    from agentkit.transports.websocket import (
        InsecureTransportWarning as InsecureTransportWarning,
    )
    from agentkit.transports.websocket import (
        SameSocketApprovalAuthority as SameSocketApprovalAuthority,
    )
    from agentkit.transports.websocket import (
        WSApprovalAuthority as WSApprovalAuthority,
    )
    from agentkit.transports.websocket import (
        WSAuth as WSAuth,
    )
    from agentkit.transports.websocket import (
        mount_websocket_route as mount_websocket_route,
    )

    __all__ = [
        "ANY_ORIGIN",
        "WS_CLOSE_LIFETIME_EXCEEDED",
        "WS_CLOSE_LIFETIME_EXCEEDED_REASON",
        "InsecureAllowAllAuth",
        "InsecureTransportWarning",
        "SameSocketApprovalAuthority",
        "WSApprovalAuthority",
        "WSAuth",
        "mount_websocket_route",
    ]
except ImportError:  # FastAPI not installed
    __all__ = []
