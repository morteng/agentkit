"""StdioMCPClient — talks to an MCP server subprocess over JSON-RPC stdio."""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentkit._content import Provenance
from agentkit._logging import get_logger
from agentkit.mcp_client.base import MCPClient, ProgressCallback
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolError,
    ToolResult,
    ToolSpec,
)

log = get_logger(__name__)

MCPToolClassifier = Callable[[ToolSpec], ToolSpec | None]
"""Consumer hook that assigns real agentkit metadata to an MCP tool.

Called once per tool during :meth:`StdioMCPClient.list_tools` with the
fail-closed default spec. Return a refined :class:`ToolSpec` (same ``name``)
to classify the tool, or ``None`` to say "I do not know this tool" — which
leaves it at the unclassified default, or drops it entirely under
``require_classification``.
"""

UNCLASSIFIED_RISK = RiskLevel.HIGH_WRITE
"""Risk assumed for an MCP tool nobody has classified.

An external MCP server is code we did not write, describing itself. Its
self-description says nothing about blast radius, so the only safe reading of
"unknown" is "high" — ``HIGH_WRITE`` is the lowest rung that
``DEFAULT_APPROVAL_POLICY`` routes to ``NEEDS_USER``. The previous default of
``LOW_WRITE`` is auto-approved by that same table, which meant connecting a
server silently granted it unattended write access.
"""

UNCLASSIFIED_APPROVAL = ApprovalPolicy.ALWAYS
"""Approval policy for an unclassified MCP tool.

``ALWAYS`` rather than ``BY_RISK`` on purpose: ``BY_RISK`` is only as safe as
the deployment's risk table, and a consumer that remaps ``HIGH_WRITE`` to
auto-approve for its own trusted builtins would silently re-open this hole for
third-party servers. ``ALWAYS`` short-circuits the table. A consumer that
genuinely wants an MCP tool unattended must say so per tool, via a classifier
or ``RiskBasedApprovalGate(policy_overrides=...)``.
"""

UNCLASSIFIED_SIDE_EFFECTS = SideEffects.EXTERNAL_IRREVERSIBLE
"""Side effects assumed for an unclassified MCP tool — the worst case."""


class StdioMCPClient(MCPClient):
    """Spawn an MCP server subprocess and speak JSON-RPC over its stdio.

    Tools arrive unclassified: the MCP protocol carries a name, a description
    and a JSON Schema, and nothing that maps onto agentkit's risk model. They
    are therefore treated as the most dangerous thing they could be until a
    consumer says otherwise — see :data:`UNCLASSIFIED_RISK` and ``classifier``.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        startup_timeout_seconds: float = 10.0,
        classifier: MCPToolClassifier | None = None,
        require_classification: bool = False,
    ) -> None:
        """Create a client for one MCP server subprocess.

        Args:
            name: Server name; the registry uses it to namespace tool names.
            command: argv of the server process. Must be non-empty.
            env: Environment for the subprocess.
            cwd: Working directory for the subprocess.
            startup_timeout_seconds: Cap on how long the handshake may take.
            classifier: Per-tool classification hook. Receives the fail-closed
                default spec and returns the real one (same ``name``), or
                ``None`` for "unknown".
            require_classification: When True, a tool the classifier declines
                to classify is not exposed at all — it is dropped from
                ``list_tools`` and refused by ``call_tool``. Use this when an
                MCP server must be fully described before any of it is
                reachable. The default (False) still fails closed, just less
                bluntly: unclassified tools stay callable but always require
                user approval.
        """
        if not command:
            raise ValueError("command must not be empty")
        self.name = name
        self._params = StdioServerParameters(
            command=command[0],
            args=command[1:],
            env=env,
            cwd=cwd,
        )
        self._startup_timeout = startup_timeout_seconds
        self._classifier = classifier
        self._require_classification = require_classification
        self._classified: set[str] = set()
        self._session: ClientSession | None = None
        self._stdio_ctx: Any = None
        self._client_ctx: Any = None

    async def initialize(self) -> None:
        self._stdio_ctx = stdio_client(self._params)
        try:
            read, write = await asyncio.wait_for(
                self._stdio_ctx.__aenter__(),
                timeout=self._startup_timeout,
            )
            self._client_ctx = ClientSession(read, write)
            session: ClientSession = await self._client_ctx.__aenter__()
            await session.initialize()
            self._session = session
        except BaseException:
            # Subprocess may have spawned and/or session may be partially
            # entered. shutdown() is idempotent and nulls all handles.
            await self.shutdown()
            raise

    async def list_tools(self) -> list[ToolSpec]:
        """List the server's tools, classified or fail-closed.

        Every tool is first rendered as an unclassified spec (approval always
        required); the consumer's ``classifier`` may then replace it with the
        real thing. Under ``require_classification`` anything the classifier
        declines is dropped here, which is what makes it uncallable — the
        registry only ever learns about tools this method returns.
        """
        if self._session is None:
            raise RuntimeError("call initialize() first")
        result = await self._session.list_tools()
        specs: list[ToolSpec] = []
        self._classified = set()
        for tool in result.tools:
            default = _mcp_tool_to_spec(tool)
            refined = self._classifier(default) if self._classifier is not None else None
            if refined is None:
                if self._require_classification:
                    log.warning(
                        "mcp_tool_dropped_unclassified",
                        server=self.name,
                        tool=default.name,
                    )
                    continue
                log.warning(
                    "mcp_tool_unclassified",
                    server=self.name,
                    tool=default.name,
                    risk=str(default.risk),
                    requires_approval=str(default.requires_approval),
                )
                specs.append(default)
                continue
            if refined.name != default.name:
                raise ValueError(
                    "classifier must not rename a tool: "
                    f"{default.name!r} was returned as {refined.name!r}"
                )
            self._classified.add(refined.name)
            specs.append(refined)
        return specs

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult:
        if self._session is None:
            raise RuntimeError("call initialize() first")
        if self._require_classification and name not in self._classified:
            # Belt to list_tools' braces: a caller holding a stale spec (or
            # reaching past the registry) must not be able to run a tool the
            # deployment never classified.
            msg = (
                f"tool {name!r} on MCP server {self.name!r} has not been classified; "
                "this client requires an explicit classification before a tool is callable"
            )
            log.warning("mcp_call_refused_unclassified", server=self.name, tool=name)
            return ToolResult(
                call_id="",
                status="denied",
                content=[ContentBlockOut(type="text", text=msg)],
                error=ToolError(code="mcp_tool_unclassified", message=msg),
                duration_ms=0,
                cached=False,
            )
        # The MCP SDK's progress callback signature is
        # (progress, total, message) — adapt to our (message, progress, total)
        # so callers can forward straight to ctx.report_tool_progress.
        progress_callback = None
        if on_progress is not None:

            async def _adapter(progress: float, total: float | None, message: str | None) -> None:
                await on_progress(message or "", progress, total)

            progress_callback = _adapter

        started = time.perf_counter()
        try:
            response = await self._session.call_tool(
                name, arguments, progress_callback=progress_callback
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                error=ToolError(code="mcp_call_failed", message=str(exc)),
                duration_ms=elapsed,
                cached=False,
            )
        elapsed = int((time.perf_counter() - started) * 1000)
        if getattr(response, "isError", False):
            return ToolResult(
                call_id="",
                status="error",
                content=[],
                error=ToolError(code="mcp_tool_error", message=_text_of(response.content)),
                duration_ms=elapsed,
                cached=False,
            )
        return ToolResult(
            call_id="",
            status="ok",
            content=[_mcp_content_to_block(c) for c in response.content],
            error=None,
            duration_ms=elapsed,
            cached=False,
            # This content came from a subprocess we spoke JSON-RPC to, not
            # code we wrote — the same "outside world" UNCLASSIFIED_RISK
            # already fails closed for above. A compromised or malicious MCP
            # server can shape its response to look like an instruction the
            # same way a scraped web page can, so the taint guard needs to see
            # it: see agentkit.guards.taint / Provenance.UNTRUSTED.
            provenance=Provenance.UNTRUSTED,
        )

    async def shutdown(self) -> None:
        if self._client_ctx is not None:
            await self._client_ctx.__aexit__(None, None, None)
            self._client_ctx = None
        if self._stdio_ctx is not None:
            await self._stdio_ctx.__aexit__(None, None, None)
            self._stdio_ctx = None
        self._session = None

    async def health_check(self) -> bool:
        if self._session is None:
            return False
        try:
            await self._session.list_tools()
            return True
        except Exception:
            return False


def _mcp_tool_to_spec(tool: Any) -> ToolSpec:
    """Translate the official ``mcp.types.Tool`` into an *unclassified* ToolSpec.

    An MCP tool description carries no agentkit metadata — no risk level, no
    idempotency, no notion of reversibility — so this translation cannot know
    what the tool does. It therefore assumes the worst on every axis, which is
    the only reading of "unknown" that fails closed: the tool needs user
    approval before it runs, every time, until a consumer classifies it (see
    ``StdioMCPClient(classifier=...)``).
    """
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.inputSchema or {"type": "object"},
        returns=None,
        risk=UNCLASSIFIED_RISK,
        idempotent=False,
        side_effects=UNCLASSIFIED_SIDE_EFFECTS,
        requires_approval=UNCLASSIFIED_APPROVAL,
        cache_ttl_seconds=None,
        timeout_seconds=30.0,
    )


def _mcp_content_to_block(content: Any) -> ContentBlockOut:
    if getattr(content, "type", None) == "image":
        return ContentBlockOut(
            type="image",
            image_data=getattr(content, "data", None),
            media_type=getattr(content, "mimeType", None),
        )
    text = getattr(content, "text", "")
    return ContentBlockOut(type="text", text=text)


def _text_of(content_list: list[Any] | None) -> str:
    parts: list[str] = [getattr(c, "text", "") for c in (content_list or [])]
    return "\n".join(parts)
