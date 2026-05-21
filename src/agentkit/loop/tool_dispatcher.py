"""ToolDispatcher — runs tool calls with the right concurrency policy."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import RiskLevel, ToolCall, ToolResult, ToolSpec


@dataclass(frozen=True)
class DispatchPolicy:
    max_parallel: int = 8
    """Cap on concurrent tool executions when running in parallel mode."""


class ToolDispatcher:
    def __init__(self, *, registry: ToolRegistry, policy: DispatchPolicy) -> None:
        self._registry = registry
        self._policy = policy

    async def run(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        if not calls:
            return []
        if self._safe_for_parallel(calls):
            return await self._run_parallel(calls, ctx)
        return await self._run_sequential(calls, ctx)

    def _safe_for_parallel(self, calls: Sequence[ToolCall]) -> bool:
        """All calls must be READ + idempotent for parallel dispatch.

        An unknown tool name has no spec — treat it as not-parallel-safe so
        dispatch falls to the sequential path and ``registry.invoke`` produces
        the error ToolResult. Never raise here: a raised exception bubbles to
        the orchestrator and kills the turn with no result for the model.
        """
        for call in calls:
            spec = self._find_spec(call.name)
            if spec is None or spec.risk != RiskLevel.READ or not spec.idempotent:
                return False
        return True

    def _find_spec(self, name: str) -> ToolSpec | None:
        for spec in self._registry.list_specs():
            if spec.name == name:
                return spec
        return None

    async def _run_parallel(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        sem = asyncio.Semaphore(self._policy.max_parallel)

        async def _bounded(call: ToolCall) -> ToolResult:
            async with sem:
                return await self._registry.invoke(call, ctx)

        results: list[ToolResult] = list(await asyncio.gather(*(_bounded(c) for c in calls)))
        return results

    async def _run_sequential(self, calls: Sequence[ToolCall], ctx: Any) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            results.append(await self._registry.invoke(call, ctx))
        return results
