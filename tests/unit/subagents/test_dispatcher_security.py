"""Security properties of ``kit.subagent.spawn``.

Every argument to a spawn — prompt, tool allowlist, and the ``context`` dict —
is authored by the model, so this file is written from the attacker's side.
The audit found three ways in:

* ``context`` was merged *over* the runtime's own metadata, so a model could
  set ``subagent_depth`` (defeating the recursion guard), ``owner``
  (impersonating a principal), or ``allowed_tools``;
* ``allowed_tools`` was written into metadata and never read, so the child
  inherited its parent's whole tool surface including privileged tools;
* a call needing human approval inside a child was silently dropped along
  with an orphaned checkpoint.
"""

import asyncio
from typing import Any

import pytest

from agentkit._ids import OwnerId
from agentkit.events import ToolCallProgress
from agentkit.guards.approval import RiskBasedApprovalGate
from agentkit.loop.context import TurnContext
from agentkit.loop.message_builder import MessageBuilder
from agentkit.loop.tool_dispatcher import DispatchPolicy, ToolDispatcher
from agentkit.providers.fakes import FakeProvider, ScriptedResponse
from agentkit.store.fakes import FakeCheckpointStore
from agentkit.subagents.dispatcher import (
    SubagentApprovalRequired,
    SubagentContextRejected,
    SubagentDepthExceeded,
    SubagentDispatcher,
)
from agentkit.subagents.isolation import RESERVED_CONTEXT_KEYS, RestrictedToolRegistry
from agentkit.tools.registry import ToolRegistry
from agentkit.tools.spec import (
    ApprovalPolicy,
    ContentBlockOut,
    RiskLevel,
    SideEffects,
    ToolCall,
    ToolResult,
    ToolSpec,
)

PARENT_OWNER = OwnerId("u:parent")


def _spec(
    name: str,
    *,
    risk: RiskLevel = RiskLevel.READ,
    approval: ApprovalPolicy = ApprovalPolicy.BY_RISK,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}},
        returns=None,
        risk=risk,
        idempotent=True,
        side_effects=SideEffects.NONE,
        requires_approval=approval,
        cache_ttl_seconds=None,
        timeout_seconds=5.0,
    )


class _Env:
    """A parent turn wired to a SubagentDispatcher, plus observability hooks."""

    def __init__(self, *, child_script: list[ScriptedResponse]) -> None:
        self.invoked: list[str] = []
        self.captured_metadata: list[dict[str, Any]] = []
        self.registry = ToolRegistry()
        self.registry.register_default_builtins()

        async def _record(tool: str) -> ToolResult:
            self.invoked.append(tool)
            return ToolResult(
                call_id="",
                status="ok",
                content=[ContentBlockOut(type="text", text=f"{tool} ran")],
                error=None,
                duration_ms=0,
                cached=False,
            )

        async def _safe(_args: dict[str, Any], ctx: TurnContext) -> ToolResult:
            self.captured_metadata.append(dict(ctx.metadata))
            return await _record("safe.echo")

        async def _wipe(_args: dict[str, Any], _ctx: TurnContext) -> ToolResult:
            return await _record("admin.wipe")

        async def _write(_args: dict[str, Any], _ctx: TurnContext) -> ToolResult:
            return await _record("admin.write")

        self.registry.register_builtin(_spec("safe.echo"), _safe)
        self.registry.register_builtin(_spec("admin.wipe", risk=RiskLevel.DESTRUCTIVE), _wipe)
        self.registry.register_builtin(_spec("admin.write", risk=RiskLevel.HIGH_WRITE), _write)

        self.provider = FakeProvider().script(*child_script)
        self.checkpoints = FakeCheckpointStore()
        self.deps: dict[str, Any] = {
            "provider": self.provider,
            "message_builder": MessageBuilder(model="m", max_tokens=128),
            "registry": self.registry,
            "system_blocks": [],
            "approval_gate": RiskBasedApprovalGate(),
            "dispatcher": ToolDispatcher(
                registry=self.registry, policy=DispatchPolicy(max_parallel=4)
            ),
            "checkpoint_store": self.checkpoints,
            "max_iterations": 5,
        }
        self.dispatcher = SubagentDispatcher(deps=self.deps, max_depth=2)

        self.parent_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.parent = TurnContext.empty(call_id="parent-call")
        self.parent.event_queue = self.parent_queue
        self.parent.metadata["owner"] = PARENT_OWNER

    def progress_messages(self) -> list[str]:
        out: list[str] = []
        while not self.parent_queue.empty():
            item = self.parent_queue.get_nowait()
            if isinstance(item, ToolCallProgress):
                out.append(item.message)
        return out


def _text_env(text: str = "done") -> _Env:
    return _Env(child_script=[FakeProvider.text(text)])


# ---- the context dict cannot escalate ---------------------------------------


@pytest.mark.parametrize("key", sorted(RESERVED_CONTEXT_KEYS))
async def test_every_reserved_key_is_rejected(key: str):
    env = _text_env()
    with pytest.raises(SubagentContextRejected, match=key):
        await env.dispatcher.spawn(
            env.parent, prompt="p", tools=["safe.echo"], extra_context={key: "attacker"}
        )


async def test_context_cannot_reset_the_recursion_guard():
    """``subagent_depth: 0`` would make the depth cap unreachable forever."""
    env = _text_env()
    env.parent.metadata["subagent_depth"] = 1
    with pytest.raises(SubagentContextRejected, match="subagent_depth"):
        await env.dispatcher.spawn(
            env.parent, prompt="p", tools=[], extra_context={"subagent_depth": 0}
        )


async def test_a_depth_reset_attempt_at_the_cap_still_hits_the_cap():
    """Order matters: the depth check runs before the context is even read."""
    env = _text_env()
    env.parent.metadata["subagent_depth"] = 2  # max_depth is 2
    with pytest.raises(SubagentDepthExceeded):
        await env.dispatcher.spawn(
            env.parent, prompt="p", tools=[], extra_context={"subagent_depth": 0}
        )


async def test_context_cannot_impersonate_another_owner():
    env = _text_env()
    with pytest.raises(SubagentContextRejected, match="owner"):
        await env.dispatcher.spawn(
            env.parent, prompt="p", tools=[], extra_context={"owner": "u:admin"}
        )


async def test_context_cannot_widen_the_tool_allowlist():
    env = _text_env()
    with pytest.raises(SubagentContextRejected, match="allowed_tools"):
        await env.dispatcher.spawn(
            env.parent,
            prompt="p",
            tools=["safe.echo"],
            extra_context={"allowed_tools": ["admin.wipe"]},
        )


async def test_rejection_happens_before_the_child_runs():
    """No provider call, no child turn — the spawn is refused, not repaired."""
    env = _text_env()
    with pytest.raises(SubagentContextRejected):
        await env.dispatcher.spawn(
            env.parent, prompt="p", tools=[], extra_context={"capabilities": ["*"]}
        )
    assert env.progress_messages() == []
    assert env.invoked == []


async def test_benign_context_still_reaches_the_child():
    """The rejection is targeted: ordinary task context is merged as before."""
    env = _Env(child_script=[FakeProvider.tool_call("safe.echo", {}), FakeProvider.text("done")])
    await env.dispatcher.spawn(
        env.parent,
        prompt="p",
        tools=["safe.echo"],
        extra_context={"ticket": "T-1", "locale": "nb-NO"},
    )
    assert env.captured_metadata, "child never invoked the probe tool"
    metadata = env.captured_metadata[0]
    assert metadata["ticket"] == "T-1"
    assert metadata["locale"] == "nb-NO"


async def test_trusted_metadata_is_stamped_after_the_merge():
    env = _Env(child_script=[FakeProvider.tool_call("safe.echo", {}), FakeProvider.text("done")])
    await env.dispatcher.spawn(
        env.parent, prompt="p", tools=["safe.echo"], extra_context={"ticket": "T-1"}
    )
    metadata = env.captured_metadata[0]
    assert metadata["subagent_depth"] == 1
    assert metadata["owner"] == PARENT_OWNER
    assert "admin.wipe" not in metadata["allowed_tools"]


async def test_depth_guard_still_fires_at_the_cap():
    env = _text_env()
    env.parent.metadata["subagent_depth"] = 2  # max_depth is 2
    with pytest.raises(SubagentDepthExceeded):
        await env.dispatcher.spawn(env.parent, prompt="p", tools=[], extra_context={})


# ---- allowed_tools is enforced, not decorative ------------------------------


async def test_child_cannot_invoke_a_tool_outside_its_allowlist():
    """The core of the second finding: the child asked for the privileged tool
    it inherited, and it must not run."""
    env = _Env(
        child_script=[FakeProvider.tool_call("admin.wipe", {}), FakeProvider.text("blocked")]
    )
    summary = await env.dispatcher.spawn(
        env.parent, prompt="delete everything", tools=["safe.echo"], extra_context={}
    )
    assert env.invoked == [], "a tool outside the allowlist executed"
    assert summary == "blocked"


async def test_child_can_invoke_a_tool_inside_its_allowlist():
    env = _Env(child_script=[FakeProvider.tool_call("safe.echo", {}), FakeProvider.text("ok")])
    await env.dispatcher.spawn(env.parent, prompt="echo", tools=["safe.echo"], extra_context={})
    assert env.invoked == ["safe.echo"]


async def test_restricted_view_hides_unlisted_tools_from_the_provider():
    registry = ToolRegistry()
    registry.register_builtin(_spec("safe.echo"), _null_handler)
    registry.register_builtin(_spec("admin.wipe", risk=RiskLevel.DESTRUCTIVE), _null_handler)

    view = RestrictedToolRegistry(inner=registry, allowed={"safe.echo"})
    assert [s.name for s in view.list_specs()] == ["safe.echo"]
    assert view.spec_for("admin.wipe") is None
    assert view.spec_for("safe.echo") is not None


async def test_restricted_view_denies_a_direct_invoke():
    """Defence in depth: bypassing the listing API still hits a default deny."""
    registry = ToolRegistry()
    registry.register_builtin(_spec("admin.wipe", risk=RiskLevel.DESTRUCTIVE), _null_handler)
    view = RestrictedToolRegistry(inner=registry, allowed=set())

    result = await view.invoke(
        ToolCall(id="c1", name="admin.wipe", arguments={}), TurnContext.empty()
    )
    assert result.status == "denied"
    assert result.error is not None
    assert result.error.code == "not_authorized"


async def test_restricted_view_never_widens():
    registry = ToolRegistry()
    for name in ("a", "b", "c"):
        registry.register_builtin(_spec(name), _null_handler)

    child = RestrictedToolRegistry(inner=registry, allowed={"a", "b"})
    grandchild = child.restrict({"b", "c"})
    assert grandchild.allowed_tools == frozenset({"b"}), "restrict() must intersect, not replace"


async def test_restricted_view_does_not_shut_down_the_parents_mcp_clients():
    """A child's teardown must not break every later parent turn."""
    closed: list[str] = []

    class _Registry(ToolRegistry):
        async def shutdown(self) -> None:
            closed.append("inner")

    inner = _Registry()
    await RestrictedToolRegistry(inner=inner, allowed=set()).shutdown()
    assert closed == []


async def test_grandchild_cannot_regain_a_tool_the_child_lost():
    """A depth-2 subagent naming its grandparent's tool gets nothing."""
    env = _Env(
        child_script=[
            # depth 1: spawn a grandchild that asks for the privileged tool
            FakeProvider.tool_call(
                "kit.subagent.spawn",
                {"prompt": "wipe it", "tools": ["admin.wipe"], "context": {}},
            ),
            # depth 2: the grandchild tries to use it
            FakeProvider.tool_call("admin.wipe", {}),
            FakeProvider.text("grandchild blocked"),
            FakeProvider.text("child done"),
        ]
    )
    summary = await env.dispatcher.spawn(
        env.parent,
        prompt="delegate",
        tools=["kit.subagent.spawn", "safe.echo"],
        extra_context={},
    )
    # The child's last scripted response only gets consumed if the grandchild
    # actually ran and returned — otherwise this test would pass vacuously.
    assert summary == "child done"
    assert env.invoked == [], "grandchild reached a tool its parent did not have"


# ---- approval inside a subagent ---------------------------------------------


async def test_approval_inside_a_subagent_is_denied_and_surfaced():
    """The call is refused *visibly* — the old behaviour dropped it silently."""
    env = _Env(
        child_script=[FakeProvider.tool_call("admin.write", {}), FakeProvider.text("could not")]
    )
    summary = await env.dispatcher.spawn(
        env.parent, prompt="write it", tools=["admin.write"], extra_context={}
    )

    assert env.invoked == [], "a call needing approval executed without one"
    assert summary == "could not"
    messages = env.progress_messages()
    assert any("subagent denied admin.write" in m for m in messages), messages


async def test_a_subagent_writes_no_orphan_checkpoint():
    env = _Env(
        child_script=[FakeProvider.tool_call("admin.write", {}), FakeProvider.text("could not")]
    )
    await env.dispatcher.spawn(
        env.parent, prompt="write it", tools=["admin.write"], extra_context={}
    )
    assert env.checkpoints._data == {}, "subagent left a checkpoint nothing can resume"


async def test_a_child_left_suspended_raises_rather_than_returning_partial_text():
    """Fail-safe for a consumer gate the dispatcher did not wrap."""
    child = TurnContext.empty()
    child.metadata["pending_user_approvals"] = [{"id": "c1", "name": "admin.write"}]
    with pytest.raises(SubagentApprovalRequired, match="not permitted in nested context"):
        SubagentDispatcher._check_no_pending_approvals(child)


# ---- progress surfacing ------------------------------------------------------


async def test_subagent_text_reaches_the_parent_stream():
    """``loop.run()`` yields only lifecycle events, so filtering it alone could
    never surface anything. The merged stream must carry the child's text."""
    env = _text_env("the answer is 42")
    summary = await env.dispatcher.spawn(
        env.parent, prompt="what is it?", tools=[], extra_context={}
    )

    assert summary == "the answer is 42"
    messages = env.progress_messages()
    assert messages, "no subagent progress reached the parent"
    assert "42" in " ".join(messages)


async def test_subagent_tool_calls_are_announced_to_the_parent():
    env = _Env(child_script=[FakeProvider.tool_call("safe.echo", {}), FakeProvider.text("done")])
    await env.dispatcher.spawn(env.parent, prompt="echo", tools=["safe.echo"], extra_context={})
    messages = env.progress_messages()
    assert any("subagent calling safe.echo" in m for m in messages), messages


async def _null_handler(_args: dict[str, Any], _ctx: TurnContext) -> ToolResult:
    return ToolResult(call_id="", status="ok", content=[], error=None, duration_ms=0, cached=False)
