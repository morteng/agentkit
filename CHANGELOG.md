# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0.0 onward. Pre-1.0 minor versions may include breaking changes.

## [0.10.0] - 2026-05-23

### Added
- `AgentConfig.continuation_evaluator` hook — a consumer-supplied async callable
  that, after every terminal envelope, decides whether the session's goal is
  met. Mirrors the `provider_selector` shape (typed `Any` to keep
  `AgentConfig` import-light). v0.10.0 ships `trigger="every_turn"` dispatch;
  the `self_declared` trigger is reserved in the literal for v0.11.0 and the
  loop raises if a v0.10.0 caller wires it up.
- `AgentSession.goal: GoalState | None` plus `set_goal(condition, *, state_id=None, resume_from=None)`
  and `clear_goal()` methods. `state_id` is opaque to agentkit — consumers
  persisting goal state externally (e.g. a Pikkolo Task row) use it to
  correlate. `resume_from` lets a consumer rehydrate counters and
  `last_reason` from external persistence; without it, counters start at
  zero, matching Claude Code's resume semantics.
- Four new public events: `GoalSet`, `GoalEvaluated`, `GoalAchieved`,
  `GoalAbandoned`. Each carries the goal condition and the consumer's opaque
  `state_id`. Wire-contract snapshots pinned under `tests/wire/snapshots/`.
- `agentkit.continuation` module: `ContinuationEvaluator` (runtime-checkable
  Protocol), `ContinuationRequest`, `ContinuationDecision`, `GoalState`,
  `TriggerMode` literal.

### Changed
- `AgentSession.run()` no longer ends unconditionally on the first
  `TurnEnded`. When a goal is active and a `continuation_evaluator` is
  configured, the same `run()` invocation streams multiple turn cycles —
  each separated by a `GoalEvaluated` event — until the evaluator returns
  `met=True` (→ `GoalAchieved`) or `clear_goal()` is called (→
  `GoalAbandoned(cause="cleared")`). Sessions without a goal behave
  identically to v0.9.0.
- Evaluator reasons matching the `GoalAbandoned.cause` literal
  (`"budget_exceeded"`, `"max_turns"`, `"max_iterations"`) short-circuit to
  `GoalAbandoned` instead of `GoalAchieved`, so consumer evaluators can end
  the loop on budget or turn caps without inventing a parallel signal.
- Rejected (`met=False`) decisions append a system-role
  `[goal continuation: <reason>]` message to history, annotated with
  `metadata.annotations["goal_continuation"] = True` so consumer event
  translators can filter it from the editor-facing transcript.

## [0.8.0] - 2026-05-22

### Changed
- A turn that ends **without** a `finalize_response` call is no longer silently accepted as "the conversation naturally ended". When a finalize validator is configured, `handle_finalize_check` now re-prompts the model once (bounded by `LoopConfig.max_missing_finalize_reprompts`, default 1) to emit a real envelope, then lets the turn end if the model still won't finalize. This fixes turns that stop mid-thought — typically by asking the user a question — settling with no envelope: the model now gets an explicit chance to classify them (e.g. `intent_kind="clarify"`). Consumers with no finalize validator are unaffected (pass-through).
- `FINALIZE_RESPONSE_DESCRIPTION` reworded: "call at the END of EVERY turn, including turns where you stop to ask the user a question", and the `clarify` bullet broadened to cover "asking a question / offering a choice / needing a decision", including turns that already did some work.

### Fixed
- Finalize-retry corrections now actually reach the model. `finalize_correction` was stashed in `ctx.metadata` but never surfaced — `MessageBuilder` reads `ctx.history`, so the rejected-finalize retry re-ran blind. The correction (and the new missing-finalize re-prompt) is now appended to `ctx.history` as a user-role message before re-streaming.
- Injected correction messages are tagged `metadata.annotations[INJECTED_CORRECTION_ANNOTATION]` so `_summaries_since_last_user_turn` (Rule 9 scoping) does not mistake a finalize re-prompt for a fresh human prompt — which would otherwise drop the turn's reads and false-fail `answer_evidence="tool_results"`.

### Added
- `LoopConfig.max_missing_finalize_reprompts: int = 1` — how many times a missing `finalize_response` is re-prompted before the turn is allowed to end.
- `INJECTED_CORRECTION_ANNOTATION` constant (`agentkit._messages`) — the `Message.metadata.annotations` key marking a loop-injected correction. Consumer code that infers turn boundaries from the most recent USER message should skip messages carrying it.

## [0.7.2] - 2026-05-21

### Fixed
- Unknown tool names no longer silently kill a turn. When the model calls an
  unregistered tool, `handle_tool_phase` now files it under a new
  `unknown_tool_calls` bucket and routes to `TOOL_EXECUTING` (previously it
  fell straight to `TOOL_RESULTS`, skipping result construction entirely — the
  model got no ToolResult, no error, just silence, and could never
  self-correct). `handle_tool_executing` builds a `status="error"` ToolResult
  naming the bad tool so the model can retry with a registered name.
- `handle_tool_results` counts unknown-tool errors toward the consecutive
  error abort (F20), so a model that keeps hallucinating the same tool name
  trips the abort instead of looping.

### Changed
- Defense-in-depth: `ToolRegistry.invoke` and `ToolDispatcher` no longer raise
  on an unknown tool name. `invoke` returns a `status="error"` ToolResult;
  `ToolDispatcher._safe_for_parallel` treats a spec-less call as
  not-parallel-safe instead of raising. A raised exception there bubbles to
  the orchestrator and ends the turn with no result for the model.

## [0.7.0] - 2026-05-15

### Added
- `Envelope.answer_evidence: Literal["tool_results", "context", "general_knowledge"] | None` field. Required when `intent_kind="answer"` (enforced by `validate_envelope`), ignored otherwise. Lets the model self-attest what evidence its answer rests on so the structural validator can check claim ↔ tool-log consistency.
- Validator Rule 8 (`answer_evidence_required`): rejects `intent_kind="answer"` envelopes missing `answer_evidence`.
- Validator Rule 9 (`answer_evidence_consistent`): rejects `answer_evidence="tool_results"` claims when the current turn has no successful read tool call. Uses a new `_summaries_since_last_user_turn` helper to scope reads to this turn only.
- `validate_envelope` accepts an optional `turn_summaries=` kwarg; when omitted, falls back to the full `tool_calls` list (backwards-compatible for existing callers).

### Fixed
- `recall_*` tools (e.g. `recall_memories`) now classify as reads in `_DEFAULT_READ_PREFIXES`. Previously misclassified as writes by the conservative default, which would have invalidated Rule 9 for memory-recall turns.

## [0.1.0] — 2026-05-04

### Added
- Provider abstraction: `Provider` protocol, `AnthropicProvider`, `OpenRouterProvider`
  with model-quirks-driven prompt caching.
- 11-phase agent loop with explicit transition table and `PhaseChanged` events.
- Tool registry with `kit.*` built-ins (finalize, current_time, memory, approval,
  subagent, note) plus MCP transports: `InProcessMCPClient` and `StdioMCPClient`.
- Storage protocols: `SessionStore`, `MemoryStore`, `CheckpointStore` with
  Redis-backed default implementations and in-memory fakes.
- Guards: `RiskBasedApprovalGate`, `DefaultIntentGate`, `RuleBasedFinalizeValidator`,
  `RegexSuccessClaimGuard`.
- 18 Pydantic event types with discriminated union.
- Optional FastAPI WebSocket bridge under `agentkit[fastapi]`.
- Multi-replica fan-out via `RedisEventBus`.
- `AgentSession` high-level entry point with `resume_with_approval` for
  suspend/resume flows.
- Subagent dispatch with isolated child contexts.
- Examples: `minimal/` and `with_mcp_tools/`.
- MkDocs documentation site.
