# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v1.0.0 onward. Pre-1.0 minor versions may include breaking changes.

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
