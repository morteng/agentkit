# Contributing to agentkit

This file collects rules and patterns for working in this repo. See
`README.md` for general project context.

## Adding fields to provider request payloads

Any PR that adds or changes a field in a provider's request payload (e.g.
`extra_body.reasoning`, `thinking`, new top-level kwargs) MUST add or update a
test in `tests/contract/`. The test constructs the real provider SDK client
(`openai.AsyncOpenAI`, `anthropic.AsyncAnthropic`) with a stubbed httpx
transport and exercises the new payload shape end-to-end.

This rule exists because mock-based provider tests miss kwarg-shape errors
that the real SDK would catch. v0.2.0 shipped `reasoning` as a top-level
kwarg, passed all mock tests, and crashed in production against the real
openai SDK. The contract harness in `tests/contract/` exists to prevent
this class of regression.

See `tests/contract/test_openrouter_real_client.py` for the canonical
pattern.

## Wire-event snapshots

`tests/wire/snapshots/*.json` pins the on-the-wire shape of every event
emitted by the loop. Any change to event field names, types, or default
values shows up as a JSON diff in PR review — that diff IS the signal to
consumers (Pikkolo's translator, future consumers' frontends) that the
wire contract changed.

To accept a deliberate change:

```bash
WIRE_SNAPSHOT_UPDATE=1 uv run pytest tests/wire/test_event_snapshots.py
```

Then commit the updated `.json` files alongside the source change. The PR
description should explain why the wire shape changed and what consumers
need to do.

**Adding a new event class:** Add a fixture entry in
`tests/wire/test_event_snapshots.py:EVENT_FIXTURES`. The meta-test
`test_no_event_class_lacks_snapshot` will fail otherwise.

## Nightly cross-repo eval

`.github/workflows/nightly-pikkolo.yml` runs once daily at 03:00 UTC. It
checks out Pikkolo at `main`, force-installs agentkit at the current `main`
SHA, and runs Pikkolo's full test suite plus the smoke-mode eval harness
(5 canonical prompts × 3 replicates ≈ 3 minutes, ~$1 OpenRouter spend).

**On a single failure:** the workflow logs red but does not page. This
absorbs single-run flakiness in the eval scoring.

**On two consecutive failures:** the workflow opens or comments on a
GitHub Issue tagged `nightly-regression`, listing both failing run URLs.
Investigate before the next agentkit tag.

**Required repo secrets:** `PIKKOLO_REPO_TOKEN` (read-only token for
morteng/pikkolo-cms-mvp), `OPENROUTER_API_KEY` (budget-capped via
OpenRouter dashboard, ~$1/night).

**Manual trigger:** Actions → nightly-pikkolo → Run workflow.
