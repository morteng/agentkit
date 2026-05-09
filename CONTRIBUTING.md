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
