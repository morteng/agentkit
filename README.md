# agentkit

Domain-blind agent runtime. Provider abstraction (Anthropic + OpenRouter with caching), 11-phase loop, MCP tool boundary, pluggable guards, Pydantic event stream.

**Private project — proprietary license. See `LICENSE`.**

## Installation

```bash
uv add git+https://github.com/morteng/agentkit@v0.1.0
```

With optional extras:

```bash
uv add 'agentkit[fastapi] @ git+https://github.com/morteng/agentkit@v0.1.0'
```

## Quickstart

See `examples/minimal/`.

## Architecture

See `docs/architecture.md` for the full design. Spec lives in the consuming project (Ampæra) at `docs/superpowers/specs/2026-05-03-agentkit-library-design.md`.
