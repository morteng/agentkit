# agentkit

Domain-blind Python agent runtime. Provider abstraction (Anthropic + OpenRouter
with prompt caching), 11-phase loop, MCP tool boundary, pluggable guards,
Pydantic event stream, optional FastAPI WebSocket bridge.

**Private project — proprietary license.**

## Quickstart

See [examples/minimal](https://github.com/morteng/agentkit/tree/main/examples/minimal).

## Install

```bash
uv add git+https://github.com/morteng/agentkit@v0.1.0
```

With FastAPI extras:

```bash
uv add 'agentkit[fastapi] @ git+https://github.com/morteng/agentkit@v0.1.0'
```
