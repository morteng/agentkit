# Third-Party Notices

agentkit is Copyright 2026 Morten Gulden, licensed under Apache-2.0 (see `LICENSE`).

It depends on the following open-source packages. All are permissively licensed
(MIT / Apache-2.0 / BSD), with two MPL-2.0 exceptions noted at the end.

- `anthropic` — MIT
- `openai` — Apache-2.0
- `mcp` — MIT
- `pydantic`, `pydantic-settings` — MIT
- `redis-py` — MIT
- `httpx` — BSD
- `python-ulid` — MIT
- `structlog` — Apache-2.0 / MIT
- `json_repair` — MIT

Two MPL-2.0 packages appear in the resolved tree: `certifi` (a CA-certificate
bundle, pulled in transitively by `httpx`) and `hypothesis` (dev-only). MPL-2.0
is file-level copyleft — it obliges you only if you modify and redistribute
those packages' own files, which agentkit does not do — so neither constrains
this project's licence.

Dev-only dependencies (pytest, ruff, pyright, etc.) are not listed here. See `pyproject.toml` for the complete dev set.
