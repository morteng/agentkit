# Security Policy

## Reporting a vulnerability

Email **morteng@gmail.com** with a description and, if you have one, a
reproduction. Please don't open a public issue for a security report — file
an ordinary issue only after a fix has shipped, if at all.

You'll get an acknowledgement within **5 days**. agentkit is a solo-maintained
project, not a company with an SLA: for anything you'd rate low-severity,
expect a fix on a best-effort timeline rather than a committed date. For
anything that looks like an active data-exposure bug (see scope below),
say so in the subject line and it'll get priority.

## Scope

agentkit is a domain-blind agent runtime with a built-in PII/secret firewall
(`src/agentkit/pii/`) and a tool-authorization boundary (`src/agentkit/guards/`,
`src/agentkit/toolplane/`). The reports of most interest are ones that break
what those two subsystems promise:

- **Firewall bypass** — text or a tool-call argument that should be
  tokenized (`Action.TOKENIZE`) or blocked (`Action.NEVER_SEND`) but reaches
  a provider request unredacted. This includes gaps in the secret detector
  (`pii/secrets.py`: PEM blocks, vendor-prefixed keys, JWTs, high-entropy
  runs, secret-named fields) as well as the identity recognizers a consuming
  app supplies.
- **Rehydration/tokenization errors** — a token that resolves to the wrong
  span, leaks across sessions, or lets a scrubbed value be reconstructed by
  an unauthorized caller.
- **Approval-gate / tool-authorization bypass** — a tool invoked without
  going through `guards/approval.py`'s risk check, or invoke-time
  authorization in `toolplane/` that can be sidestepped (e.g. the class of
  bug that made `codeexec` escapable via `str.format` reaching module
  globals in 0.14–0.21).
- **MCP tool boundary escapes** — anything that lets a tool call reach code
  or data outside the schema it was registered with.
- **Wire-event leakage** — a firewall-scrubbed field that still shows up
  unredacted in an emitted event (`src/agentkit/events/`).

Out of scope: vulnerabilities in a provider SDK itself (report to
Anthropic/OpenAI), or in a consuming application's own tool implementations —
agentkit only controls what it wraps.

## Supported versions

Only the latest tagged release is supported. There is no LTS branch; fixes
land on `main` and ship in the next release.
