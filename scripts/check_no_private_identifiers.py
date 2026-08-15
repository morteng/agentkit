#!/usr/bin/env python3
"""Fail if a private-system identifier appears anywhere in the tree.

Why this exists: agentkit was public from its first commit while carrying, at
various times, a 392-line audit of three private downstream repos, source
comments naming a private consumer and one of its security defects, a private
product's whole tool namespace used as test fixtures, and test files using a
real household domain and a family member's name. None of it was a credential
— gitleaks was clean across every commit — so a secret scanner could not have
caught any of it. This is the check that would have.

The failure mode it guards is a reflex, not an accident: the person writing a
comment knows exactly which downstream system taught them the lesson, and
naming it feels like good sourcing. In a private repo it is. Here it publishes
someone else's architecture.

The list of names lives OUTSIDE this repo, and that is the whole design. The
first version of this file hard-coded it, so the guard published two children's
first names, a private domain and three private product names — in the one file
guaranteed to be read by anyone curious about what the guard is for. A
blocklist of secret words is itself the secret. The words are therefore
supplied at runtime and never committed:

    AGENTKIT_PRIVATE_IDENTIFIERS       comma-separated, or
    AGENTKIT_PRIVATE_IDENTIFIERS_FILE  path to a newline-separated file
                                       (default: ~/.config/agentkit/private-identifiers)

With no list configured the check says it is unconfigured and exits 0. That is
deliberate: an outside contributor cannot know these names and must not be
blocked by a list they cannot see. The guard's real posts are the maintainer's
pre-commit hook and a CI job holding the list as a secret — both places where
the names are already known.

Scope note: the maintainer's own name (LICENSE, NOTICE, pyproject author) and
the contact address in SECURITY.md / CODE_OF_CONDUCT.md are deliberate. The
author's name is stripped from each line before matching, so a surname that is
also part of a forbidden domain does not trip on its own.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

#: Where the blocklist comes from when the environment does not name a file.
DEFAULT_LIST_PATH = Path.home() / ".config" / "agentkit" / "private-identifiers"

#: Paths where a hit is expected and legitimate.
ALLOWED_PATHS = {
    # The maintainer is named on purpose in these four.
    "LICENSE",
    "NOTICE.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    # This file describes the guard. It no longer contains the list.
    "scripts/check_no_private_identifiers.py",
}
# Nothing else is allowlisted, and the empty space is the point. The two
# entries that used to be here — a nightly job that checked a private repo out
# by name, and the section of CONTRIBUTING documenting it — were allowlisted
# because they were functional, which is a reason to move a job, not a reason
# to publish a name. The job now lives in the consumer's own repository, where
# naming this one is free. An allowlist entry should always feel like the
# expensive option.

#: The author's own name, which is supposed to be here.
AUTHOR_RE = re.compile(r"morten gulden", re.IGNORECASE)


def load_forbidden() -> list[str]:
    """Lowercased substrings that must not appear in tracked files.

    Blank lines and ``#`` comments are ignored, so the list file can explain
    itself to whoever maintains it.
    """
    inline = os.environ.get("AGENTKIT_PRIVATE_IDENTIFIERS")
    if inline:
        raw = inline.replace(",", "\n")
    else:
        path = Path(os.environ.get("AGENTKIT_PRIVATE_IDENTIFIERS_FILE") or DEFAULT_LIST_PATH)
        if not path.is_file():
            return []
        raw = path.read_text(encoding="utf-8")
    out: list[str] = []
    for line in raw.splitlines():
        word = line.split("#", 1)[0].strip().lower()
        if word:
            out.append(word)
    return out


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    forbidden = load_forbidden()
    files = tracked_files()

    if not forbidden:
        # Not an error, and not silent either. A guard that passes without
        # saying it did nothing is worse than no guard: the green tick reads as
        # "checked and clean".
        print(
            "skipped — no blocklist configured, so nothing was checked.\n"
            f"Put one name per line in {DEFAULT_LIST_PATH}, or set\n"
            "AGENTKIT_PRIVATE_IDENTIFIERS / AGENTKIT_PRIVATE_IDENTIFIERS_FILE.\n"
            "Contributors outside the project are expected to see this."
        )
        return 0

    hits: list[tuple[str, int, str, str]] = []
    for path in files:
        if path in ALLOWED_PATHS:
            continue
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing to match
        for lineno, line in enumerate(lines, 1):
            # Strip the author's name first so a surname that also appears
            # inside a forbidden domain never trips on its own.
            probe = AUTHOR_RE.sub("", line).lower()
            for needle in forbidden:
                if needle in probe:
                    hits.append((path, lineno, needle, line.rstrip()[:110]))

    if not hits:
        print(f"ok — no private identifiers in {len(files)} tracked files")
        return 0

    print("Private identifiers found in tracked files:\n", file=sys.stderr)
    for path, lineno, needle, text in hits:
        print(f"  {path}:{lineno}  [{needle}]  {text}", file=sys.stderr)
    print(
        f"\n{len(hits)} hit(s). This repo is publishable; those names are not.\n"
        "Keep the lesson, drop the identity: say 'a downstream consumer', not\n"
        "which one. If a hit is legitimate, add its path to ALLOWED_PATHS in\n"
        "scripts/check_no_private_identifiers.py and say why in the commit message.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
