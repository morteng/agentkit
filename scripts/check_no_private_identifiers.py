#!/usr/bin/env python3
"""Fail if a private-system identifier appears anywhere in the tree, or in history.

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

Two properties this file learned the hard way, on 2026-08-16:

1. THE REPORT MUST NOT ECHO WHAT IT FOUND. The first version printed the
   matching needle and the offending source line. On a public repo that output
   lands in a public Actions log, and a comma-joined secret is masked by
   GitHub only as one whole string — never term by term. So the guard would
   have published every name it exists to protect, precisely when it fired.
   Findings are now reported as a location plus an opaque term index. Whoever
   holds the list can resolve the index in one grep; nobody else learns
   anything.

2. CHECKING THE WORKING TREE IS NOT CHECKING THE REPO. This script used to
   scan `git ls-files` and nothing else, so it went green on a checkout whose
   history held the names in 183 of 185 commits and all 33 tags. Removing a
   name in an ordinary commit does nothing to a public repo — the old blob is
   still served. `--history` is the mode that answers the question the guard's
   name implies, and CI runs it on main.

3. THE NAMES OF THINGS ARE PUBLISHED TOO. A tracked file's path shows in the
   forge's file tree of any branch, and a ref name shows in the branch list and
   in every CI log that prints a checkout or a status. The two content scans
   above never look at either surface, which is exactly how a filename survived:
   its content was scrubbed and pushed by someone who had every reason to
   believe they were done — the passing guard was their evidence. Measured on
   2026-08-27 against the real blocklist: one distinct bad filename across all
   remote refs, and one ref carrying a term outright. The guard now reads path
   names and ref names alongside everything else.

   A name finding needs stricter masking than a content finding, because the
   offending string IS what would normally be printed as the location label.
   So these findings print an opaque sha256 prefix of the name plus term
   indices; --verbose unmasks locally only (never set it in CI), so whoever
   holds the list can resolve labels to files. Severity follows control:
   tracked paths and local branches/tags block — renaming takes seconds and
   they have not left the machine yet. refs/remotes/** warns but does not fail,
   because that namespace is already public and fixing it is the maintainer's
   call, not CI's; a detector wired to turn those red re-creates the permanent
   red-dot problem this file exists to end. The warnings vanish on their own as
   the backlog gets cleaned. There is deliberately no ALLOWED_REFS, for the
   same reason ALLOWED_PATHS's empty space matters: nothing functional can
   require a private identifier inside a ref name, so allowlisting stays the
   expensive option.

   And what sits INSIDE published trees: blobs are pathless, so history-mode
   content findings can say that a term exists but never where it lived. In
   --history mode the guard therefore also walks every ref's tree via ls-tree,
   reporting offending path NAMES as warnings. Severity is warn-only because
   this surface is inherited public state (45 of 59 measured on 2026-08-27
   carry one filename whose content was already scrubbed; removal is the
   maintainer's call); findings clear as cleanup lands instead of redden ing
   main forever. This walk carries its own hard rule learned from that
   measurement, twice miscounted by an instrument that swallowed resolver and
   ls-tree failures as "clean": every ref is rev-parsed BEFORE use, a failed
   resolution or tree read produces a WARNING saying NOT counted as clean,
   stderr is captured rather than discarded, and attempted/resolved/path counts
   are printed so absence is always distinguishable from coverage.

Scope note: the maintainer's own name (LICENSE, NOTICE, pyproject author) and
the contact address in SECURITY.md / CODE_OF_CONDUCT.md are deliberate. The
author's name is stripped from each line before matching, so a surname that is
also part of a forbidden domain does not trip on its own.
"""

from __future__ import annotations

import argparse
import hashlib
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

# The commit a forge fabricates to test a pull request — its message is the
# literal "Merge <sha> into <sha>" and nothing else. It is not in this
# repository, it is deleted when the PR closes, and no human wrote a word of
# it, so the only thing it can ever contribute is a false positive: matching is
# plain substring, and a blocklist term made of [0-9a-f] will eventually appear
# inside eighty random hex characters. That is not a hypothetical — it is how
# this check first fired, on a PR whose own tree and commits were clean, and it
# reported a commit the author could not even fetch. Skipping it loses no
# coverage: both its parents are scanned on their own.
SYNTHETIC_MERGE_RE = re.compile(r"Merge [0-9a-f]{40} into [0-9a-f]{40}")

#: Fields in a ``git cat-file --batch`` header line: "<sha> <type> <size>".
#: Anything else is a "<sha> missing" style response and is skipped.
BATCH_HEADER_FIELDS = 3

#: Findings printed before the report truncates. A history hit is usually
#: hundreds of blobs saying the same thing; the count below is the real signal.
MAX_REPORTED = 200

#: Local-only unmasking for the name surfaces (see principle 3). CI must never
#: set this: verbose output contains the raw offending paths and ref names.
VERBOSE_ENV = "AGENTKIT_GUARD_VERBOSE"


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


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def scan_text(text: str, forbidden: list[str]) -> list[tuple[int, int]]:
    """(1-based line number, blocklist index) for every match in ``text``."""
    found: list[tuple[int, int]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        probe = _probe(line)
        for idx, needle in enumerate(forbidden):
            if needle in probe:
                found.append((lineno, idx))
    return found


def _probe(line: str) -> str:
    """The author's name stripped, then lowercased — shared by every surface."""
    return AUTHOR_RE.sub("", line).lower()


def _hit_indices(name: str, forbidden: list[str]) -> set[int]:
    """Blocklist indices whose term appears inside a *name* (path or ref)."""
    probe = _probe(name)
    return {i for i, needle in enumerate(forbidden) if needle in probe}


def _masked_label(verbose: bool, name: str) -> str:
    """The name itself when unmasking locally; otherwise a stable sha256 prefix.

    For these surfaces the offending string is also the natural location label,
    so printing it verbatim would publish exactly what the guard exists to
    protect. A short hash keeps repeat local runs correlatable at zero cost to
    anonymity.
    """
    if verbose:
        return name
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def check_worktree(forbidden: list[str], verbose: bool = False) -> tuple[list[str], int]:
    """Scan the files git currently tracks. Fast; what pre-commit wants."""
    files = [line for line in _git("ls-files").splitlines() if line]
    hits: list[str] = []
    for path in files:
        # Name first, BEFORE the allowlist skip: ALLOWED_PATHS permits deliberate
        # prose inside listed files, never a forbidden token embedded in a path.
        seen_names = _hit_indices(path, forbidden)
        if seen_names:
            terms = ",".join(f"#{i}" for i in sorted(seen_names))
            hits.append(f"path {_masked_label(verbose, path)}  [terms {terms}]")
            # Scanning this file's content too would echo the offending path
            # through the `path:lineno` label below. The name finding already
            # carries everything this report may say about the file.
            continue
        if path in ALLOWED_PATHS:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing to match
        for lineno, idx in scan_text(text, forbidden):
            hits.append(f"{path}:{lineno}  [term #{idx}]")
    return hits, len(files)


def check_ref_names(
    forbidden: list[str], verbose: bool = False
) -> tuple[list[str], list[str], int]:
    """Scan ref names: the forge publishes them in branch/tag lists and logs.

    Returns (blocking_hits, warnings, refs_scanned). Blocking covers the
    namespaces the author still controls (local heads and tags); anything under
    refs/remotes/** has already left every machine behind it, so it warns
    without failing CI — see principle 3 for why this split exists.
    """
    refs = [r for r in _git("for-each-ref", "--format=%(refname)").splitlines() if r]
    hits: list[str] = []
    warnings: list[str] = []
    for refname in refs:
        short = refname
        for prefix in ("refs/heads/", "refs/remotes/", "refs/tags/"):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        seen = _hit_indices(short, forbidden)
        if not seen:
            continue
        terms = ",".join(f"#{i}" for i in sorted(seen))
        where = _masked_label(verbose, refname)
        if refname.startswith(("refs/heads/", "refs/tags/")):
            hits.append(f"ref name {where}  [terms {terms}]")
        else:
            warnings.append(
                f"WARNING ref name {where} (remote namespace; maintainer's call)  [terms {terms}]"
            )
    return hits, warnings, len(refs)


def _resolve_commit(refname: str) -> str | None:
    """Rev-parse a ref to the commit it names; None means unreadable."""
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{refname}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,  # non-zero IS the signal here; nothing is swallowed
    )
    if proc.returncode != 0:
        # Deliberately NOT swallowed-as-empty: the caller warns + counts this.
        return None
    return proc.stdout.strip()


def _tree_paths(rev: str) -> tuple[bool, list[str]]:
    """Every path in a commit's tree. False => ls-tree failed loudly-not-cleanly.

    Returncode decides; stderr is captured for humans if needed and NEVER
    discarded into /dev/null. A false is reported upstream as unreadable —
    folding it to an empty success would make an unresolvable ref read exactly
    like a clean scan, which is how the real exposure got undercounted twice.
    """
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev],
        capture_output=True,
        text=True,
        check=False,  # non-zero IS the signal here; nothing is swallowed
    )
    if proc.returncode != 0:
        return False, []
    return True, [p for p in proc.stdout.splitlines() if p]


def _print_warnings(warnings: list[str]) -> None:
    """Opaque warning lines, truncated like findings; this output is public."""
    for warning in warnings[:MAX_REPORTED]:
        print(f"  {warning}", file=sys.stderr)
    if len(warnings) > MAX_REPORTED:
        print(f"  ... and {len(warnings) - MAX_REPORTED} more", file=sys.stderr)


def check_published_trees(
    forbidden: list[str], verbose: bool = False
) -> tuple[list[str], int, int, int, int]:
    """--history companion: scan EVERY ref's file tree for offending path names.

    Returns (warnings, refs_attempted, refs_resolved, refs_unreadable,
    paths_examined). Offending paths are aggregated per distinct name so a
    leak spread across many refs prints once with its reach count instead of
    hundreds of lines saying the same thing; verbose detail lines list each
    contributing ref. Everything here warns — see principle 3 — but coverage
    is always explicit: {resolved}/{attempted} trees read, path total shown,
    unreadable refs flagged individually, never folded into 'clean'.
    """
    refs = [r for r in _git("for-each-ref", "--format=%(refname)").splitlines() if r]
    attempted = len(refs)
    resolved = unreadable = 0
    paths_examined = 0
    warnings: list[str] = []
    agg: dict[str, dict] = {}
    for refname in refs:
        rev = _resolve_commit(refname)
        ok, paths = (False, []) if rev is None else _tree_paths(rev)
        if not ok:
            unreadable += 1
            warnings.append(
                f"WARNING ref unreadable {_masked_label(verbose, refname)} "
                f"(resolver or ls-tree failed; NOT counted as clean)"
            )
            continue
        resolved += 1
        paths_examined += len(paths)
        for path in paths:
            idxs = _hit_indices(path, forbidden)
            if not idxs:
                continue
            entry = agg.setdefault(path.lower(), {"refs": set(), "indices": set(), "count": 0})
            entry["indices"].update(idxs)
            entry["refs"].add(_masked_label(verbose, refname))
            entry["count"] += 1
    for key, entry in sorted(agg.items()):
        terms = ",".join(f"#{i}" for i in sorted(entry["indices"]))
        where = _masked_label(verbose, key)
        line = (
            f"WARNING path {where}  [terms {terms}] in {entry['count']} of {resolved} readable refs"
        )
        if verbose:
            line += ": " + ", ".join(sorted(entry["refs"]))
        warnings.append(line)
    return warnings, attempted, resolved, unreadable, paths_examined


def _all_blobs() -> list[str]:
    """Every blob in the object database, reachable or not.

    ``--batch-all-objects`` rather than ``rev-list`` on purpose: after a history
    rewrite the interesting objects are exactly the ones no ref points at any
    more, and those are the ones a forge may still serve by SHA.
    """
    out = _git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")
    return [line.split()[0] for line in out.splitlines() if line.endswith(" blob")]


def _read_blobs(shas: list[str]) -> list[tuple[str, bytes]]:
    """Stream blob contents via one ``git cat-file --batch``."""
    if not shas:
        return []
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(shas) + "\n").encode(),
        capture_output=True,
        check=True,
    )
    data, pos, out = proc.stdout, 0, []
    while pos < len(data):
        nl = data.find(b"\n", pos)
        if nl == -1:
            break
        header = data[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) != BATCH_HEADER_FIELDS:  # "<sha> missing" and friends
            continue
        sha, size = header[0], int(header[2])
        out.append((sha, data[pos : pos + size]))
        pos += size + 1  # trailing newline git appends
    return out


def _unmask_requested(args) -> bool:
    """--verbose or AGENTKIT_GUARD_VERBOSE; locals only, CI must never set it."""
    return args.verbose or os.environ.get(VERBOSE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def check_history(forbidden: list[str]) -> tuple[list[str], int, bool]:
    """Scan every blob and every commit message in the object database.

    Deliberately blob-oriented rather than commit-oriented: a blob shared by
    two hundred commits is read once, and a blob orphaned by a rewrite is still
    read. Blobs carry no path, which is why findings name the object — and
    ``git log --all --find-object=<sha>`` turns that back into a path locally,
    for whoever is allowed to know.

    Returns (hits, blobs_scanned, message_sweep_skipped). A single unreadable
    ref makes `git log --all` refuse outright (exit 128); that must degrade the
    sweep loudly — flagged True, never swallowed into a fake-clean exit code.
    """
    hits: list[str] = []
    blobs = _all_blobs()
    for sha, raw in _read_blobs(blobs):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        seen: set[int] = set()
        for _lineno, idx in scan_text(text, forbidden):
            seen.add(idx)
        if seen:
            terms = ",".join(f"#{i}" for i in sorted(seen))
            hits.append(f"blob {sha}  [terms {terms}]")

    # %x1e (ASCII record separator) delimits commits, not %x00. git log ends
    # every record with a newline of its own, so a NUL-NUL delimiter never
    # appears: the whole log arrived as ONE record, every message was scanned
    # as one blob of text, and the SHA printed beside a finding was simply the
    # newest commit in the repository. The check still fired — it just named a
    # commit that had nothing to do with it, and on a pull request that is the
    # forge's fabricated merge, which the author cannot fetch, let alone
    # rewrite. Found 2026-08-27 while chasing exactly that report.
    try:
        messages = _git("log", "--all", "--format=%H%x00%B%x1e")
        message_sweep_skipped = False
    except subprocess.CalledProcessError:
        # One unreadable ref must not zero out history coverage silently.
        messages = ""
        message_sweep_skipped = True

    for record in messages.split("\x1e"):
        if "\x00" not in record:
            continue
        commit, _, body = record.strip().partition("\x00")
        if SYNTHETIC_MERGE_RE.fullmatch(body.strip()):
            continue
        seen = {idx for _ln, idx in scan_text(body, forbidden)}
        if seen:
            terms = ",".join(f"#{i}" for i in sorted(seen))
            hits.append(f"commit {commit[:12]} (message)  [terms {terms}]")
    return hits, len(blobs), message_sweep_skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--history",
        action="store_true",
        help="scan every blob and commit message in the object database, "
        "not just the files currently tracked",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="unmask name-surface findings locally (paths, ref names). Never "
        f"set this in CI; env equivalent: {VERBOSE_ENV}=1",
    )
    args = parser.parse_args()

    forbidden = load_forbidden()

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

    verbose = _unmask_requested(args)

    warnings: list[str] = []

    if args.history:
        hits, blobs, msgs_skipped = check_history(forbidden)
        scope = f"{blobs} blobs in history"
        if msgs_skipped:
            warnings.append(
                "WARNING commit-message sweep skipped (git log --all hit an "
                "unreadable ref); NOT counted as covered"
            )
        tree_warnings, attempted, resolved, unreadable, n_paths = check_published_trees(
            forbidden, verbose=verbose
        )
        warnings += tree_warnings
        scope += f", {resolved}/{attempted} ref trees read ({n_paths} paths)"
        if unreadable:
            scope += f", {unreadable} unreadable"
    else:
        hits, files = check_worktree(forbidden, verbose=verbose)
        scope = f"{files} tracked files"

    ref_hits, ref_name_warnings, n_refs = check_ref_names(forbidden, verbose=verbose)
    hits = hits + ref_hits
    scope = f"{scope}, {n_refs} ref names"
    warnings += ref_name_warnings

    if not hits:
        if warnings:
            print(
                f"ok — no blocking private identifiers in {scope}; "
                f"{len(warnings)} inherited public-state warning(s) below"
            )
        else:
            print(f"ok — no private identifiers in {scope}")
        _print_warnings(warnings)
        return 0

    # Locations and term indices only. Never the term, never the matching line:
    # this output is public whenever CI is.
    print(f"Private identifiers found ({scope}):\n", file=sys.stderr)
    for hit in hits[:MAX_REPORTED]:
        print(f"  {hit}", file=sys.stderr)
    if len(hits) > MAX_REPORTED:
        print(f"  ... and {len(hits) - MAX_REPORTED} more", file=sys.stderr)
    _print_warnings(warnings)
    print(
        f"\n{len(hits)} hit(s). This repo is publishable; those names are not.\n"
        "Term indices are positions in your blocklist file — resolve them locally,\n"
        'with `sed -n "$((N+1))p"` against it. They are printed instead of the words\n'
        "because this output is public whenever CI is.\n"
        "For a blob, `git log --all --find-object=<sha>` names the commits and path.\n"
        "`path <hash>` / `ref name <hash>` findings are sha256 prefixes of the\n"
        "offending name (the name itself would be the leak). Rerun locally with\n"
        "--verbose to unmask them; CI must never do that. Remote-namespace ref\n"
        "warnings are the inherited backlog: fixing them is the maintainer's call,\n"
        "not this check's. Tree-path warnings aggregate per distinct name with\n"
        "their reach count; `ref trees read` coverage is printed with the scope\n"
        "so nothing can pass silently unreadable.\n"
        "Keep the lesson, drop the identity: say 'a downstream consumer', not\n"
        "which one. If a hit is legitimate, add its path to ALLOWED_PATHS in\n"
        "scripts/check_no_private_identifiers.py and say why in the commit message.\n"
        "A history hit cannot be fixed by a new commit — it needs the history\n"
        "rewritten (git filter-repo --replace-text) and every consumer re-pinned.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
