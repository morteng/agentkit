"""The identifier guard, driven through the seam it is actually used at.

The interesting properties are not "does substring matching work". They are:

* a name deleted in an ordinary commit is still in the repository, and the
  guard must say so — this is the bug that let 183 of 185 commits carry the
  names while CI stayed green;
* the guard's own failure output must not contain the names, because on a
  public repo that output is a public Actions log, and a comma-joined secret
  is masked by the forge as one whole string, never term by term.

Both are tested by running the script over a real throwaway git repository
rather than by calling the matcher directly. A matcher with full coverage
proves nothing about a caller that never reaches it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_private_identifiers.py"

# Not a real private identifier — an invented token that behaves like one.
SECRET = "zorbulon"


def _load_guard():
    spec = importlib.util.spec_from_file_location("_guard_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "notes.py").write_text(f"# built for {SECRET}\nX = 1\n", encoding="utf-8")
    _git(r, "add", "notes.py")
    _git(r, "commit", "-qm", "add notes")
    return r


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTKIT_PRIVATE_IDENTIFIERS", SECRET)
    monkeypatch.delenv("AGENTKIT_PRIVATE_IDENTIFIERS_FILE", raising=False)


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,  # a non-zero exit is the thing under test
    )


def test_a_name_in_a_tracked_file_fails(repo: Path, configured) -> None:
    result = _run(repo)
    assert result.returncode == 1
    assert "notes.py:1" in result.stderr


def test_deleting_the_name_in_a_new_commit_does_not_clean_the_repository(
    repo: Path, configured
) -> None:
    """The bug this whole mode exists for.

    Rewriting the file is what a person does when told "take that name out".
    The working tree is then clean and the repository is not, and for a public
    repo only the second one matters.
    """
    (repo / "notes.py").write_text("# built for a downstream consumer\nX = 1\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "drop the name")

    worktree = _run(repo)
    assert worktree.returncode == 0, "the checkout really is clean now"

    history = _run(repo, "--history")
    assert history.returncode == 1, "but the old blob is still in the object database"
    assert "blob " in history.stderr


def test_history_mode_is_satisfied_only_by_an_actual_rewrite(repo: Path, configured) -> None:
    (repo / "notes.py").write_text("# built for a downstream consumer\nX = 1\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "drop the name")
    # What filter-repo does, in miniature: rebuild the branch with no commit
    # that ever held the name, then drop every reference to the old chain.
    _git(repo, "checkout", "-q", "--orphan", "clean")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "rebuilt")
    _git(repo, "branch", "-qD", "main")
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "gc", "-q", "--prune=now", "--aggressive")

    assert _run(repo, "--history").returncode == 0


def test_a_name_in_a_commit_message_is_caught(repo: Path, configured) -> None:
    (repo / "other.txt").write_text("nothing here\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-qm", f"wire up {SECRET} support")

    result = _run(repo, "--history")
    assert result.returncode == 1
    assert "(message)" in result.stderr


@pytest.mark.parametrize("mode", [(), ("--history",)])
def test_the_report_never_repeats_the_name_it_found(repo: Path, configured, mode) -> None:
    """The output is a public CI log. It may say where, never what."""
    result = _run(repo, *mode)
    assert result.returncode == 1
    combined = (result.stdout + result.stderr).lower()
    assert SECRET not in combined
    assert "term" in combined, "it should still say which blocklist entry, by index"


def test_unconfigured_says_so_rather_than_passing_quietly(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENTKIT_PRIVATE_IDENTIFIERS", raising=False)
    # Point at a path that does not exist, so the maintainer's real list at the
    # default location cannot make this pass for the wrong reason.
    monkeypatch.setenv("AGENTKIT_PRIVATE_IDENTIFIERS_FILE", str(repo / "nope"))
    result = _run(repo)
    assert result.returncode == 0
    assert "skipped" in result.stdout
    assert "nothing was checked" in result.stdout


# A blocklist term made only of [0-9a-f]. Matching is plain substring, so such a
# term can appear inside a SHA by chance — which is exactly how the forge's
# fabricated merge commit turns into a finding nobody can act on.
HEX_SECRET = "beadface"


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    """A repo with no forbidden name anywhere, so any hit comes from the test."""
    r = tmp_path / "clean"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "ok.py").write_text("X = 1\n", encoding="utf-8")
    _git(r, "add", "ok.py")
    _git(r, "commit", "-qm", "initial")
    return r


@pytest.fixture
def hex_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENTKIT_PRIVATE_IDENTIFIERS", HEX_SECRET)
    monkeypatch.delenv("AGENTKIT_PRIVATE_IDENTIFIERS_FILE", raising=False)


def test_the_forges_fabricated_merge_commit_is_not_a_finding(
    clean_repo: Path, hex_configured
) -> None:
    """A PR run scans a commit the forge invented and deletes again.

    Its message is two random SHAs, so a hex-shaped blocklist term matches it
    sooner or later. Reporting that tells the author to rewrite the history of
    a commit they cannot even fetch.
    """
    left = HEX_SECRET + "0" * (40 - len(HEX_SECRET))
    right = "8bd66c945f7b9bb4a101cf7e79be4e181ee9c98e"
    _git(clean_repo, "commit", "-q", "--allow-empty", "-m", f"Merge {left} into {right}")

    result = _run(clean_repo, "--history")
    assert result.returncode == 0, result.stderr


def test_a_real_merge_commit_is_still_scanned(clean_repo: Path, hex_configured) -> None:
    """The skip is the forge's exact template and nothing wider."""
    _git(
        clean_repo,
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        f"Merge branch 'topic'\n\nbrought over from {HEX_SECRET}\n",
    )

    result = _run(clean_repo, "--history")
    assert result.returncode == 1
    assert "(message)" in result.stderr


def test_a_finding_names_the_commit_that_carries_it(clean_repo: Path, hex_configured) -> None:
    """Not the newest one.

    The commit scan used to split the log on NUL-NUL, a delimiter git never
    emits, so every message arrived as a single record and every finding was
    attributed to whichever commit happened to be at the tip. That is the
    difference between "rewrite this commit" and "rewrite a commit you have
    never seen".
    """
    _git(clean_repo, "commit", "-q", "--allow-empty", "-m", f"carries {HEX_SECRET}")
    guilty = _git(clean_repo, "rev-parse", "HEAD").strip()
    _git(clean_repo, "commit", "-q", "--allow-empty", "-m", "innocent tip")
    tip = _git(clean_repo, "rev-parse", "HEAD").strip()

    result = _run(clean_repo, "--history")
    assert result.returncode == 1
    assert guilty[:12] in result.stderr
    assert tip[:12] not in result.stderr


def _plant_clean_content(repo: Path) -> None:
    """A tracked file whose CONTENT is clean, planted under a bad NAME."""
    docs = repo / "docs"
    docs.mkdir()
    (docs / f"{SECRET}-notes.md").write_text("clean prose\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add docs")


def test_a_private_name_in_a_tracked_path_fails(repo: Path, configured) -> None:
    """The forge publishes a branch's file tree; the content scans never look at paths.

    This is how one survived: its file's content was scrubbed and pushed while
    the filename itself stayed, and every scan read only what was inside.
    """
    _plant_clean_content(repo)

    result = _run(repo)
    assert result.returncode == 1
    lines = [line.strip() for line in result.stderr.splitlines()]
    assert any(line.startswith("path ") for line in lines)
    assert "[terms #0]" in result.stderr
    assert SECRET not in result.stdout + result.stderr


def test_the_scope_line_counts_ref_names_too(repo: Path, configured) -> None:
    _plant_clean_content(repo)

    result = _run(repo)
    assert ", ref names)" in result.stdout + result.stderr or "ref names)" in result.stderr


@pytest.mark.parametrize("mode", [(), ("--history",)])
def test_a_private_name_in_a_local_ref_fails(repo: Path, configured, mode) -> None:
    """Branch and tag lists are public. Local heads/tags are still fixable."""
    _git(repo, "branch", f"feature-{SECRET}")
    _git(repo, "tag", f"v1-{SECRET}")

    result = _run(repo, *mode)
    assert result.returncode == 1
    lines = [line.strip() for line in result.stderr.splitlines()]
    labels = [line for line in lines if line.startswith("ref name ")]
    assert len(labels) == 2, "one local branch and one tag"
    for line in labels:
        assert "[terms #0]" in line
        assert SECRET not in line, "the name must not echo, ever"


def _benign_repo(tmp_path: Path) -> Path:
    """A repo with no bad content anywhere; name findings only."""
    r = tmp_path / "benign"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "app.py").write_text("X = 1\n", encoding="utf-8")
    _git(r, "add", "app.py")
    _git(r, "commit", "-qm", "base")
    return r


def test_an_inherited_remote_tracking_ref_warns_but_does_not_block(
    tmp_path: Path, configured
) -> None:
    """refs/remotes/** is already public and fixing it is the maintainer's call.

    Wiring those to turn CI red would re-create the permanent red-dot failure
    this whole guard exists to end: an advisory that says 'stale backlog' and
    clears on its own as cleanup lands, without breaking every PR meanwhile.
    """
    repo = _benign_repo(tmp_path)
    _git(repo, "update-ref", f"refs/remotes/origin/{SECRET}-release", "HEAD")

    result = _run(repo)
    assert result.returncode == 0, "warnings must not fail CI"
    assert "ok — no blocking private identifiers" in result.stdout
    lines = [line.strip() for line in result.stderr.splitlines()]
    warning = [line for line in lines if line.startswith("WARNING ref name ")]
    assert len(warning) == 1
    assert "[terms #0]" in warning[0]
    assert SECRET not in result.stdout + result.stderr


def test_verbose_unmasks_name_surfaces_locally_only(
    repo: Path, configured, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default output masks; --verbose prints raw names for list-holders locally.

    Whoever holds the blocklist may see the offending names — on their own
    terminal. CI logs are public, so CI must never run with this on.
    """
    _plant_clean_content(repo)

    masked = _run(repo)
    assert SECRET not in masked.stdout + masked.stderr

    monkeypatch.setenv("AGENTKIT_GUARD_VERBOSE", "1")
    verbose = _run(repo)
    assert verbose.returncode == 1
    assert f"{SECRET}-notes.md" in verbose.stderr
