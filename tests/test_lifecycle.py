"""pull_latest, against real git — offline, local remotes only.

/api/restart's whole point is publishing an edit made somewhere else (GitHub's
own editor, on a phone) onto this machine. That's a real git operation, not
something worth faking, so these build an actual bare "remote" and clones of
it in tmp_path and drive real `git pull` against them. No network involved —
everything here is local paths acting as remotes.
"""

import subprocess
from pathlib import Path

import pytest

from sevanya import lifecycle


def run_git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def repo_pair(tmp_path):
    """A bare 'remote' and a clone of it, one commit in, both local."""
    remote = tmp_path / "remote.git"
    run_git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)

    clone = tmp_path / "clone"
    run_git("clone", str(remote), str(clone), cwd=tmp_path)
    run_git("config", "user.email", "t@example.com", cwd=clone)
    run_git("config", "user.name", "Test", cwd=clone)
    (clone / "f.txt").write_text("one\n")
    run_git("add", "f.txt", cwd=clone)
    run_git("commit", "-m", "first", cwd=clone)
    run_git("push", "-u", "origin", "main", cwd=clone)

    return remote, clone


def test_fast_forwards_a_clean_checkout_onto_a_new_commit(repo_pair):
    """The core case: an edit landed on the remote, this checkout picks it up."""
    remote, clone = repo_pair
    second = clone.parent / "second"
    run_git("clone", str(remote), str(second), cwd=clone.parent)

    # A change pushed from elsewhere — GitHub's own editor, on a phone.
    (clone / "f.txt").write_text("two\n")
    run_git("commit", "-am", "second", cwd=clone)
    run_git("push", cwd=clone)

    ok, message = lifecycle.pull_latest(second)
    assert ok, message
    assert (second / "f.txt").read_text() == "two\n"


def test_already_up_to_date_still_reports_success(repo_pair):
    _remote, clone = repo_pair
    ok, message = lifecycle.pull_latest(clone)
    assert ok
    assert "up to date" in message.lower()


def test_a_real_divergence_is_refused_not_merged_or_discarded(repo_pair):
    """--ff-only, so a genuine conflict comes back as a failure to report —

    never a merge commit, and never local work silently thrown away.
    """
    remote, clone = repo_pair
    second = clone.parent / "second"
    run_git("clone", str(remote), str(second), cwd=clone.parent)
    run_git("config", "user.email", "t@example.com", cwd=second)
    run_git("config", "user.name", "Test", cwd=second)

    (clone / "f.txt").write_text("from clone\n")
    run_git("commit", "-am", "from clone", cwd=clone)
    run_git("push", cwd=clone)

    # Committed, not pushed — this is the "edited on the PC too" case.
    (second / "f.txt").write_text("from second\n")
    run_git("commit", "-am", "from second", cwd=second)

    ok, message = lifecycle.pull_latest(second)
    assert not ok
    assert message
    assert (second / "f.txt").read_text() == "from second\n", \
        "a refused pull must leave local work exactly as it was"


def test_reports_rather_than_raises_outside_a_repository(tmp_path):
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()
    ok, message = lifecycle.pull_latest(not_a_repo)
    assert not ok
    assert message
