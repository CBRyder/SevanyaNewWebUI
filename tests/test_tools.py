"""Tools: the schemas the model sees, and the functions behind them.

The first test in here is the one that matters most. A tool definition with a
misspelled `input_schema` key is not a broken tool — it's a broken assistant,
because the whole tools array is validated on every request and a malformed
entry gets the entire call rejected. Sevanya couldn't answer "hello". It cost
a day to find, and it is one assert to prevent.
"""

import pytest

from sevanya import tools


# --- the contract between TOOLS, REGISTRY and dispatch ---------------------


def test_every_tool_schema_is_well_formed():
    assert tools.TOOLS, "no tools defined at all"
    for tool in tools.TOOLS:
        assert "name" in tool, f"nameless tool: {tool}"
        # Not `.get("input_Schema")` or any other near-miss — the key has to be
        # exactly this or the API rejects every request the tool appears in.
        assert "input_schema" in tool, f"{tool['name']}: no input_schema"
        schema = tool["input_schema"]
        assert schema.get("type") == "object", f"{tool['name']}: schema is not an object"
        assert "properties" in schema, f"{tool['name']}: no properties"
        for name in schema.get("required", []):
            assert name in schema["properties"], f"{tool['name']}: required '{name}' is not a property"
        assert tool.get("description"), f"{tool['name']}: no description"


def test_every_advertised_tool_is_callable():
    """A schema without an implementation is worse than no tool at all.

    The model reads the description, decides this is the right move, calls it,
    and gets "unknown tool" back — so it tries again, and burns the step limit
    on a tool that was never there.
    """
    for tool in tools.TOOLS:
        assert tool["name"] in tools.REGISTRY, f"{tool['name']} is advertised but not in REGISTRY"


def test_nothing_is_registered_that_isnt_advertised():
    advertised = {t["name"] for t in tools.TOOLS}
    for name in tools.REGISTRY:
        assert name in advertised, f"{name} is callable but the model is never told about it"


def test_store_tools_are_registered_and_advertised():
    for name in tools.NEEDS_STORE:
        assert name in tools.REGISTRY, f"{name} needs a store but isn't registered"


def test_dispatch_reports_unknown_tools_as_errors_not_exceptions():
    output, is_error = tools.dispatch("no_such_tool", {})
    assert is_error
    assert "unknown tool" in output


def test_dispatch_turns_exceptions_into_readable_text(project):
    """Errors come back as text so the model can read them and try again."""
    output, is_error = tools.dispatch("read_file", {"path": "nope.py"})
    assert is_error
    assert "nope.py" in output
    assert "FileNotFoundError" in output


def test_store_tools_refuse_politely_without_a_store():
    output, is_error = tools.dispatch("remember", {"topic": "t", "note": "n"})
    assert is_error
    assert "unavailable" in output


# --- path containment ------------------------------------------------------


@pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd", "pkg/../../escape"])
def test_paths_cannot_escape_the_project(project, path):
    """Every path from the model is untrusted input.

    A file it reads is a file that ends up in the transcript, so this is the
    difference between a code reader and an exfiltration tool.
    """
    with pytest.raises(ValueError, match="escapes the project root"):
        tools._resolve(path)


def test_the_project_root_itself_is_allowed(project):
    assert tools._resolve(".") == project


def test_read_file_truncates_rather_than_flooding_the_context(project, monkeypatch):
    monkeypatch.setattr(tools, "MAX_CHARS", 50)
    (project / "long.txt").write_text("x" * 500)
    out = tools.read_file("long.txt")
    assert "truncated" in out
    assert len(out) < 200


def test_read_file_rejects_directories(project):
    with pytest.raises(IsADirectoryError):
        tools.read_file("pkg")


def test_list_files_hides_noise(project):
    listing = tools.list_files("pkg")
    assert "code.py" in listing
    assert "__pycache__" not in listing


# --- grep ------------------------------------------------------------------


def test_grep_finds_a_match_and_reports_file_and_line(project):
    out = tools.grep("MAX_STEPS")
    assert "pkg/code.py:1:" in out
    assert "MAX_STEPS = 12" in out


def test_grep_is_case_insensitive(project):
    assert "notes.md" in tools.grep("MAX_STEPS")


def test_grep_is_literal_not_a_regex(project):
    """A regex would make '.' and '(' in an identifier mean something else.

    The model reaches for this with a name it just read on screen; turning
    that name into pattern syntax produces confidently wrong answers.
    """
    (project / "dotted.py").write_text("value = obj.attr\nvalue = objXattr\n")
    out = tools.grep("obj.attr")
    assert "obj.attr" in out
    assert "objXattr" not in out


def test_grep_skips_build_dirs_and_binaries(project):
    out = tools.grep("MAX_STEPS")
    assert "__pycache__" not in out
    assert "blob.bin" not in out


def test_grep_can_search_a_single_file(project):
    out = tools.grep("MAX_STEPS", "pkg/code.py")
    assert "pkg/code.py" in out
    assert "notes.md" not in out


def test_grep_says_plainly_when_there_is_nothing(project):
    out = tools.grep("nothing-matches-this")
    assert "No matches" in out


def test_grep_caps_its_output(project, monkeypatch):
    monkeypatch.setattr(tools, "MAX_MATCHES", 5)
    (project / "many.txt").write_text("needle\n" * 100)
    out = tools.grep("needle")
    assert "stopped at 5 matches" in out
    assert len(out.splitlines()) <= 6


def test_grep_refuses_an_empty_pattern(project):
    with pytest.raises(ValueError):
        tools.grep("")


def test_grep_cannot_escape_the_project(project):
    output, is_error = tools.dispatch("grep", {"pattern": "root", "path": "../.."})
    assert is_error
    assert "escapes the project root" in output


# --- the second root: her cache of cloned repos ----------------------------
#
# Adding a root is adding a way out. Everything the project root is checked
# for, the cache root is checked for too.


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A repo cache with one fake clone in it, wired into tools."""
    from sevanya import net

    root = tmp_path / "cache"
    (root / "owner" / "name").mkdir(parents=True)
    (root / "owner" / "name" / "lib.py").write_text("def widget():\n    return 42\n")
    monkeypatch.setattr(tools, "REPO_ROOT", root)
    monkeypatch.setattr(net, "REPO_CACHE", root)
    return root


def test_a_cloned_repo_reads_like_any_other_directory(project, cache):
    assert "def widget()" in tools.read_file("repos/owner/name/lib.py")
    assert "lib.py" in tools.list_files("repos/owner/name")


def test_grep_works_across_a_cloned_repo(project, cache):
    out = tools.grep("widget", "repos/owner/name")
    assert "repos/owner/name/lib.py:1:" in out


def test_paths_reported_from_the_cache_can_be_read_back(project, cache):
    """Whatever grep prints has to be a path read_file accepts.

    If the label and the addressable path differ, the model has to translate
    between them, and it will eventually get it wrong.
    """
    line = tools.grep("widget", "repos/owner/name").splitlines()[0]
    path = line.split(":")[0]
    assert "def widget()" in tools.read_file(path)


@pytest.mark.parametrize("path", [
    "repos/../../../etc/passwd",
    "repos/owner/../../../../etc",
    "repos/owner/name/../../../../../etc/shadow",
])
def test_the_cache_root_cannot_be_escaped_either(project, cache, path):
    with pytest.raises(ValueError, match="escapes"):
        tools._resolve(path)


def test_the_cache_is_separate_from_the_project(project, cache):
    """A repos/ path must not be able to reach the user's own files."""
    (project / "secret.txt").write_text("private")
    with pytest.raises(FileNotFoundError):
        tools.read_file("repos/secret.txt")


def test_a_projects_own_repos_directory_still_wins(project, cache):
    """Adding this prefix must not change what an existing path means."""
    (project / "repos").mkdir()
    (project / "repos" / "mine.txt").write_text("the user's own file")
    assert tools.read_file("repos/mine.txt") == "the user's own file"


def test_sync_repo_says_so_when_the_prefix_is_shadowed(project, cache, monkeypatch):
    """Silently returning an unreachable path would be the worst outcome."""
    from sevanya import net

    (project / "repos").mkdir()

    def fake_sync(spec, cache=None):
        return cache_root / "owner" / "name", "cloned owner/name at abc1234 (msg)"

    cache_root = cache
    monkeypatch.setattr(net, "sync", fake_sync)

    out = tools.sync_repo("owner/name")
    assert "own repos/ directory" in out
    assert "can't be read" in out


def test_sync_repo_hands_back_a_path_that_works(project, cache, monkeypatch):
    """The reply should tell the model exactly how to read what it just fetched."""
    from sevanya import net

    def fake_sync(spec, cache=None):   # noqa: ARG001 - matches net.sync's signature
        return cache_root / "owner" / "name", "cloned owner/name at abc1234 (msg)"

    cache_root = cache
    monkeypatch.setattr(net, "sync", fake_sync)
    out = tools.sync_repo("owner/name")
    assert "repos/owner/name/" in out
    assert "lib.py" in out, "it should show what's in there"
