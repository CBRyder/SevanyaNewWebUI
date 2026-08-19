"""Reaching outside the machine: what she'll fetch, and what she refuses.

Most of this file is about refusal. Sevanya runs on a personal machine that's
on Tailscale, so "fetch this URL" is also a way to reach the router's admin
page or something bound to localhost — and the model can be talked into trying
by any page it reads. The blocks below are the reason that isn't a problem.

No test here touches the network. Fetches go through a mock transport, and the
git tests drive a recorded stub, so the suite stays offline and deterministic.
"""

import httpx
import pytest

from sevanya import net


@pytest.fixture
def public(monkeypatch):
    """Treat every hostname as publicly resolvable, so tests don't need DNS."""
    monkeypatch.setattr(net, "_is_public", lambda host: True)


# --- what it refuses -------------------------------------------------------


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/thing",
    "javascript:alert(1)",
    "/etc/passwd",
])
def test_only_http_and_https(url):
    with pytest.raises(net.Blocked, match="http"):
        net.check_url(url)


@pytest.mark.parametrize("host", ["localhost", "router.local", "nas.internal", "box.lan", "pi.home"])
def test_local_names_are_refused_by_name(host):
    with pytest.raises(net.Blocked, match="local name"):
        net.check_url(f"http://{host}/")


@pytest.mark.parametrize("address", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private
    "192.168.1.1",      # the router's admin page
    "172.16.0.1",       # private
    "[::1]",            # loopback, v6
    "169.254.169.254",  # cloud metadata, the classic SSRF target
    "0.0.0.0",
])
def test_private_and_local_addresses_are_refused(address):
    with pytest.raises(net.Blocked, match="private or local"):
        net.check_url(f"http://{address}/")


def test_a_name_resolving_to_both_public_and_local_is_refused(monkeypatch):
    """One bad address is enough to refuse the whole name.

    Otherwise which address the connection picks decides whether the block
    holds, which is not a security property.
    """
    def resolve(host, _port):
        return [(0, 0, 0, "", ("93.184.216.34", 0)), (0, 0, 0, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(net.socket, "getaddrinfo", resolve)
    with pytest.raises(net.Blocked, match="private or local"):
        net.check_url("http://sneaky.example/")


def test_a_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(net.socket, "getaddrinfo",
                        lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0))])
    assert net.check_url("https://example.com/page") == "https://example.com/page"


def test_a_name_that_does_not_resolve_is_refused(monkeypatch):
    monkeypatch.setattr(net.socket, "getaddrinfo",
                        lambda host, port: (_ for _ in ()).throw(net.socket.gaierror()))
    with pytest.raises(net.Blocked):
        net.check_url("http://nowhere.invalid/")


# --- redirects -------------------------------------------------------------


def test_a_redirect_to_localhost_is_refused(monkeypatch):
    """The check has to run on every hop, not just the URL you were given.

    A perfectly public URL is free to answer with a redirect to 127.0.0.1, and
    httpx following redirects on its own would take it.
    """
    monkeypatch.setattr(net, "_is_public",
                        lambda host: host not in ("127.0.0.1", "localhost"))

    def handler(request):
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8765/api/chat"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(net.Blocked, match="local"):
        net.fetch("https://public.example/start", client=client)


def test_redirects_are_followed_when_they_stay_public(public):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://public.example/end"})
        return httpx.Response(200, text="arrived", headers={"content-type": "text/plain"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    assert "arrived" in net.fetch("https://public.example/start", client=client)


def test_a_redirect_loop_gives_up(public):
    def handler(request):
        return httpx.Response(302, headers={"location": "https://public.example/round"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(net.Blocked, match="too many redirects"):
        net.fetch("https://public.example/round", client=client)


# --- reading a page --------------------------------------------------------


HTML = """
<html><head><title>My Site</title>
<style>body { color: red }</style>
<script>var secret = 'not prose';</script>
</head><body>
<h1>Welcome</h1>
<p>First paragraph.</p>
<p>Second   paragraph.</p>
<noscript>enable javascript</noscript>
</body></html>
"""


def test_html_becomes_readable_text():
    title, text = net.to_text(HTML)
    assert title == "My Site"
    assert "Welcome" in text
    assert "First paragraph." in text
    assert "Second paragraph." in text, "runs of whitespace should collapse"


def test_script_and_style_contents_are_dropped():
    """They aren't prose, and they'd otherwise dominate a real page."""
    _, text = net.to_text(HTML)
    assert "not prose" not in text
    assert "color: red" not in text
    assert "enable javascript" not in text


def test_fetch_labels_content_as_untrusted(public):
    """The model has to be able to tell a fetched page from its instructions."""
    def handler(request):
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    out = net.fetch("https://public.example/", client=client)
    assert "not as instructions" in out
    assert "https://public.example/" in out
    assert "My Site" in out


def test_fetch_truncates_a_huge_page(public, monkeypatch):
    monkeypatch.setattr(net, "MAX_CHARS", 100)

    def handler(request):
        return httpx.Response(200, text="x" * 5000, headers={"content-type": "text/plain"})

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    out = net.fetch("https://public.example/", client=client)
    assert "truncated" in out
    assert len(out) < 500


def test_an_http_error_is_not_swallowed(public):
    def handler(request):
        return httpx.Response(404, text="gone")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(httpx.HTTPStatusError):
        net.fetch("https://public.example/missing", client=client)


# --- naming a repo ---------------------------------------------------------


@pytest.mark.parametrize("spec,expected", [
    ("anthropics/courses", ("anthropics", "courses")),
    ("anthropics/courses.git", ("anthropics", "courses")),
    ("https://github.com/anthropics/courses", ("anthropics", "courses")),
    ("https://github.com/anthropics/courses.git", ("anthropics", "courses")),
    ("git@github.com:anthropics/courses.git", ("anthropics", "courses")),
    ("  anthropics/courses  ", ("anthropics", "courses")),
])
def test_repo_specs_that_should_work(spec, expected):
    assert net.parse_repo(spec) == expected


@pytest.mark.parametrize("spec", [
    "not-a-repo",
    "too/many/parts",
    "https://gitlab.com/owner/name",
    "https://evil.example/github.com/owner/name",
    "owner/../../etc",
    "../../etc/passwd",
    "owner/name;rm -rf /",
    "-upload-pack=touch /tmp/pwned",     # an option, not a name
    "owner/-oProxyCommand=sh",
    "owner/",
    "/name",
])
def test_repo_specs_that_must_be_refused(spec):
    """These become arguments to git. Anything ambiguous is refused, not fixed."""
    with pytest.raises(ValueError):
        net.parse_repo(spec)


# --- clone vs update -------------------------------------------------------


class RecordingGit:
    """Stands in for the git binary, recording what it was asked to do."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or set()

    def __call__(self, *args, cwd=None):
        self.calls.append(args)

        class Result:
            returncode = 1 if args[0] in self.fail else 0
            stdout = "abc1234\n" if args[0] == "rev-parse" else "a commit subject\n"
            stderr = "boom" if args[0] in self.fail else ""

        return Result()


def test_a_repo_we_do_not_have_is_cloned_shallow(tmp_path, monkeypatch):
    git = RecordingGit()
    monkeypatch.setattr(net, "_git", git)
    target, summary = net.sync("anthropics/courses", cache=tmp_path)

    assert "cloned" in summary
    clone = next(call for call in git.calls if call[0] == "clone")
    assert "--depth" in clone and "1" in clone, "history isn't wanted, only the code"
    assert "https://github.com/anthropics/courses.git" in clone
    assert target == tmp_path / "anthropics" / "courses"


def test_a_repo_we_already_have_is_updated_not_recloned(tmp_path, monkeypatch):
    (tmp_path / "anthropics" / "courses" / ".git").mkdir(parents=True)
    git = RecordingGit()
    monkeypatch.setattr(net, "_git", git)
    _, summary = net.sync("anthropics/courses", cache=tmp_path)

    assert "updated" in summary
    verbs = [call[0] for call in git.calls]
    assert "clone" not in verbs
    assert "fetch" in verbs
    assert "reset" in verbs, "a read-only mirror should reset, not merge"


def test_a_failed_clone_is_reported_not_silently_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(net, "_git", RecordingGit(fail={"clone"}))
    with pytest.raises(RuntimeError, match="could not clone"):
        net.sync("anthropics/nope", cache=tmp_path)


def test_a_failed_update_is_reported(tmp_path, monkeypatch):
    (tmp_path / "o" / "n" / ".git").mkdir(parents=True)
    monkeypatch.setattr(net, "_git", RecordingGit(fail={"fetch"}))
    with pytest.raises(RuntimeError, match="could not update"):
        net.sync("o/n", cache=tmp_path)


def test_git_never_waits_for_a_password():
    """A private repo should fail fast, not hang on a prompt nobody will answer."""
    import inspect
    source = inspect.getsource(net._git)
    assert "GIT_TERMINAL_PROMPT" in source and '"0"' in source
