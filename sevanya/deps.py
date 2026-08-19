"""Are the requirements installed, and do they actually import?

Two different questions, and the second is the one that bites. A package can be
present, the right version, and still fail on import because something it
depends on isn't there — which surfaces as the server dying at startup with a
traceback about a module nobody has heard of, rather than as "you're missing a
dependency". Checking imports as well as metadata is what turns that into a
sentence you can act on.
"""

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from pathlib import Path

TIMEOUT = 300  # pip can be slow on a cold cache

try:  # pip vendors this, so it's nearly always here — but don't require it
    from packaging.requirements import Requirement
except ImportError:  # pragma: no cover - only on an unusual install
    Requirement = None


def requirements_path(root: Path) -> Path:
    return root / "requirements.txt"


def read(path: Path) -> list[str]:
    """The requirement lines, without comments, blanks or pip's own options."""
    if not path.exists():
        return []
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            lines.append(line)
    return lines


def _name_of(line: str) -> str:
    if Requirement is not None:
        return Requirement(line).name
    for separator in ("==", ">=", "<=", "~=", ">", "<", "!=", "["):
        line = line.split(separator, 1)[0]
    return line.strip()


def _modules_for(dist_name: str) -> list[str]:
    """Top-level modules a distribution provides.

    Asked of the metadata rather than guessed from the name, because the two
    often differ and a guess produces confident nonsense ('no module named
    python_dateutil').
    """
    normalised = dist_name.replace("-", "_").lower()
    found = [
        module
        for module, dists in packages_distributions().items()
        if any(d.replace("-", "_").lower() == normalised for d in dists)
    ]
    return found


def check(lines: list[str]) -> list[dict]:
    """One entry per requirement: name, whether it's there, and what's wrong."""
    results = []
    for line in lines:
        name = _name_of(line)
        entry = {"line": line, "name": name, "installed": None, "problem": None}

        try:
            entry["installed"] = version(name)
        except PackageNotFoundError:
            entry["problem"] = "not installed"
            results.append(entry)
            continue

        if Requirement is not None:
            specifier = Requirement(line).specifier
            if specifier and not specifier.contains(entry["installed"], prereleases=True):
                entry["problem"] = f"have {entry['installed']}, needs {specifier}"
                results.append(entry)
                continue

        for module in _modules_for(name):
            try:
                __import__(module)
            except Exception as exc:
                # Installed but broken. This is the case worth catching: the
                # metadata says yes and the import says no.
                entry["problem"] = f"installed but won't import ({type(exc).__name__}: {exc})"
                break

        results.append(entry)
    return results


def install(path: Path) -> tuple[bool, str]:
    """pip install -r, and hand back what it said."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(path)],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def ensure(root: Path, install_missing: bool = True) -> tuple[bool, str]:
    """Check requirements, install if something's wrong, check again.

    Returns (everything_ok, a report worth showing a person). Re-checking after
    an install matters: pip exiting 0 doesn't prove the thing imports, which is
    the failure this exists to catch.
    """
    path = requirements_path(root)
    if not path.exists():
        return True, f"no requirements.txt in {root} — nothing to check"

    lines = read(path)
    if not lines:
        return True, "requirements.txt is empty — nothing to check"

    problems = [entry for entry in check(lines) if entry["problem"]]
    if not problems:
        return True, f"all {len(lines)} requirements installed and importable"

    summary = "\n".join(f"  {e['name']}: {e['problem']}" for e in problems)
    if not install_missing:
        return False, f"{len(problems)} of {len(lines)} requirements have problems:\n{summary}"

    ok, output = install(path)
    if not ok:
        tail = "\n".join(output.splitlines()[-8:])
        return False, f"pip install failed:\n{tail}"

    remaining = [entry for entry in check(lines) if entry["problem"]]
    if remaining:
        still = "\n".join(f"  {e['name']}: {e['problem']}" for e in remaining)
        return False, f"pip finished but these are still wrong:\n{still}"

    fixed = ", ".join(e["name"] for e in problems)
    return True, f"installed {fixed} — all {len(lines)} requirements now good"
