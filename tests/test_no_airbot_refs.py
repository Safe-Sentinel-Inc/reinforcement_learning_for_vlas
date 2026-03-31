"""Scan paths and content for unwanted 'airbot' references."""
import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AIRBOT_RE = re.compile(r"airbot", re.IGNORECASE)

# These are allowed: config names defined in src/openpi, FlatBuffers schema, hardware SDK imports
ALLOWED_PATTERNS = [
    re.compile(r"pi06_rl_\w*airbot"),           # config name strings
    re.compile(r"airbot_fbs\.FloatArray"),        # FlatBuffers schema name
    re.compile(r"from airbot_ie\."),              # hardware SDK import
    re.compile(r"airbot_play"),                   # hardware SDK class/module name
    re.compile(r"AIRBOTPlay"),                    # hardware SDK class name
    re.compile(r"airbot_policy"),                 # policy module in src/openpi
    re.compile(r"test_no_airbot_refs"),           # this test file itself
]

SKIP_DIRS = {"src/openpi", ".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", "tests"}
SKIP_EXTENSIONS = {".pyc", ".pyo", ".so", ".pkl", ".npy", ".npz", ".h5", ".lock", ".ipynb"}


def _should_skip(path: pathlib.Path) -> bool:
    parts = path.relative_to(PROJECT_ROOT).parts
    for skip in SKIP_DIRS:
        skip_parts = skip.split("/")
        if parts[:len(skip_parts)] == tuple(skip_parts):
            return True
    return path.suffix in SKIP_EXTENSIONS


def _is_allowed(line: str) -> bool:
    return any(p.search(line) for p in ALLOWED_PATTERNS)


def test_no_airbot_in_paths():
    violations = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))
        if AIRBOT_RE.search(rel) and not _is_allowed(rel):
            violations.append(rel)
    assert not violations, f"Found 'airbot' in file paths:\n" + "\n".join(violations)

def test_no_airbot_in_content():
    violations = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip(path):
            continue
        if path.suffix not in {".py", ".sh", ".md", ".txt", ".toml", ".cfg", ".yml", ".yaml", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if AIRBOT_RE.search(line) and not _is_allowed(line):
                rel = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel}:{i}: {line.strip()[:100]}")
    assert not violations, f"Found unwanted 'airbot' references:\n" + "\n".join(violations[:20])
