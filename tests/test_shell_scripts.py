"""Validate shell scripts: no Chinese, no airbot dir refs, correct file references."""
import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CMDS_DIR = PROJECT_ROOT / "scripts" / "cmds"
CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def test_shell_scripts_no_chinese():
    violations = []
    for sh in CMDS_DIR.glob("*.sh"):
        text = sh.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if CHINESE_RE.search(line):
                violations.append(f"{sh.name}:{i}: {line.strip()[:80]}")
    assert not violations, f"Found Chinese in shell scripts:\n" + "\n".join(violations)


def test_shell_scripts_no_airbot_dir():
    """Shell scripts should reference examples/robot/, not examples/airbot/."""
    violations = []
    for sh in CMDS_DIR.glob("*.sh"):
        text = sh.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if "examples/airbot" in line:
                violations.append(f"{sh.name}:{i}: {line.strip()[:80]}")
    assert not violations, f"Found 'examples/airbot' in shell scripts:\n" + "\n".join(violations)


def test_shell_scripts_valid_syntax():
    """All .sh files should be valid bash (non-empty, has shebang)."""
    for sh in CMDS_DIR.glob("*.sh"):
        text = sh.read_text()
        assert text.strip(), f"{sh.name} is empty"
        assert text.startswith("#!/"), f"{sh.name} missing shebang"
