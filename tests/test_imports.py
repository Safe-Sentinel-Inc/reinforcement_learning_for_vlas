"""Test that all new split modules can be imported."""
import ast
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def _parse_file(path):
    """Parse a Python file and return True if syntax is valid."""
    with open(path) as f:
        ast.parse(f.read())
    return True

def test_labeling_package_syntax():
    pkg = PROJECT_ROOT / "scripts" / "labeling"
    for py in pkg.glob("*.py"):
        assert _parse_file(py), f"Syntax error in {py}"

def test_evaluation_package_syntax():
    pkg = PROJECT_ROOT / "scripts" / "evaluation"
    for py in pkg.glob("*.py"):
        assert _parse_file(py), f"Syntax error in {py}"


def test_inference_shared_modules_syntax():
    robot = PROJECT_ROOT / "examples" / "robot"
    for py in ["keyboard_listener.py", "inference_helpers.py", "inference_sync.py", "inference_async.py"]:
        assert _parse_file(robot / py), f"Syntax error in {py}"

def test_wrapper_scripts_syntax():
    assert _parse_file(PROJECT_ROOT / "scripts" / "add_returns_to_lerobot.py")
    assert _parse_file(PROJECT_ROOT / "scripts" / "evaluate_pi06_offline.py")
