"""Tests for examples/robot/ shared inference modules."""
import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "robot"))

from inference_helpers import RemotePolicyConfig, AutoConfig, interpolate_action


def test_remote_policy_config_defaults():
    cfg = RemotePolicyConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 8000

def test_auto_config_defaults():
    cfg = AutoConfig()
    assert cfg.chunk_size_predict == 0
    assert cfg.state_dim == -1

def test_interpolate_action_no_interp():
    prev = np.zeros(7)
    cur = np.zeros(7)
    result = interpolate_action([0.01]*7, prev, cur)
    assert result.shape == (1, 7)

def test_interpolate_action_smooth():
    prev = np.zeros(7)
    cur = np.ones(7)
    step_length = [0.1]*7
    result = interpolate_action(step_length, prev, cur)
    assert result.shape[0] > 1
    # Last row should be close to cur
    np.testing.assert_allclose(result[-1], cur, atol=1e-6)

def test_interpolate_action_mismatched_step_length():
    prev = np.zeros(14)
    cur = np.ones(14)
    step_length = [0.1]*7  # shorter than action dim
    result = interpolate_action(step_length, prev, cur)
    assert result.shape[1] == 14
