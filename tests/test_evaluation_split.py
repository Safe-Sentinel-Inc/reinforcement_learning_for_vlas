"""Tests for scripts/evaluation/ split modules."""
import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.metrics import _safe_corr, _linear_slope, _sample_indices, _bin_progress_statistics


def test_safe_corr_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(_safe_corr(a, a) - 1.0) < 1e-6

def test_safe_corr_constant():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 3.0, 4.0])
    assert np.isnan(_safe_corr(a, b))

def test_safe_corr_too_short():
    assert np.isnan(_safe_corr(np.array([1.0]), np.array([2.0])))

def test_linear_slope_positive():
    values = np.array([0.0, 1.0, 2.0, 3.0])
    slope = _linear_slope(values)
    assert slope > 0

def test_linear_slope_single():
    assert _linear_slope(np.array([5.0])) == 0.0

def test_sample_indices_no_downsample():
    idx = _sample_indices(5, 10)
    np.testing.assert_array_equal(idx, np.arange(5))

def test_sample_indices_downsample():
    idx = _sample_indices(100, 10)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx[-1] == 99

def test_bin_progress_statistics():
    progress = np.linspace(0, 1, 100)
    values = progress * 2
    centers, stats = _bin_progress_statistics(progress, values, bins=10)
    assert len(centers) == 10
    assert not np.isnan(stats).all()
    # Mean should increase across bins
    non_nan = stats[~np.isnan(stats)]
    if len(non_nan) > 1:
        assert non_nan[-1] > non_nan[0]
