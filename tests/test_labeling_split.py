"""Tests for scripts/labeling/ split modules."""
import sys
import pathlib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.labeling.dataset_utils import parse_range_string, compute_fold_assignments, resolve_success_episodes
from scripts.labeling.progress_labeling import compute_binned_value_progress, compute_intervention_labels
from scripts.labeling.vf_inference import _np_collate


def test_parse_range_string_basic():
    assert parse_range_string("0-3,5,7-9") == {0, 1, 2, 3, 5, 7, 8, 9}

def test_parse_range_string_empty():
    assert parse_range_string("") == set()

def test_parse_range_string_single():
    assert parse_range_string("42") == {42}

def test_compute_fold_assignments_coverage():
    indices = list(range(10))
    folds = compute_fold_assignments(indices, num_folds=3, seed=42)
    assert set(folds.keys()) == set(indices)
    assert set(folds.values()) <= {0, 1, 2}
    # Every fold should have at least one episode
    for f in range(3):
        assert any(v == f for v in folds.values())

def test_compute_fold_assignments_deterministic():
    indices = list(range(20))
    a = compute_fold_assignments(indices, num_folds=5, seed=99)
    b = compute_fold_assignments(indices, num_folds=5, seed=99)
    assert a == b

def test_resolve_success_episodes_all():
    all_eps = {0, 1, 2, 3, 4}
    result = resolve_success_episodes("all", [2, 4], all_eps)
    assert result == {0, 1, 3}

def test_resolve_success_episodes_range():
    all_eps = set(range(10))
    result = resolve_success_episodes("0-5", [3], all_eps)
    assert result == {0, 1, 2, 4, 5}

def test_resolve_success_episodes_list():
    all_eps = set(range(10))
    result = resolve_success_episodes([1, 3, 5], [3], all_eps)
    assert result == {1, 5}

def test_compute_binned_value_progress_success():
    lengths = {0: 100}
    binned = compute_binned_value_progress(lengths, success_episodes={0}, num_bins=200)
    assert 0 in binned
    assert len(binned[0]) == 100
    assert binned[0][0] == 0
    assert binned[0][-1] == 199

def test_compute_binned_value_progress_failed():
    lengths = {0: 50, 1: 100}
    binned = compute_binned_value_progress(lengths, success_episodes={1}, num_bins=200)
    # Failed episode (0) should end near 0 due to decay
    assert binned[0][-1] < 50
    # Success episode (1) should end at 199
    assert binned[1][-1] == 199

def test_compute_intervention_labels():
    lengths = {0: 100, 1: 50}
    ranges = {0: [[10, 20], [30, 40]]}
    labels = compute_intervention_labels(lengths, ranges)
    assert labels[0][15] == True
    assert labels[0][5] == False
    assert labels[1].sum() == 0

def test_np_collate():
    batch = [
        {"a": np.array([1.0, 2.0]), "b": np.array([3.0])},
        {"a": np.array([4.0, 5.0]), "b": np.array([6.0])},
    ]
    result = _np_collate(batch)
    assert result["a"].shape == (2, 2)
    assert result["b"].shape == (2, 1)
    np.testing.assert_array_equal(result["a"][0], [1.0, 2.0])
