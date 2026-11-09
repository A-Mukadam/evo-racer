"""Tests for the run-group statistics (experiments/compare.py)."""

import pytest

from experiments.compare import fisher_p, permutation_p


def test_fisher_matches_textbook_values():
    assert fisher_p(3, 5, 4, 5) == pytest.approx(1.0)
    assert fisher_p(0, 5, 5, 5) == pytest.approx(2 / 252)        # 2 / C(10, 5)
    assert fisher_p(1, 9, 11, 14) == pytest.approx(0.002759, abs=1e-5)


def test_permutation_p():
    assert permutation_p([1, 2, 3], [1, 2, 3], n=2000) == pytest.approx(1.0)
    assert permutation_p([0, 0, 0, 0, 0], [3, 3, 3, 3, 3], n=5000) < 0.02
