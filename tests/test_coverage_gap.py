"""
Tests for tasking_helper.coverage gap analysis functions.

Tests are implementation-agnostic: they exercise the public API and
verify correctness regardless of whether the C extension is present.
"""

import numpy as np
import pytest

from tasking_helper.coverage import (
    gap_count,
    gap_count_and_max,
    gap_lengths,
    max_gap,
    total_gap,
)


# ── fixtures / helpers ────────────────────────────────────────────────────────

def make_matrix(*cols):
    """Build a 2-D uint8 matrix from column vectors (lists)."""
    return np.column_stack([np.array(c, dtype=np.uint8) for c in cols])


# ── gap_lengths ───────────────────────────────────────────────────────────────

class TestGapLengths:
    def test_single_gap(self):
        A = np.array([[0], [0], [0], [1]], dtype=np.uint8)
        result = gap_lengths(A)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [3])

    def test_two_gaps(self):
        # col: 0 0 1 0 0 0 1 0
        col = [0, 0, 1, 0, 0, 0, 1, 0]
        A = np.array(col, dtype=np.uint8).reshape(-1, 1)
        result = gap_lengths(A)
        assert len(result) == 1
        np.testing.assert_array_equal(sorted(result[0]), sorted([2, 3, 1]))

    def test_no_gaps(self):
        A = np.ones((5, 3), dtype=np.uint8)
        result = gap_lengths(A)
        assert len(result) == 3
        for col_gaps in result:
            assert col_gaps.size == 0

    def test_all_zeros(self):
        A = np.zeros((6, 2), dtype=np.uint8)
        result = gap_lengths(A)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], [6])
        np.testing.assert_array_equal(result[1], [6])

    def test_multiple_columns(self):
        # col0: gap of 3; col1: two gaps of 1 each; col2: no gaps
        A = make_matrix(
            [0, 0, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 2, 3, 4, 5],
        )
        result = gap_lengths(A)
        np.testing.assert_array_equal(result[0], [3])
        np.testing.assert_array_equal(result[1], [1, 1])
        assert result[2].size == 0

    def test_axis1(self):
        # Transpose: analyse along columns → one result per row
        A = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8)
        result = gap_lengths(A, axis=1)
        assert len(result) == 2
        # row 0: [0, 0, 1] → one gap of length 2
        np.testing.assert_array_equal(result[0], [2])
        # row 1: [1, 1, 1] → no gaps
        assert result[1].size == 0

    def test_1d_input(self):
        a = np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=np.uint8)
        result = gap_lengths(a)
        assert len(result) == 1
        np.testing.assert_array_equal(sorted(result[0]), sorted([2, 1, 3]))

    def test_gap_at_boundary_start(self):
        col = [0, 0, 1, 1, 1]
        A = np.array(col, dtype=np.uint8).reshape(-1, 1)
        result = gap_lengths(A)
        np.testing.assert_array_equal(result[0], [2])

    def test_gap_at_boundary_end(self):
        col = [1, 1, 0, 0]
        A = np.array(col, dtype=np.uint8).reshape(-1, 1)
        result = gap_lengths(A)
        np.testing.assert_array_equal(result[0], [2])

    def test_alternating(self):
        col = [0, 1, 0, 1, 0]
        A = np.array(col, dtype=np.uint8).reshape(-1, 1)
        result = gap_lengths(A)
        assert len(result[0]) == 3
        np.testing.assert_array_equal(result[0], [1, 1, 1])

    def test_dtype_int32(self):
        A = np.array([[0, 0, 1]], dtype=np.uint8).T
        result = gap_lengths(A)
        assert result[0].dtype == np.int32


# ── max_gap ───────────────────────────────────────────────────────────────────

class TestMaxGap:
    def test_simple(self):
        A = make_matrix(
            [0, 0, 0, 1, 0],  # max gap = 3
            [0, 1, 0, 0, 1],  # max gap = 2
            [1, 1, 1, 1, 1],  # max gap = 0
        )
        result = max_gap(A)
        np.testing.assert_array_equal(result, [3, 2, 0])

    def test_all_zero_column(self):
        A = np.zeros((10, 1), dtype=np.uint8)
        result = max_gap(A)
        np.testing.assert_array_equal(result, [10])

    def test_single_zero(self):
        A = np.array([[1], [0], [1]], dtype=np.uint8)
        result = max_gap(A)
        np.testing.assert_array_equal(result, [1])

    def test_dtype_int64(self):
        A = np.zeros((5, 3), dtype=np.uint8)
        assert max_gap(A).dtype == np.int64

    def test_large_matrix(self):
        rng = np.random.default_rng(42)
        A = rng.integers(0, 2, size=(1000, 500), dtype=np.uint8)
        result = max_gap(A)
        assert result.shape == (500,)
        assert result.dtype == np.int64
        # Each max gap must be <= T
        assert result.max() <= 1000

    def test_axis1(self):
        # row 0: [0,0,1,0] → max gap = 2; row 1: [1,1,1,1] → max gap = 0
        A = np.array([[0, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
        result = max_gap(A, axis=1)
        np.testing.assert_array_equal(result, [2, 0])

    def test_non_binary_values_treated_as_covered(self):
        # Values > 1 should count as "covered" (nonzero)
        A = np.array([[0, 5, 0, 3, 0]], dtype=np.uint8).T
        result = max_gap(A)
        # gaps: [1], [1], [1] — all length 1
        np.testing.assert_array_equal(result, [1])


# ── gap_count ─────────────────────────────────────────────────────────────────

class TestGapCount:
    def test_simple(self):
        A = make_matrix(
            [0, 0, 1, 0, 1, 0],  # 3 gaps
            [1, 1, 1, 1, 1, 1],  # 0 gaps
            [0, 0, 0, 0, 0, 0],  # 1 gap
        )
        result = gap_count(A)
        np.testing.assert_array_equal(result, [3, 0, 1])

    def test_dtype_int64(self):
        A = np.zeros((5, 3), dtype=np.uint8)
        assert gap_count(A).dtype == np.int64

    def test_gap_count_matches_gap_lengths_len(self):
        rng = np.random.default_rng(7)
        A = rng.integers(0, 3, size=(200, 50), dtype=np.uint8)
        counts = gap_count(A)
        lengths = gap_lengths(A)
        for c in range(50):
            assert counts[c] == len(lengths[c]), f"col {c}"


# ── gap_count_and_max ─────────────────────────────────────────────────────────

class TestGapCountAndMax:
    def test_matches_individual_functions(self):
        rng = np.random.default_rng(99)
        A = rng.integers(0, 2, size=(300, 80), dtype=np.uint8)
        counts, maxima = gap_count_and_max(A)
        np.testing.assert_array_equal(counts, gap_count(A))
        np.testing.assert_array_equal(maxima, max_gap(A))

    def test_return_shapes(self):
        A = np.zeros((10, 5), dtype=np.uint8)
        counts, maxima = gap_count_and_max(A)
        assert counts.shape == (5,)
        assert maxima.shape == (5,)


# ── total_gap ─────────────────────────────────────────────────────────────────

class TestTotalGap:
    def test_simple(self):
        A = make_matrix(
            [0, 0, 0, 1, 0],  # 4 zeros
            [1, 1, 1, 1, 1],  # 0 zeros
            [0, 0, 0, 0, 0],  # 5 zeros
        )
        result = total_gap(A)
        np.testing.assert_array_equal(result, [4, 0, 5])

    def test_equals_column_zero_sum(self):
        rng = np.random.default_rng(3)
        A = rng.integers(0, 4, size=(100, 30), dtype=np.uint8)
        expected = (A == 0).sum(axis=0).astype(np.int64)
        np.testing.assert_array_equal(total_gap(A), expected)

    def test_dtype_int64(self):
        A = np.zeros((5, 3), dtype=np.uint8)
        assert total_gap(A).dtype == np.int64

    def test_axis1(self):
        A = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        # row 0: 2 zeros; row 1: 2 zeros
        result = total_gap(A, axis=1)
        np.testing.assert_array_equal(result, [2, 2])


# ── cross-consistency ─────────────────────────────────────────────────────────

class TestCrossConsistency:
    def test_total_gap_equals_sum_of_gap_lengths(self):
        rng = np.random.default_rng(11)
        A = rng.integers(0, 3, size=(150, 40), dtype=np.uint8)
        totals = total_gap(A)
        lengths = gap_lengths(A)
        for c in range(40):
            expected = int(lengths[c].sum()) if lengths[c].size else 0
            assert totals[c] == expected, f"col {c}"

    def test_max_gap_le_total_gap(self):
        rng = np.random.default_rng(13)
        A = rng.integers(0, 2, size=(200, 60), dtype=np.uint8)
        assert (max_gap(A) <= total_gap(A)).all()

    def test_empty_matrix(self):
        A = np.zeros((0, 5), dtype=np.uint8)
        assert max_gap(A).shape == (5,)
        assert gap_count(A).shape == (5,)
        assert total_gap(A).shape == (5,)
        np.testing.assert_array_equal(max_gap(A), np.zeros(5))
        np.testing.assert_array_equal(gap_count(A), np.zeros(5))
        np.testing.assert_array_equal(total_gap(A), np.zeros(5))

    def test_single_row_zero(self):
        A = np.zeros((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(max_gap(A), [1, 1, 1, 1])
        np.testing.assert_array_equal(gap_count(A), [1, 1, 1, 1])
        np.testing.assert_array_equal(total_gap(A), [1, 1, 1, 1])

    def test_single_row_nonzero(self):
        A = np.ones((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(max_gap(A), [0, 0, 0, 0])
        np.testing.assert_array_equal(gap_count(A), [0, 0, 0, 0])
        np.testing.assert_array_equal(total_gap(A), [0, 0, 0, 0])


# ── error handling ────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_3d_raises(self):
        A = np.zeros((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            gap_lengths(A)

    def test_invalid_axis_raises(self):
        A = np.zeros((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            gap_lengths(A, axis=2)
