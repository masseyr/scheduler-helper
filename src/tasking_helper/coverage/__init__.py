"""
tasking_helper.coverage — fast gap analysis for coverage matrices.

A *coverage matrix* is a 2-D array of non-negative integers where 0 means
"no coverage" and any positive value means "covered".  A *gap* is a maximal
contiguous run of zeros in a single column.

Public API
----------
gap_lengths(A, axis=0) -> list[np.ndarray]
    All individual gap lengths for each column (or row if axis=1).

max_gap(A, axis=0) -> np.ndarray
    Maximum gap length per column.

gap_count(A, axis=0) -> np.ndarray
    Number of gaps per column.

total_gap(A, axis=0) -> np.ndarray
    Total number of zero cells per column (sum of all gap lengths).

Implementation note
-------------------
This module provides a pure-NumPy implementation that is always available.
If the optional C extension ``_gap_ext`` has been compiled and installed
(run ``python setup_ext.py build_ext --inplace`` inside this package
directory), it is used automatically for ``max_gap``, ``gap_count``, and
``total_gap`` — typically 5-10× faster on large matrices.
"""

from __future__ import annotations

import numpy as np

# ── Try to import the optional C extension ────────────────────────────────────
try:
    from tasking_helper.coverage import _gap_ext as _ext  # type: ignore[import]
    _HAS_EXT = True
except ImportError:
    _HAS_EXT = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _zero_mask(A: np.ndarray) -> np.ndarray:
    """Return a uint8 array: 1 where A == 0, else 0."""
    return (A == 0).view(np.uint8)


def _ensure_2d_colwise(A: np.ndarray, axis: int) -> np.ndarray:
    """
    Return a 2-D array whose *columns* are the sequences to analyse.

    axis=0  → columns of A  (default: time runs along rows)
    axis=1  → rows    of A  (transposed so rows become columns)
    """
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, np.newaxis]
    elif A.ndim != 2:
        raise ValueError("A must be 1-D or 2-D")
    if axis == 1:
        A = A.T
    elif axis != 0:
        raise ValueError("axis must be 0 or 1")
    return A


# ─────────────────────────────────────────────────────────────────────────────
# NumPy implementation (always available)
# ─────────────────────────────────────────────────────────────────────────────

def _gap_lengths_numpy(mask2d: np.ndarray) -> list[np.ndarray]:
    """
    Return gap lengths per column for a pre-built uint8 zero-mask.

    Parameters
    ----------
    mask2d : np.ndarray, shape (T, N), dtype uint8
        1 where the original matrix is zero, 0 elsewhere.

    Returns
    -------
    list of length N; each element is an int32 array of gap lengths.
    """
    T, N = mask2d.shape

    # Pad with a zero row above and below so every gap is bounded.
    padded = np.zeros((T + 2, N), dtype=np.int8)
    padded[1 : T + 1] = mask2d.view(np.int8)

    diff = np.diff(padded, axis=0)           # shape (T+1, N)
    # np.where returns indices in row-major order, so starts/ends from
    # different columns are interleaved.  We must sort by (column, row)
    # before pairing them so that the k-th start pairs with the k-th end.
    starts_r, starts_c = np.where(diff == 1)  # gap opens  (0→1 transition)
    ends_r,   ends_c   = np.where(diff == -1) # gap closes (1→0 transition)

    # Sort both sets by (column, row) — lexsort keys are in reverse priority.
    s_order = np.lexsort((starts_r, starts_c))
    e_order = np.lexsort((ends_r,   ends_c))
    starts_r = starts_r[s_order]; starts_c = starts_c[s_order]
    ends_r   = ends_r[e_order]

    lengths = (ends_r - starts_r).astype(np.int32)

    if N == 0:
        return []

    # starts_c is now sorted; find where each column's block begins.
    split_points = np.searchsorted(starts_c, np.arange(1, N))
    return list(np.split(lengths, split_points))


def _gap_count_numpy(mask2d: np.ndarray) -> np.ndarray:
    lists = _gap_lengths_numpy(mask2d)
    return np.array([len(l) for l in lists], dtype=np.int64)


# Vectorised max_gap — sort-based, no Python loop over columns
def _max_gap_numpy_fast(mask2d: np.ndarray) -> np.ndarray:
    T, N = mask2d.shape
    padded = np.zeros((T + 2, N), dtype=np.int8)
    padded[1 : T + 1] = mask2d.view(np.int8)
    diff = np.diff(padded, axis=0)
    starts_r, starts_c = np.where(diff == 1)
    ends_r,   ends_c   = np.where(diff == -1)

    # Sort both by (column, row) before pairing.
    s_order = np.lexsort((starts_r, starts_c))
    e_order = np.lexsort((ends_r,   ends_c))
    starts_c_s = starts_c[s_order]
    lengths = (ends_r[e_order] - starts_r[s_order]).astype(np.int64)

    out = np.zeros(N, dtype=np.int64)
    np.maximum.at(out, starts_c_s, lengths)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def gap_lengths(A, axis: int = 0) -> list[np.ndarray]:
    """
    Return all individual gap lengths for each column (or row).

    Parameters
    ----------
    A : array-like, shape (T, N)
        Coverage matrix of non-negative integers.
    axis : {0, 1}
        0 (default) → analyse along rows, one result per column.
        1           → analyse along columns, one result per row.

    Returns
    -------
    list of np.ndarray (dtype int32)
        Element ``i`` contains the lengths of every gap in column ``i``.
        Empty array if column ``i`` has no zeros.
    """
    A = _ensure_2d_colwise(np.asarray(A), axis)
    return _gap_lengths_numpy(_zero_mask(A))


def max_gap(A, axis: int = 0) -> np.ndarray:
    """
    Maximum gap length per column (or row).

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        0 for columns with no zeros.
    """
    A = _ensure_2d_colwise(np.asarray(A), axis)
    mask = _zero_mask(A)
    if _HAS_EXT:
        return _ext.max_gap(mask)  # type: ignore[union-attr]
    return _max_gap_numpy_fast(mask)


def gap_count(A, axis: int = 0) -> np.ndarray:
    """
    Number of gaps per column (or row).

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
    """
    A = _ensure_2d_colwise(np.asarray(A), axis)
    mask = _zero_mask(A)
    if _HAS_EXT:
        return _ext.gap_count(mask)  # type: ignore[union-attr]
    return _gap_count_numpy(mask)


def gap_count_and_max(A, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (gap_count, max_gap) in a single pass (more efficient).

    Returns
    -------
    counts : np.ndarray, shape (N,), dtype int64
    maxima : np.ndarray, shape (N,), dtype int64
    """
    A = _ensure_2d_colwise(np.asarray(A), axis)
    mask = _zero_mask(A)
    if _HAS_EXT:
        return _ext.gap_count_and_max(mask)  # type: ignore[union-attr]
    lists = _gap_lengths_numpy(mask)
    N = mask.shape[1]
    counts = np.array([len(l) for l in lists], dtype=np.int64)
    maxima = np.zeros(N, dtype=np.int64)
    for col, lens in enumerate(lists):
        if lens.size:
            maxima[col] = int(lens.max())
    return counts, maxima


def total_gap(A, axis: int = 0) -> np.ndarray:
    """
    Total number of zero cells per column (sum of all gap lengths).

    This is simply the column sum of the zero-mask and is very fast.

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
    """
    A = _ensure_2d_colwise(np.asarray(A), axis)
    return _zero_mask(A).sum(axis=0).astype(np.int64)


__all__ = [
    "gap_lengths",
    "max_gap",
    "gap_count",
    "gap_count_and_max",
    "total_gap",
]
