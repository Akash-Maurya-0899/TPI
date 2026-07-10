# coding: utf-8

#  Copyright (C) 2026, Michael Puerrer, Jonathan Blackman.
#
#  This file is part of TPI.
#
#  TPI is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  TPI is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with TPI.  If not, see <http://www.gnu.org/licenses/>.
#

"""Banded assembly and solve for TPI's not-a-knot cubic spline systems.

The collocation matrix assembled by ``BsplineBasis1D.AssembleSplineMatrix`` is
banded with bandwidths (kl, ku) = (4, 4): each interior row holds the (at most
4) active cubic B-spline basis values at one node, and the two not-a-knot
boundary rows hold 5 entries each in the leading/trailing corner. This module
assembles that matrix directly in LAPACK banded storage and solves with
``scipy.linalg.solve_banded``, replacing the dense (n+2)^2 assembly and
explicit inverse: O(n) memory and time per axis instead of O(n^2)/O(n^3).

Both backends share this module so their coefficient numerics agree.
"""

import numpy as np
from scipy.linalg import solve_banded

# Bandwidths of the not-a-knot collocation matrix.
KL = 4
KU = 4

# Axes with N = n + 2 at or below this use a cached dense inverse applied as
# a single matrix product per solve, exactly like the historical dense
# implementation: LAPACK's banded solve processes right-hand sides column by
# column plus pivot row swaps, which is memory-bound for the many-RHS solves
# of small ND grid axes, while the inverse-times-RHS is one BLAS-3 gemm.
# Larger axes use the O(n)-memory banded solve, where a dense inverse would
# need O(n^2) memory.
DENSE_SOLVE_MAX_N = 512


def construct_knots(nodes):
    """Clamped cubic knot vector: endpoints with multiplicity 3 around the nodes."""
    nodes = np.asarray(nodes, dtype=np.float64)
    return np.concatenate((nodes[:1].repeat(3), nodes, nodes[-1:].repeat(3)))


def active_basis_at_nodes(nodes):
    """The 4 active cubic B-spline basis values at every node, vectorized.

    Nodes double as knots, so the span of node i is known by position
    (no search): span = i + 3, with the last node belonging to the final
    non-empty interval. Returns (basis4, starts) where basis4[i] are the
    values of basis functions starts[i]..starts[i]+3 at nodes[i].
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    knots = construct_knots(nodes)
    spans = np.minimum(np.arange(n) + 3, n + 1)

    # The 6 knots around each span, then the cubic de Boor recurrence unrolled
    # on length-n vectors; 0/0 from repeated boundary knots is guarded to 0.
    k = knots[spans[:, None] + np.arange(-2, 4)[None, :]]
    l1 = nodes - k[:, 2]
    l2 = nodes - k[:, 1]
    l3 = nodes - k[:, 0]
    r1 = k[:, 3] - nodes
    r2 = k[:, 4] - nodes
    r3 = k[:, 5] - nodes

    def _safe_div(num, den):
        out = np.zeros_like(num)
        np.divide(num, den, out=out, where=den != 0.0)
        return out

    b0 = _safe_div(r1, r1 + l1)
    b1 = _safe_div(l1, r1 + l1)
    q0 = r1 * _safe_div(b0, r1 + l2)
    q1 = l2 * _safe_div(b0, r1 + l2) + r2 * _safe_div(b1, r2 + l1)
    q2 = l1 * _safe_div(b1, r2 + l1)
    d0 = r1 * _safe_div(q0, r1 + l3)
    d1 = l3 * _safe_div(q0, r1 + l3) + r2 * _safe_div(q1, r2 + l2)
    d2 = l2 * _safe_div(q1, r2 + l2) + r3 * _safe_div(q2, r3 + l1)
    d3 = l1 * _safe_div(q2, r3 + l1)

    basis4 = np.stack((d0, d1, d2, d3), axis=1)
    return basis4, spans - 3


def _cubic_bspline_3rd_derivs_active(knots, x):
    """3rd derivatives of the 4 active cubic B-splines at scalar x.

    Standard derivative recurrence (The NURBS Book, DersBasisFuns) at degree 3,
    matching the GSL and JAX-backend conventions. Returns (values, start) where
    values are the 3rd derivatives of basis functions start..start+3.
    """
    p = 3
    span = int(np.searchsorted(knots[3:-3], x, side="right")) + 2
    span = min(max(span, 3), len(knots) - 5)

    ndu = np.zeros((p + 1, p + 1))
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    for j in range(1, p + 1):
        left[j] = x - knots[span + 1 - j]
        right[j] = knots[span + j] - x
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r] if ndu[j, r] != 0.0 else 0.0
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    ders3 = np.zeros(p + 1)
    for r in range(p + 1):
        a = np.zeros((2, p + 1))
        a[0, 0] = 1.0
        s1, s2 = 0, 1
        d = 0.0
        for k in range(1, p + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk] if ndu[pk + 1, rk] != 0.0 else 0.0
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1, rk + j]
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / denom if denom != 0.0 else 0.0
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r] if ndu[pk + 1, r] != 0.0 else 0.0
                d += a[s2, k] * ndu[r, pk]
            s1, s2 = s2, s1
        ders3[r] = d
    # p! / (p - 3)! scaling for the 3rd derivative
    return ders3 * 6.0, span - 3


def notaknot_boundary_rows(nodes):
    """The two not-a-knot rows of the collocation matrix (full length n + 2).

    Row 0 and row n+1: differences of 3rd-derivative basis values at the
    midpoints of the first/last two intervals, imposing 3rd-derivative
    continuity across the 2nd and penultimate nodes.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    knots = construct_knots(nodes)
    N = nodes.shape[0] + 2

    def full_row(x):
        values, start = _cubic_bspline_3rd_derivs_active(knots, x)
        row = np.zeros(N)
        row[start:start + 4] = values
        return row

    row_first = full_row(0.5 * (nodes[0] + nodes[1])) - full_row(0.5 * (nodes[1] + nodes[2]))
    row_last = full_row(0.5 * (nodes[-3] + nodes[-2])) - full_row(0.5 * (nodes[-2] + nodes[-1]))
    return row_first, row_last


def assemble_spline_matrix_banded(nodes):
    """Banded not-a-knot collocation matrix with self-contained boundary rows."""
    row_first, row_last = notaknot_boundary_rows(nodes)
    return assemble_banded_ab(nodes, row_first, row_last)


def assemble_banded_ab(nodes, row_first, row_last):
    """Assemble the not-a-knot collocation matrix in LAPACK banded storage.

    Arguments:
      * nodes: 1D array of n grid nodes (non-decreasing).
      * row_first, row_last: the two not-a-knot boundary rows (3rd-derivative
        differences) as produced for row 0 and row n+1 of the dense matrix;
        only entries 0..4 of row_first and N-5..N-1 of row_last are nonzero.

    Returns ab of shape (KL + KU + 1, n + 2) with ab[KU + i - j, j] = A[i, j],
    suitable for scipy.linalg.solve_banded((KL, KU), ab, rhs).
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    N = n + 2
    row_first = np.asarray(row_first, dtype=np.float64)
    row_last = np.asarray(row_last, dtype=np.float64)

    basis4, starts = active_basis_at_nodes(nodes)

    ab = np.zeros((KL + KU + 1, N))
    cols_first = np.arange(5)
    ab[KU - cols_first, cols_first] = row_first[:5]
    cols_last = np.arange(N - 5, N)
    ab[KU + (N - 1) - cols_last, cols_last] = row_last[N - 5:]
    rows = np.arange(1, n + 1)
    for offset in range(4):
        cols = starts + offset
        ab[KU + rows - cols, cols] = basis4[:, offset]
    return ab


def solve_banded_axis(ab, tensor, axis):
    """Solve the banded system along one tensor axis, keeping axis order."""
    tensor = np.asarray(tensor, dtype=np.float64)
    moved = np.moveaxis(tensor, axis, 0)
    solved = solve_banded(
        (KL, KU), ab, moved.reshape(moved.shape[0], -1), check_finite=False
    )
    return np.moveaxis(solved.reshape(moved.shape), 0, axis)


def hermite_polynomial_pieces(x, f, s):
    """Per-interval cubic coefficients from node values f and derivatives s.

    Returns (c0, c1, c2, c3), each of length n - 1, such that the spline on
    [x_i, x_{i+1}] is c0[i] + t*(c1[i] + t*(c2[i] + t*c3[i])) with t = x - x_i.
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    dx = np.diff(x)
    slope = np.diff(f) / dx
    c0 = f[:-1]
    c1 = s[:-1]
    c2 = (3.0 * slope - 2.0 * s[:-1] - s[1:]) / dx
    c3 = (s[:-1] + s[1:] - 2.0 * slope) / dx ** 2
    return c0, c1, c2, c3


def hermite_to_bspline_coefficients(x, c0, c1, c2, c3):
    """Standard TPI B-form coefficients (shape (n + 2,)) of a piecewise cubic.

    Uses the polar form (blossom) of the cubic pieces: coefficient j equals
    the blossom evaluated at knots t_{j+1}, t_{j+2}, t_{j+3}, computed on a
    polynomial piece whose interval touches those knots.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    t = construct_knots(x)
    j = np.arange(n + 2)
    piece = np.clip(j, 3, n + 1) - 3
    u = t[j + 1] - x[piece]
    v = t[j + 2] - x[piece]
    w = t[j + 3] - x[piece]
    p0 = np.asarray(c0)[piece]
    p1 = np.asarray(c1)[piece]
    p2 = np.asarray(c2)[piece]
    p3 = np.asarray(c3)[piece]
    return (
        p0
        + p1 * (u + v + w) / 3.0
        + p2 * (u * v + u * w + v * w) / 3.0
        + p3 * (u * v * w)
    )


def bspline_to_hermite(x, coeffs):
    """Exact node values and first derivatives of a B-form cubic spline.

    Works for arbitrary coefficient vectors on the TPI knot vector (any C^2
    cubic spline in B-form), not only not-a-knot interpolants: values come
    from the 4 active cubic basis functions per node, derivatives from the
    quadratic B-form of the derivative spline. Requires strictly increasing
    nodes.
    """
    x = np.asarray(x, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = x.shape[0]
    N = n + 2
    t = construct_knots(x)

    basis4, starts = active_basis_at_nodes(x)
    idx = starts[:, None] + np.arange(4)
    f = np.sum(basis4 * coeffs[idx], axis=1)

    # Derivative spline: quadratic B-form with coefficients e_j (j = 1..N-1)
    e = np.zeros(N)
    jj = np.arange(1, N)
    e[1:] = 3.0 * (coeffs[1:] - coeffs[:-1]) / (t[jj + 3] - t[jj])

    # The 3 active quadratic basis values at each node (de Boor, degree 2)
    spans = np.minimum(np.arange(n) + 3, n + 1)
    k = t[spans[:, None] + np.arange(-1, 3)]
    l1 = x - k[:, 1]
    l2 = x - k[:, 0]
    r1 = k[:, 2] - x
    r2 = k[:, 3] - x
    b0 = r1 / (r1 + l1)
    b1 = l1 / (r1 + l1)
    q0 = r1 * b0 / (r1 + l2)
    q1 = l2 * b0 / (r1 + l2) + r2 * b1 / (r2 + l1)
    q2 = l1 * b1 / (r2 + l1)

    qidx = spans[:, None] + np.arange(-2, 1)
    s = (
        q0 * e[qidx[:, 0]]
        + q1 * e[qidx[:, 1]]
        + q2 * e[qidx[:, 2]]
    )
    return f, s


def _banded_to_dense(ab, N):
    """Expand LAPACK banded storage back to the dense (N, N) matrix."""
    A = np.zeros((N, N))
    for offset in range(-KL, KU + 1):
        j = np.arange(max(0, -offset), min(N, N - offset))
        A[j + offset, j] = ab[KU + offset, j]
    return A


def factor_collocation_matrix(nodes, row_first=None, row_last=None):
    """Factor the collocation matrix for repeated solves on one node array.

    Returns ("dense_inv", A_inv) for small axes (one gemm per solve, matching
    the historical dense implementation) or ("banded", ab) for large axes
    (O(n) memory). If the boundary rows are omitted they are computed with
    notaknot_boundary_rows.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    if row_first is None or row_last is None:
        row_first, row_last = notaknot_boundary_rows(nodes)
    ab = assemble_banded_ab(nodes, row_first, row_last)
    N = nodes.shape[0] + 2
    if N <= DENSE_SOLVE_MAX_N:
        return ("dense_inv", np.linalg.inv(_banded_to_dense(ab, N)))
    return ("banded", ab)


def solve_collocation_axis(factor, tensor, axis):
    """Solve a factored collocation system along one tensor axis."""
    kind, data = factor
    tensor = np.asarray(tensor, dtype=np.float64)
    moved = np.moveaxis(tensor, axis, 0)
    rhs = moved.reshape(moved.shape[0], -1)
    if kind == "dense_inv":
        solved = data @ rhs
    else:
        solved = solve_banded((KL, KU), data, rhs, check_finite=False)
    return np.moveaxis(solved.reshape(moved.shape), 0, axis)
