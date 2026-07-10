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
