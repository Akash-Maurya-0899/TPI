# coding: utf-8

#  Copyright (C) 2017, Michael Pürrer, Jonathan Blackman.
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

# A py.test compliant unit test

#!/usr/bin/env python

"""
TPI.pyx

designed to be run-able with pytest

MP 02/2017
"""

import pytest
import numpy as np
import os
import sys
import inspect
import time
import jax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import TPI
import TPI_jax


def _reference_find_active_cubic_bspline_span_batched(knots, xs):
    knots = np.asarray(knots, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)

    span = np.searchsorted(knots[3:-3], xs, side="right") + 2
    span = np.clip(span, 3, knots.shape[0] - 5)

    lefts = np.stack(
        (
            xs - knots[span],
            xs - knots[span - 1],
            xs - knots[span - 2],
        ),
        axis=-1,
    )
    rights = np.stack(
        (
            knots[span + 1] - xs,
            knots[span + 2] - xs,
            knots[span + 3] - xs,
        ),
        axis=-1,
    )

    basis = np.zeros((xs.shape[0], 4), dtype=np.float64)
    basis[:, 0] = 1.0
    for j in range(3):
        saved = np.zeros(xs.shape[0], dtype=np.float64)
        for r in range(j + 1):
            denom = rights[:, r] + lefts[:, j - r]
            temp = np.where(denom != 0.0, basis[:, r] / denom, 0.0)
            basis[:, r] = saved + rights[:, r] * temp
            saved = lefts[:, j - r] * temp
        basis[:, j + 1] = saved

    return basis, span - 3


def _span_case_data():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0], dtype=np.float64)
    x2 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64)
    y2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64)
    z3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0], dtype=np.float64)
    x4 = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    return [
        ("1D", TPI_jax.construct_knots(x1)),
        ("2D-x", TPI_jax.construct_knots(x2)),
        ("2D-y", TPI_jax.construct_knots(y2)),
        ("3D-z", TPI_jax.construct_knots(z3)),
        ("4D-x4", TPI_jax.construct_knots(x4)),
    ]

def test_BsplineBasis1D():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
    b = TPI.BsplineBasis1D(x1)
    atol = np.finfo(float).eps

    # Test B-spline basis functions at fixed x
    b_eval_Mma = np.array([0, 0.00210526315789473, 0.2988222605694564, 0.6262726488352028, 0.07279982743744609, 0, 0, 0])
    b_eval_TPI = b.EvaluateBsplines(4.7)
    assert np.array_equal(b_eval_Mma, b_eval_TPI)

    # Test 3rd derivative of B-spline basis functions at fixed x
    Db_eval_Mma = np.array([0., -0.19736842105263172, 0.4562122519413291, -0.38826574633304595, 0.1294219154443486, 0., 0., 0.])
    Db_eval_TPI = b.EvaluateBsplines3rdDerivatives(4.7)
    # print 'Db_eval_Mma - Db_eval_TPI', Db_eval_Mma - Db_eval_TPI
    assert np.allclose(Db_eval_Mma, Db_eval_TPI, atol=atol, rtol=0)


def test_BsplineBasis1D_fail():
    with pytest.raises(ValueError):
        x2 = np.array([1.1, -3.2, 5.1, 7.2, 9.3, 12])
        TPI.BsplineBasis1D(x2)

    with pytest.raises(ValueError):
        x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
        b = TPI.BsplineBasis1D(x1)
        b_eval_TPI = b.EvaluateBsplines(-4.7)

    with pytest.raises(ValueError):
        x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
        b = TPI.BsplineBasis1D(x1)
        b_eval_TPI = b.EvaluateBsplines3rdDerivatives(-4.7)

    with pytest.raises(ValueError):
        TPI.BsplineBasis1D(None)

    with pytest.raises(ValueError):
        x3 = np.array([1.1])
        TPI.BsplineBasis1D(x3)

    with pytest.raises(ValueError):
        x3 = np.array([np.nan, np.nan])
        TPI.BsplineBasis1D(x3)

    with pytest.raises(ValueError):
        x3 = np.array([1.1, np.inf, 5.1, 7.2, 9.3, 12])
        TPI.BsplineBasis1D(x3)

    with pytest.raises(ValueError):
        x2 = np.array([1.1, 3.4+2.1j, 5.1, 7.2, 9.3, 12])
        TPI.BsplineBasis1D(x2)

    with pytest.raises(ValueError):
        x2 = np.arange(12, dtype=np.float64).reshape((3,2,2))
        TPI.BsplineBasis1D(x2)

    # with pytest.raises(ValueError):
    #     x2 = np.array([1.1, "-3.4", 5.1, 7.2, 9.3, 12])
    #     TPI.BsplineBasis1D(x2)


def test_jax_construct_knots_matches_gsl():
    grids_and_knots = [
        (
            np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0]),
            np.array([1.1, 1.1, 1.1, 1.1, 3.2, 5.1, 7.2, 9.3, 12.0, 12.0, 12.0, 12.0]),
        ),
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25, 0.25, 0.25, 0.25]),
        ),
        (
            np.array([-1.0, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0]),
        ),
    ]

    for nodes, expected_knots in grids_and_knots:
        knots = TPI_jax.construct_knots(nodes)
        assert np.array_equal(np.asarray(knots), expected_knots)
        assert np.asarray(knots).dtype == np.float64


def test_jax_find_active_cubic_bspline_span_matches_searchsorted_dense_grid():
    diagnostics = []
    for case_label, knots in _span_case_data():
        xs = np.linspace(float(knots[3]), float(knots[-4]), 30, dtype=np.float64)

        jax_eval = jax.jit(jax.vmap(lambda x: TPI_jax._find_active_cubic_bspline_span_jax(knots, x)))
        # warmup: trigger JIT compilation before assertions
        warmup_basis, warmup_start = jax_eval(xs)

        actual_basis, actual_start = jax_eval(xs)
        expected_basis, expected_start = _reference_find_active_cubic_bspline_span_batched(knots, xs)

        actual_basis = np.asarray(actual_basis)
        actual_start = np.asarray(actual_start)
        expected_basis = np.asarray(expected_basis)
        expected_start = np.asarray(expected_start)

        assert np.array_equal(np.asarray(warmup_basis), actual_basis)
        assert np.array_equal(np.asarray(warmup_start), actual_start)
        assert np.array_equal(actual_start, expected_start)
        assert np.allclose(actual_basis, expected_basis, atol=1e-14, rtol=0)

        x_repeated = np.repeat(xs, actual_basis.shape[1])
        basis_indices = np.tile(np.arange(actual_basis.shape[1]), xs.shape[0])
        diagnostics.extend(
            [
                (case_label, x_repeated[idx], basis_indices[idx], actual_basis.reshape(-1)[idx], expected_basis.reshape(-1)[idx])
                for idx in range(actual_basis.size)
            ]
        )

    actual_flat = np.array([entry[3] for entry in diagnostics], dtype=np.float64)
    expected_flat = np.array([entry[4] for entry in diagnostics], dtype=np.float64)
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_x, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_case, max_rel_x, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at x={max_abs_x} (case={max_abs_case}, basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at x={max_rel_x} (case={max_rel_case}, basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-14, rtol=0)


def test_jax_find_active_cubic_bspline_span_boundary_points():
    diagnostics = []
    for case_label, knots in _span_case_data():
        xs = np.asarray(knots[3:-3], dtype=np.float64)
        if xs.size > 10:
            sample_idx = np.unique(np.linspace(0, xs.size - 1, 10, dtype=int))
            xs = xs[sample_idx]

        jax_eval = jax.jit(jax.vmap(lambda x: TPI_jax._find_active_cubic_bspline_span_jax(knots, x)))
        # warmup: trigger JIT compilation before assertions
        warmup_basis, warmup_start = jax_eval(xs)

        actual_basis, actual_start = jax_eval(xs)
        expected_basis, expected_start = _reference_find_active_cubic_bspline_span_batched(knots, xs)

        actual_basis = np.asarray(actual_basis)
        actual_start = np.asarray(actual_start)
        expected_basis = np.asarray(expected_basis)
        expected_start = np.asarray(expected_start)

        assert np.array_equal(np.asarray(warmup_basis), actual_basis)
        assert np.array_equal(np.asarray(warmup_start), actual_start)
        assert np.array_equal(actual_start, expected_start)
        assert np.all((0 <= actual_start) & (actual_start <= len(knots) - 8))
        assert np.allclose(actual_basis.sum(axis=1), 1.0, atol=1e-14, rtol=0)
        assert np.allclose(actual_basis, expected_basis, atol=1e-14, rtol=0)

        x_repeated = np.repeat(xs, actual_basis.shape[1])
        basis_indices = np.tile(np.arange(actual_basis.shape[1]), xs.shape[0])
        diagnostics.extend(
            [
                (case_label, x_repeated[idx], basis_indices[idx], actual_basis.reshape(-1)[idx], expected_basis.reshape(-1)[idx])
                for idx in range(actual_basis.size)
            ]
        )

    actual_flat = np.array([entry[3] for entry in diagnostics], dtype=np.float64)
    expected_flat = np.array([entry[4] for entry in diagnostics], dtype=np.float64)
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_x, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_case, max_rel_x, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at x={max_abs_x} (case={max_abs_case}, basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at x={max_rel_x} (case={max_rel_case}, basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-14, rtol=0)


def test_jax_EvaluateBsplines_matches_gsl():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    cases = [
        (
            1.1,
            np.array([1., 0., 0., 0., 0., 0., 0., 0.]),
        ),
        (
            2.0,
            np.array([
                0.1865889212827989,
                0.5871483236151605,
                0.21203558882569423,
                0.014227166276346603,
                0.,
                0.,
                0.,
                0.,
            ]),
        ),
        (
            4.7,
            np.array([
                0.,
                0.00210526315789473,
                0.2988222605694564,
                0.6262726488352028,
                0.07279982743744609,
                0.,
                0.,
                0.,
            ]),
        ),
        (
            9.3,
            np.array([
                0.,
                0.,
                0.,
                0.,
                0.22010869565217378,
                0.588485054347826,
                0.1914062500000001,
                0.,
            ]),
        ),
        (
            12.0,
            np.array([0., 0., 0., 0., 0., 0., 0., 1.]),
        ),
        (
            3.5,
            np.array([
                0.,
                0.13473684210526317,
                0.59716566005176897,
                0.26751509922346856,
                0.00058239861949956767,
                0.,
                0.,
                0.,
            ]),
        ),
        (
            5.05,
            np.array([
                0.,
                4.1118421052631171e-06,
                0.19394541361087153,
                0.66947530198446947,
                0.13657517256255389,
                0.,
                0.,
                0.,
            ]),
        ),
        (
            5.15,
            np.array([
                0.,
                0.,
                0.16813280640124892,
                0.67193343184268239,
                0.15993170779438731,
                2.05396168129096e-06,
                0.,
                0.,
            ]),
        ),
        (
            6.3,
            np.array([
                0.,
                0.,
                0.014227166276346617,
                0.44628638340582144,
                0.5110924840356669,
                0.028393966282165034,
                0.,
                0.,
            ]),
        ),
        (
            8.0,
            np.array([
                0.,
                0.,
                0.,
                0.040834913200252815,
                0.5825999679986036,
                0.365983108219133,
                0.010582010582010573,
                0.,
            ]),
        ),
        (
            10.5,
            np.array([
                0.,
                0.,
                0.,
                0.,
                0.037741545893719815,
                0.33238769793344075,
                0.5420792609739369,
                0.08779149519890253,
            ]),
        ),
    ]

    diagnostics = []

    for case_index, (x, expected) in enumerate(cases):
        assert abs(expected.sum() - 1.0) < 1e-12
        actual = np.asarray(basis.EvaluateBsplines(x))
        assert actual.shape == expected.shape
        diagnostics.extend(
            [
                (case_index, x, basis_index, actual[basis_index], expected[basis_index])
                for basis_index in range(expected.size)
            ]
        )

    actual_flat = np.array([entry[3] for entry in diagnostics])
    expected_flat = np.array([entry[4] for entry in diagnostics])
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_x, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_case, max_rel_x, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at case {max_abs_case} (x={max_abs_x}, basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at case {max_rel_case} (x={max_rel_x}, basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_EvaluateBsplines_jit_smoke():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    x = 4.7

    non_jit = np.asarray(basis.EvaluateBsplines(x))
    jit_eval = np.asarray(jax.jit(basis.EvaluateBsplines)(x))
    assert np.allclose(jit_eval, non_jit, atol=1e-10, rtol=0)


def test_jax_EvaluateBsplines3rdDerivatives_matches_gsl():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    cases = [
        (
            2.15,
            np.array([
                -0.647878198898607,
                1.1665856818918043,
                -0.6358035017285604,
                0.11709601873536303,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
        ),
        (
            4.15,
            np.array([
                0.0,
                -0.19736842105263175,
                0.45621225194132903,
                -0.3882657463330459,
                0.1294219154443486,
                0.0,
                0.0,
                0.0,
            ]),
        ),
        (
            6.15,
            np.array([
                0.0,
                0.0,
                -0.11709601873536298,
                0.34571205531392873,
                -0.32720619728052763,
                0.09859016070196189,
                0.0,
                0.0,
            ]),
        ),
        (
            8.25,
            np.array([
                0.0,
                0.0,
                0.0,
                -0.11152001784320277,
                0.2963765691593813,
                -0.308864487824115,
                0.12400793650793648,
                0.0,
            ]),
        ),
        (
            10.65,
            np.array([
                0.0,
                0.0,
                0.0,
                0.0,
                -0.06709608158883523,
                0.3350144629331426,
                -0.5727499618960528,
                0.30483158055174536,
            ]),
        ),
    ]

    diagnostics = []

    for case_index, (x, expected) in enumerate(cases):
        actual = np.asarray(basis.EvaluateBsplines3rdDerivatives(x))
        assert actual.shape == expected.shape
        diagnostics.extend(
            [
                (case_index, x, basis_index, actual[basis_index], expected[basis_index])
                for basis_index in range(expected.size)
            ]
        )

    actual_flat = np.array([entry[3] for entry in diagnostics])
    expected_flat = np.array([entry[4] for entry in diagnostics])
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_x, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_case, max_rel_x, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at case {max_abs_case} (x={max_abs_x}, basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at case {max_rel_case} (x={max_rel_x}, basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_EvaluateBsplines3rdDerivatives_jit_across_spans():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    points = [1.5, 3.5, 5.05, 5.15, 6.3, 8.0, 10.5]
    jit_eval_fn = jax.jit(basis.EvaluateBsplines3rdDerivatives)

    for x in points:
        non_jit = np.asarray(basis.EvaluateBsplines3rdDerivatives(x))
        jit_eval = np.asarray(jit_eval_fn(x))
        assert np.allclose(jit_eval, non_jit, atol=1e-10, rtol=0)


def test_jax_EvaluateBsplines3rdDerivatives_jit_smoke():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    x = 4.7
    jit_eval_fn = jax.jit(basis.EvaluateBsplines3rdDerivatives)

    non_jit = np.asarray(basis.EvaluateBsplines3rdDerivatives(x))
    jit_eval = np.asarray(jit_eval_fn(x))
    assert np.allclose(jit_eval, non_jit, atol=1e-10, rtol=0)


def test_jax_AssembleSplineMatrix_boundary_rows_matches_gsl():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    expected_first_row = np.array([
        -0.647878198898607,
        1.3639541029444362,
        -1.0920157536698893,
        0.5053617650684089,
        -0.1294219154443486,
        0.0,
        0.0,
        0.0,
    ])
    expected_last_row = np.array([
        0.0,
        0.0,
        0.0,
        -0.11152001784320277,
        0.36347265074821655,
        -0.6438789507572575,
        0.6967578984039893,
        -0.30483158055174536,
    ])

    phi, _ = basis.AssembleSplineMatrix()
    phi = np.asarray(phi)
    assert phi.shape == (8, 8)

    diagnostics = []
    for basis_index in range(phi.shape[1]):
        diagnostics.append((0, basis_index, phi[0, basis_index], expected_first_row[basis_index]))
    for basis_index in range(phi.shape[1]):
        diagnostics.append((phi.shape[0] - 1, basis_index, phi[-1, basis_index], expected_last_row[basis_index]))

    actual_flat = np.array([entry[2] for entry in diagnostics])
    expected_flat = np.array([entry[3] for entry in diagnostics])
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_row, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_row, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at row {max_abs_row} (basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at row {max_rel_row} (basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_AssembleSplineMatrix_matches_gsl():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)
    expected_knots = np.array([1.1, 1.1, 1.1, 1.1, 3.2, 5.1, 7.2, 9.3, 12.0, 12.0, 12.0, 12.0])
    expected_phi = np.array([
        [-0.647878198898607, 1.3639541029444362, -1.0920157536698893, 0.5053617650684089, -0.1294219154443486, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.22562499999999996, 0.5936372950819674, 0.18073770491803284, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1807377049180329, 0.6713114754098362, 0.14795081967213106, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.17213114754098363, 0.675694939415538, 0.1521739130434783, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.22010869565217378, 0.588485054347826, 0.1914062500000001, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, -0.11152001784320277, 0.36347265074821655, -0.6438789507572575, 0.6967578984039893, -0.30483158055174536],
    ])

    phi, knots = basis.AssembleSplineMatrix()
    phi = np.asarray(phi)
    knots = np.asarray(knots)
    assert phi.shape == expected_phi.shape
    assert knots.shape == expected_knots.shape

    diagnostics = []
    for row_index in range(phi.shape[0]):
        for basis_index in range(phi.shape[1]):
            diagnostics.append((row_index, basis_index, phi[row_index, basis_index], expected_phi[row_index, basis_index]))

    actual_flat = np.array([entry[2] for entry in diagnostics])
    expected_flat = np.array([entry[3] for entry in diagnostics])
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_row, max_abs_basis, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_row, max_rel_basis, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at row {max_abs_row} (basis_index={max_abs_basis}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at row {max_rel_row} (basis_index={max_rel_basis}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)
    assert np.array_equal(knots, expected_knots)


def test_jax_AssembleSplineMatrix_jit_smoke():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    basis = TPI_jax.BsplineBasis1D(x1)

    non_jit_phi, non_jit_knots = basis.AssembleSplineMatrix()
    jit_fn = jax.jit(basis.AssembleSplineMatrix)
    jit_phi, jit_knots = jit_fn()

    assert np.allclose(np.asarray(jit_phi), np.asarray(non_jit_phi), atol=1e-10, rtol=0)
    assert np.array_equal(np.asarray(jit_knots), np.asarray(non_jit_knots))


def test_jax_ComputeSplineCoefficientsND_matches_gsl():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    actual = np.asarray(TPint.GetSplineCoefficientsND())
    expected = np.loadtxt(os.path.join(os.path.dirname(__file__), "../data/c_Mma_3D.dat"))
    expected = expected.reshape(actual.shape)

    diagnostics = []
    for index in np.ndindex(actual.shape):
        diagnostics.append((index, actual[index], expected[index]))

    actual_flat = np.array([entry[1] for entry in diagnostics])
    expected_flat = np.array([entry[2] for entry in diagnostics])
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_index, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_index, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at index {max_abs_index} (actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at index {max_rel_index} (actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_ComputeSplineCoefficientsND_uses_cached_lu_factors(monkeypatch):
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    expected = np.asarray(TPI_jax.compute_spline_coefficients_nd(nodes, F))

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.TPInterpolationSetupND()
    assert hasattr(TPint, "spline_matrix_factors")
    assert len(TPint.spline_matrix_factors) == len(nodes)

    def _fail_if_reassembled(*args, **kwargs):
        raise AssertionError("spline matrices should be cached during setup")

    monkeypatch.setattr(TPI_jax, "_assemble_spline_matrix_jax", _fail_if_reassembled)

    TPint.ComputeSplineCoefficientsND(F)
    actual = np.asarray(TPint.GetSplineCoefficientsND())

    diff = actual - expected
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    abs_diff_flat = abs_diff.reshape(-1)
    rel_diff_flat = rel_diff.reshape(-1)
    max_abs_idx = int(np.argmax(abs_diff_flat))
    max_rel_idx = int(np.argmax(rel_diff_flat))
    max_abs_index = np.unravel_index(max_abs_idx, actual.shape)
    max_rel_index = np.unravel_index(max_rel_idx, actual.shape)
    print(
        f"max abs diff: {float(abs_diff_flat[max_abs_idx]):.3e} "
        f"at index={max_abs_index} (actual={actual[max_abs_index]}, expected={expected[max_abs_index]})"
    )
    print(
        f"max rel diff: {float(rel_diff_flat[max_rel_idx]):.3e} "
        f"at index={max_rel_index} (actual={actual[max_rel_index]}, expected={expected[max_rel_index]})"
    )
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def test_jax_ComputeSplineCoefficientsND_boundary_rows_isolated_by_axis_matches_gsl():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)
    actual = np.asarray(TPint.GetSplineCoefficientsND())

    expected = np.loadtxt(os.path.join(os.path.dirname(__file__), "../data/c_Mma_3D.dat"))
    expected = expected.reshape(actual.shape)

    rng = np.random.default_rng(42)
    fixed_positions = {
        0: np.array([[0, j, k] for j, k in zip(rng.integers(1, actual.shape[1] - 1, size=10), rng.integers(1, actual.shape[2] - 1, size=10))], dtype=int),
        1: np.array([[i, 0, k] for i, k in zip(rng.integers(1, actual.shape[0] - 1, size=10), rng.integers(1, actual.shape[2] - 1, size=10))], dtype=int),
        2: np.array([[i, j, 0] for i, j in zip(rng.integers(1, actual.shape[0] - 1, size=10), rng.integers(1, actual.shape[1] - 1, size=10))], dtype=int),
    }
    axis_names = {0: "axis=0", 1: "axis=1", 2: "axis=2"}

    for axis in range(actual.ndim):
        zero_indices = fixed_positions[axis].copy()
        minus_indices = fixed_positions[axis].copy()
        zero_indices[:, axis] = 0
        minus_indices[:, axis] = actual.shape[axis] - 1
        selected_indices = np.vstack((zero_indices, minus_indices))

        actual_selected = actual[tuple(selected_indices.T)]
        expected_selected = expected[tuple(selected_indices.T)]

        diff = actual_selected - expected_selected
        abs_diff = np.abs(diff)
        rel_den = np.maximum(np.abs(expected_selected), np.finfo(np.float64).tiny)
        rel_diff = abs_diff / rel_den
        max_abs_idx = int(np.argmax(abs_diff))
        max_rel_idx = int(np.argmax(rel_diff))
        max_abs_index = tuple(selected_indices[max_abs_idx])
        max_rel_index = tuple(selected_indices[max_rel_idx])
        print(
            f"{axis_names[axis]} max abs diff: {abs_diff[max_abs_idx]:.3e} "
            f"at index={max_abs_index} (actual={actual[max_abs_index]}, expected={expected[max_abs_index]})"
        )
        print(
            f"{axis_names[axis]} max rel diff: {rel_diff[max_rel_idx]:.3e} "
            f"at index={max_rel_index} (actual={actual[max_rel_index]}, expected={expected[max_rel_index]})"
        )
        assert np.allclose(actual_selected, expected_selected, atol=1e-10, rtol=0)


def test_jax_ComputeSplineCoefficientsND_jit_smoke():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = (xi, yi, zi)

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    non_jit = np.asarray(TPI_jax.compute_spline_coefficients_nd(nodes, F))
    jit_fn = jax.jit(TPI_jax.compute_spline_coefficients_nd)
    jit_coeffs = np.asarray(jit_fn(nodes, F))

    assert np.allclose(jit_coeffs, non_jit, atol=1e-10, rtol=0)


def test_jax_TPInterpolationSetupND_builds_explicit_jit_evaluator_and_tracks_updated_coefficients():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f1 = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    f2 = lambda x, y, z: np.cos(3.0 * x) * np.sqrt(y + 1.05) * (1.0 + z**2)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F1 = f1(xx, yy, zz)
    F2 = f2(xx, yy, zz)
    point = np.array([0.1692602, 0.2827312351474, -0.26624193])

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.TPInterpolationSetupND()

    assert hasattr(TPint, "_jit_eval")
    signature = inspect.signature(TPint._jit_eval)
    assert list(signature.parameters)[:2] == ["c", "X"]

    TPint.ComputeSplineCoefficientsND(F1)
    non_jit_1 = np.asarray(TPint.TPInterpolationND(point))
    explicit_1 = np.asarray(TPint._jit_eval(TPint.c, jax.numpy.asarray(point, dtype=jax.numpy.float64)))

    TPint.ComputeSplineCoefficientsND(F2)
    non_jit_2 = np.asarray(TPint.TPInterpolationND(point))
    explicit_2 = np.asarray(TPint._jit_eval(TPint.c, jax.numpy.asarray(point, dtype=jax.numpy.float64)))

    ref1 = TPI_jax.TP_Interpolant_ND(nodes)
    ref1.ComputeSplineCoefficientsND(F1)
    expected_1 = np.asarray(ref1.TPInterpolationND(point))

    ref2 = TPI_jax.TP_Interpolant_ND(nodes)
    ref2.ComputeSplineCoefficientsND(F2)
    expected_2 = np.asarray(ref2.TPInterpolationND(point))

    assert np.allclose(non_jit_1, expected_1, atol=1e-10, rtol=0)
    assert np.allclose(explicit_1, expected_1, atol=1e-10, rtol=0)
    assert np.allclose(non_jit_2, expected_2, atol=1e-10, rtol=0)
    assert np.allclose(explicit_2, expected_2, atol=1e-10, rtol=0)
    assert not np.allclose(expected_1, expected_2, atol=1e-12, rtol=0)


def test_jax_TPInterpolationND_jit_cache_reuses_compiled_kernel_for_same_instance_and_new_points():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.ComputeSplineCoefficientsND(F)

    x_same = np.array([0.1692602, 0.2827312351474, -0.26624193])
    x_new = np.array([0.123, -0.2, 0.4])
    call_points = [x_same, x_same, x_same, x_new, x_same, x_same, x_same, x_same, x_same, x_same]

    times = []
    outputs = []
    for point in call_points:
        start = time.perf_counter()
        outputs.append(np.asarray(TPint.TPInterpolationND(point)))
        times.append((time.perf_counter() - start) * 1e3)

    steady_times = np.array(times[1:], dtype=np.float64)
    first_time = float(times[0])
    median_steady = float(np.median(steady_times))
    max_new_point_time = float(times[3])
    print(f"same interpolant means one TP_Interpolant_ND instance, one fixed X repeated, and one different X among calls 2-10")
    print(f"call times (ms): {np.array2string(np.asarray(times), precision=3, separator=', ')}")
    print(f"first call (ms): {first_time:.3f}")
    print(f"steady median (ms): {median_steady:.3f}")
    print(f"different-X call (ms): {max_new_point_time:.3f}")

    assert first_time > 10.0 * median_steady
    assert max_new_point_time < 2.5 * first_time
    assert np.allclose(outputs[0], outputs[1], atol=1e-10, rtol=0)
    assert np.allclose(outputs[1], outputs[2], atol=1e-10, rtol=0)
    assert np.allclose(outputs[0], outputs[4], atol=1e-10, rtol=0)
    assert not np.allclose(outputs[0], outputs[3], atol=1e-12, rtol=0)


def test_jax_TPInterpolationND_grad_smoke_including_interior_knot_points():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.ComputeSplineCoefficientsND(F)

    grad_fn = jax.grad(lambda x: TPint.TPInterpolationND(x))
    points = [
        np.array([xi[3], yi[5], zi[4]], dtype=np.float64),
        np.array([0.1692602, 0.2827312351474, -0.26624193], dtype=np.float64),
        np.array([0.123, -0.2, 0.4], dtype=np.float64),
    ]

    diagnostics = []
    for point_index, point in enumerate(points):
        grad = np.asarray(grad_fn(jax.numpy.asarray(point, dtype=jax.numpy.float64)))
        diagnostics.append((point_index, point, grad))
        print(f"gradient at point_index={point_index}, point={point}: {grad}")
        assert grad.shape == point.shape
        assert np.all(np.isfinite(grad))


def test_jax_tensor_product_contraction_matches_einsum():
    rng = np.random.default_rng(42)
    diagnostics = []

    for dim in range(1, 5):
        bases = tuple(rng.normal(size=4).astype(np.float64) for _ in range(dim))
        coeff_block = rng.normal(size=(4,) * dim).astype(np.float64)
        labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        basis_labels = ",".join(labels[i] for i in range(dim))
        coeff_labels = "".join(labels[i] for i in range(dim))
        expected = np.asarray(np.einsum(f"{basis_labels},{coeff_labels}->", *bases, coeff_block))
        actual = np.asarray(TPI_jax._contract_tensor_product_jax(tuple(jax.numpy.asarray(b) for b in bases), jax.numpy.asarray(coeff_block)))
        diagnostics.append((dim, actual, expected))

    actual_flat = np.array([entry[1] for entry in diagnostics], dtype=np.float64)
    expected_flat = np.array([entry[2] for entry in diagnostics], dtype=np.float64)
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    print(f"max abs diff: {abs_diff[max_abs_idx]:.3e} at dim={diagnostics[max_abs_idx][0]}")
    print(f"max rel diff: {rel_diff[max_rel_idx]:.3e} at dim={diagnostics[max_rel_idx][0]}")
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_TPInterpolationND_matches_gsl():
    np.set_printoptions(precision=18)
    xi1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    f1 = lambda x: np.cos(10.0 * x)
    F1 = f1(xi1)
    TPint1 = TPI_jax.TP_Interpolant_ND([xi1])
    TPint1.ComputeSplineCoefficientsND(F1)

    xi2 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    f2 = lambda x, y: np.sin(x) * np.arccos(y)
    xx2, yy2 = np.meshgrid(xi2, yi2, indexing="ij")
    F2 = f2(xx2, yy2)
    TPint2 = TPI_jax.TP_Interpolant_ND([xi2, yi2])
    TPint2.ComputeSplineCoefficientsND(F2)

    xi3 = xi2
    yi3 = yi2
    zi3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    f3 = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx3, yy3, zz3 = np.meshgrid(xi3, yi3, zi3, indexing="ij")
    F3 = f3(xx3, yy3, zz3)
    TPint3 = TPI_jax.TP_Interpolant_ND([xi3, yi3, zi3])
    TPint3.ComputeSplineCoefficientsND(F3)

    cases = [
        ("1D", TPint1, np.array([0.16]),  -0.029174542430287713),
        ("1D", TPint1, np.array([0.123]), 0.3342349867624838),
        ("2D", TPint2, np.array([0.16, 0.28]), 0.20507808901038865),
        ("2D", TPint2, np.array([0.123, -0.2]), 0.21742576419081705),
        ("3D", TPint3, np.array([0.1692602, 0.2827312351474, -0.26624193]), 0.16576975057631646),
        ("3D", TPint3, np.array([0.11, -0.2, 0.4]), 0.2902256664884906),
        ("3D", TPint3, np.array([0.247, 0.95, 0.8]), 0.17279609554416436),
        ("3D", TPint3, np.array([0.2, 0.0, -0.6]), 0.17126712868162106),
        ("3D", TPint3, np.array([0.235, 0.6, 0.1]), 0.23862120199268544),
    ]

    diagnostics = []
    for case_index, (label, interpolant, point, expected) in enumerate(cases):
        actual = np.asarray(interpolant.TPInterpolationND(point))
        diagnostics.append((case_index, label, point, actual, expected))

    actual_flat = np.array([float(np.asarray(entry[3])) for entry in diagnostics], dtype=np.float64)
    expected_flat = np.array([float(entry[4]) for entry in diagnostics], dtype=np.float64)
    diff = actual_flat - expected_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_label, max_abs_point, max_abs_actual, max_abs_expected = diagnostics[max_abs_idx]
    max_rel_case, max_rel_label, max_rel_point, max_rel_actual, max_rel_expected = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at case {max_abs_case} ({max_abs_label}, point={max_abs_point}, "
        f"actual={max_abs_actual}, expected={max_abs_expected})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at case {max_rel_case} ({max_rel_label}, point={max_rel_point}, "
        f"actual={max_rel_actual}, expected={max_rel_expected})"
    )
    assert np.allclose(actual_flat, expected_flat, atol=1e-10, rtol=0)


def test_jax_TPInterpolationND_jit_smoke():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.ComputeSplineCoefficientsND(F)

    point = np.array([0.1692602, 0.2827312351474, -0.26624193])
    non_jit = np.asarray(TPint.TPInterpolationND(point))
    jit_fn = jax.jit(TPint.TPInterpolationND)
    jit_eval = np.asarray(jit_fn(point))

    assert np.allclose(jit_eval, non_jit, atol=1e-10, rtol=0)


def test_jax_TPInterpolationND_batched_matches_scalar():
    xi1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    f1 = lambda x: np.cos(10.0 * x)
    F1 = f1(xi1)
    TPint1 = TPI_jax.TP_Interpolant_ND([xi1])
    TPint1.ComputeSplineCoefficientsND(F1)

    xi2 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    f2 = lambda x, y: np.sin(x) * np.arccos(y)
    xx2, yy2 = np.meshgrid(xi2, yi2, indexing="ij")
    F2 = f2(xx2, yy2)
    TPint2 = TPI_jax.TP_Interpolant_ND([xi2, yi2])
    TPint2.ComputeSplineCoefficientsND(F2)

    xi3 = xi2
    yi3 = yi2
    zi3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    f3 = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx3, yy3, zz3 = np.meshgrid(xi3, yi3, zi3, indexing="ij")
    F3 = f3(xx3, yy3, zz3)
    TPint3 = TPI_jax.TP_Interpolant_ND([xi3, yi3, zi3])
    TPint3.ComputeSplineCoefficientsND(F3)

    cases = [
        ("1D", TPint1, np.array([[0.16], [0.123], [0.249]])),
        ("2D", TPint2, np.array([[0.16, 0.28], [0.123, -0.2], [0.2, 0.0]])),
        (
            "3D",
            TPint3,
            np.array(
                [
                    [0.1692602, 0.2827312351474, -0.26624193],
                    [0.11, -0.2, 0.4],
                    [0.235, 0.6, 0.1],
                ]
            ),
        ),
    ]

    diagnostics = []
    for case_index, (label, interpolant, batch_points) in enumerate(cases):
        scalar_values = np.array([float(np.asarray(interpolant.TPInterpolationND(point))) for point in batch_points])
        batch_values = np.asarray(interpolant.TPInterpolationND_batched(batch_points))
        diagnostics.extend(
            [
                (case_index, label, point_index, batch_points[point_index], scalar_values[point_index], batch_values[point_index])
                for point_index in range(batch_points.shape[0])
            ]
        )

    scalar_flat = np.array([entry[4] for entry in diagnostics], dtype=np.float64)
    batch_flat = np.array([entry[5] for entry in diagnostics], dtype=np.float64)
    diff = batch_flat - scalar_flat
    abs_diff = np.abs(diff)
    rel_den = np.maximum(np.abs(scalar_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_case, max_abs_label, max_abs_point_index, max_abs_point, max_abs_scalar, max_abs_batch = diagnostics[max_abs_idx]
    max_rel_case, max_rel_label, max_rel_point_index, max_rel_point, max_rel_scalar, max_rel_batch = diagnostics[max_rel_idx]
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at case {max_abs_case} ({max_abs_label}, point_index={max_abs_point_index}, "
        f"point={max_abs_point}, scalar={max_abs_scalar}, batch={max_abs_batch})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at case {max_rel_case} ({max_rel_label}, point_index={max_rel_point_index}, "
        f"point={max_rel_point}, scalar={max_rel_scalar}, batch={max_rel_batch})"
    )
    assert np.allclose(batch_flat, scalar_flat, atol=1e-14, rtol=0)


def test_jax_TPInterpolationND_batched_jit_smoke():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]

    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.ComputeSplineCoefficientsND(F)

    batch_1 = np.array(
        [
            [0.1692602, 0.2827312351474, -0.26624193],
            [0.11, -0.2, 0.4],
            [0.235, 0.6, 0.1],
        ]
    )
    batch_2 = np.array(
        [
            [0.123, -0.2, 0.4],
            [0.247, 0.95, 0.8],
            [0.2, 0.0, -0.6],
        ]
    )

    non_jit_1 = np.asarray(TPint.TPInterpolationND_batched(batch_1))
    non_jit_2 = np.asarray(TPint.TPInterpolationND_batched(batch_2))
    jit_fn = jax.jit(TPint.TPInterpolationND_batched)
    jit_1 = np.asarray(jit_fn(batch_1))
    jit_2 = np.asarray(jit_fn(batch_2))

    assert np.allclose(jit_1, non_jit_1, atol=1e-10, rtol=0)
    assert np.allclose(jit_2, non_jit_2, atol=1e-10, rtol=0)


def test_SplineMatrix():
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
    b = TPI.BsplineBasis1D(x1)
    atol = np.finfo(float).eps

    # Test knots
    b_phi, b_knots = b.AssembleSplineMatrix()
    knots = np.array([1.1, 1.1, 1.1, 1.1, 3.2, 5.1, 7.2, 9.3, 12, 12, 12, 12])
    assert np.array_equal(b_knots, knots)

    # Test spline matrix with not-a-knot boundary conditions
    phi = np.array([[-0.647878198898607, 1.363954102944436, -1.0920157536698896, 0.505361765068409, -0.1294219154443486, 0, 0, 0],
    [1., 0., 0., 0., 0, 0, 0, 0],
    [0, 0.22562499999999996, 0.5936372950819674, 0.18073770491803284, 0., 0, 0, 0],
    [0, 0, 0.1807377049180329, 0.6713114754098362, 0.14795081967213106, 0., 0, 0],
    [0, 0, 0, 0.17213114754098363, 0.675694939415538, 0.1521739130434783, 0., 0],
    [0, 0, 0, 0, 0.22010869565217378, 0.588485054347826, 0.1914062500000001, 0.],
    [0, 0, 0, 0, 0., 0., 0., 1.],
    [0, 0, 0, -0.11152001784320278, 0.3634726507482166, -0.6438789507572575, 0.6967578984039893, -0.30483158055174536]])

    # print 'b_phi - phi', np.matrix(b_phi - phi)
    assert np.allclose(b_phi, phi, atol=atol, rtol=0)

def test_TP_spline_interpolation_1D():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    X = [xi]

    def f(x):
        return np.cos(10.0*x)

    F = f(xi)

    TPint = TPI.TP_Interpolant_ND(X)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    # Check coefficients
    c_TPI = TPint.GetSplineCoefficientsND()
    assert c_TPI.shape[0] == 12

    # Check evaluated interpolant
    Y = np.array([0.16])
    res = TPint.TPInterpolationND(Y)
    res_Mma = -0.029174542430286686
    assert np.allclose(res, res_Mma, atol=1e-13, rtol=0)

def test_TP_spline_interpolation_2D():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    X = [xi, yi]

    def f(x,y):
        return np.sin(x) * np.arccos(y)

    xx, yy = np.meshgrid(xi, yi, indexing='ij')
    F = f(xx, yy)

    TPint = TPI.TP_Interpolant_ND(X)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    # Check coefficients
    c_TPI = TPint.GetSplineCoefficientsND()
    assert c_TPI.shape == (12, 15)

    # Check evaluated interpolant
    Y = np.array([0.16, 0.28])
    res = TPint.TPInterpolationND(Y)
    res_Mma = 0.2050780890103884
    assert np.allclose(res, res_Mma, atol=1e-13, rtol=0)

    with pytest.raises(ValueError):
        res = TPint.TPInterpolationND(np.array([-0.8, 12.3]))

def test_TP_spline_interpolation_3D():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    X = [xi, yi, zi]

    def f(x,y,z):
        return np.sin(x) * np.arccos(y) * np.exp(z)

    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    F = f(xx, yy, zz)

    TPint = TPI.TP_Interpolant_ND(X)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    # Check coefficients
    c_TPI = TPint.GetSplineCoefficientsND()
    assert c_TPI.shape == (12, 15, 11)

    TPint.SetSplineCoefficientsND(c_TPI)

    with pytest.raises(TypeError):
        TPint2 = TPI.TP_Interpolant_ND(["aaaa"])

    with pytest.raises(ValueError):
        c = np.arange(12, dtype=np.float64).reshape((3,2,2))
        TPint.SetSplineCoefficientsND(c)

    with pytest.raises(ValueError):
        c = []
        TPint.SetSplineCoefficientsND(c)

    pth = os.path.abspath(os.path.dirname(__file__))
    c_Mma = np.loadtxt(f"{pth}/../data/c_Mma_3D.dat") # grab external coefficient data exported from Mathematica
    assert len(c_TPI.flatten()) == len(c_Mma)
    assert np.allclose(c_TPI.flatten(), c_Mma, atol=1e-12, rtol=0)

    # Check evaluated interpolant
    Y = np.array([0.1692602, 0.2827312351474, -0.26624193])
    res = TPint.TPInterpolationND(Y)
    res_Mma = 0.16576975057631677
    assert np.allclose(res, res_Mma, atol=1e-13, rtol=0)

    # and its relative error compared to the function f
    rel_err = (f(*Y) - TPint.TPInterpolationND(Y)) / f(*Y)
    rel_err_Mma = -0.00008225243596719076
    assert np.allclose(rel_err, rel_err_Mma, atol=1e-13, rtol=0)

    # create a new object and set the spline coefficients directly
    TPint2 = TPI.TP_Interpolant_ND(X, coeffs=c_TPI)
    res2 = TPint2.TPInterpolationND(Y)
    assert np.allclose(res2, res_Mma, atol=1e-13, rtol=0)

    with pytest.raises(ValueError):
        Y = np.array([-1692602, 28.27312351474, -2.6624193])
        res = TPint.TPInterpolationND(Y)

def test_TP_spline_interpolation_4D():
    x1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    x2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    x3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    x4 = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])
    X = [x1, x2, x3, x4]

    def f(x1,x2,x3,x4):
        return np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4)

    xx1, xx2, xx3, xx4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
    F = f(xx1, xx2, xx3, xx4)

    TPint = TPI.TP_Interpolant_ND(X)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    # Check coefficients
    c_TPI = TPint.GetSplineCoefficientsND()
    assert c_TPI.shape == (12, 15, 11, 9)

    # Check evaluated interpolant
    Y = np.array([0.16, 0.28, -0.26, 0.05])
    res = TPint.TPInterpolationND(Y)
    res_Mma = 0.15790875815398853
    assert np.allclose(res, res_Mma, atol=1e-13, rtol=0)

def test_TP_spline_interpolation_7D():
    # Randomly generated nodes data
    # def gen():
    #     n = np.random.randint(5, 12, 1).item()
    #     x = np.random.random(n) /  np.random.random(1)
    #     x.sort()
    #     return x
    #
    # X = [gen() for i in range(7)]

    X = [np.array([ 0.04210023,  0.08049712,  0.10439003,  0.23567061,  0.26747638,
         0.51894333,  0.87695656,  1.13424169]),
   np.array([ 0.06512773,  0.10554492,  0.30739299,  0.52934042,  0.53375456,
           0.70565296,  0.90977329,  1.0904668 ,  1.09161535]),
   np.array([ 0.17568927,  0.20990473,  0.40272389,  0.54519648,  0.62970609,
           0.65005828,  0.67672559,  1.03551716]),
   np.array([ 0.04209146,  0.18164518,  0.32001217,  0.5469396 ,  0.65685659,
           0.69706066,  0.8338755 ,  0.84175853,  1.03421552]),
   np.array([ 0.15592869,  0.24300596,  0.53102712,  0.76409654,  0.83426527]),
   np.array([ 0.09278997,  0.60858288,  0.68604479,  0.69185573,  1.05187626,
           1.25311729,  1.83783997,  2.30367353,  2.34835024,  2.5526501 ,
           2.88134666]),
   np.array([ 0.02039465,  0.84219648,  1.34410666,  1.50315468,  1.77942063,
           4.30194875,  4.81437542,  5.30694653,  6.04394163])]

    def f(x1,x2,x3,x4,x5,x6,x7):
        return np.sin(x1) * np.arccos(x2/2.) * np.exp(x3) * np.cos(x4) * np.abs(x5) + np.sin(x6)*np.exp(x7)

    xx1, xx2, xx3, xx4, xx5, xx6, xx7 = np.meshgrid(X[0], X[1], X[2], X[3], X[4], X[5], X[6], indexing='ij')
    F = f(xx1, xx2, xx3, xx4, xx5, xx6, xx7)

    TPint = TPI.TP_Interpolant_ND(X)
    TPint.TPInterpolationSetupND()
    TPint.ComputeSplineCoefficientsND(F)

    # Check coefficients
    c_TPI = TPint.GetSplineCoefficientsND()
    assert c_TPI.shape == (10, 11, 10, 11, 7, 13, 11)

    # Check evaluated interpolant
    Y = np.array([0.673, 0.2836, 0.734, 0.089, 0.619, 1.782, 4.96])
    res_expected = 140.401088491
    res = TPint.TPInterpolationND(Y)
    assert np.allclose(res, res_expected, atol=1e-7, rtol=0)

    rel_err_expected = 0.00119488601664
    rel_err = (f(*Y) - res) / f(*Y)
    assert np.allclose(rel_err, rel_err_expected, atol=1e-8, rtol=0)

    with pytest.raises(ValueError):
        Y = np.array([-16.9, 26.02, 28.2731, 23.51474, -2.6624, 193.3, 9915.1])
        res = TPint.TPInterpolationND(Y)


# Hack for running tests since pytest does not import the Cython module under python3
# Just run: python3 test.py
'''
if __name__ == "__main__":
    test_BsplineBasis1D()
    test_BsplineBasis1D_fail()
    test_SplineMatrix()
    test_TP_spline_interpolation_1D()
    test_TP_spline_interpolation_2D()
    test_TP_spline_interpolation_3D()
    test_TP_spline_interpolation_4D()
    test_TP_spline_interpolation_7D()
    print('All tests passed successfully')
'''
