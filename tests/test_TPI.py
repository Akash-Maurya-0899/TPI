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


def _single_chain_interpolant_cases():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    wi = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])
    functions = {
        1: lambda x: np.cos(10.0 * x),
        2: lambda x, y: np.sin(x) * np.arccos(y),
        3: lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
        4: lambda x, y, z, w: np.sin(x) * np.arccos(y) * np.exp(z) * np.cos(w),
    }
    node_sets = {1: (xi,), 2: (xi, yi), 3: (xi, yi, zi), 4: (xi, yi, zi, wi)}
    cases = []
    for dim, nodes in node_sets.items():
        mesh = np.meshgrid(*nodes, indexing="ij")
        F = np.asarray(functions[dim](*mesh), dtype=np.float64)
        gsl_interp = TPI.TP_Interpolant_ND(list(nodes))
        gsl_interp.ComputeSplineCoefficientsND(F)
        jax_interp = TPI_jax.TP_Interpolant_ND(list(nodes))
        jax_interp.ComputeSplineCoefficientsND(F)
        cases.append((dim, nodes, gsl_interp, jax_interp))
    return cases


def _single_chain_interior_points(nodes, count=30):
    rng = np.random.default_rng(42)
    lows = np.array([node[0] for node in nodes], dtype=np.float64)
    highs = np.array([node[-1] for node in nodes], dtype=np.float64)
    return lows + rng.uniform(0.02, 0.98, size=(count, len(nodes))) * (highs - lows)


def _print_max_diffs(diagnostics):
    actual_flat = np.array([entry[2] for entry in diagnostics], dtype=np.float64)
    expected_flat = np.array([entry[3] for entry in diagnostics], dtype=np.float64)
    abs_diff = np.abs(actual_flat - expected_flat)
    rel_den = np.maximum(np.abs(expected_flat), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    print(
        f"max abs diff: {abs_diff[max_abs_idx]:.3e} "
        f"at dim={diagnostics[max_abs_idx][0]} (x={diagnostics[max_abs_idx][1]})"
    )
    print(
        f"max rel diff: {rel_diff[max_rel_idx]:.3e} "
        f"at dim={diagnostics[max_rel_idx][0]} (x={diagnostics[max_rel_idx][1]})"
    )
    return actual_flat, expected_flat


def test_jax_single_point_evaluator_matches_gsl_random_interior_points():
    diagnostics = []
    for dim, nodes, gsl_interp, jax_interp in _single_chain_interpolant_cases():
        points = _single_chain_interior_points(nodes)
        # warmup: trigger JIT compilation before assertions
        jax_interp.TPInterpolationND(points[0])
        jax_values = np.asarray(
            jax.vmap(lambda p, interp=jax_interp: interp.TPInterpolationND(p))(
                jax.numpy.asarray(points, dtype=jax.numpy.float64)
            )
        )
        gsl_values = np.asarray(
            [gsl_interp.TPInterpolationND(point) for point in points], dtype=np.float64
        )
        diagnostics.extend(
            (dim, points[i], jax_values[i], gsl_values[i]) for i in range(points.shape[0])
        )

    jax_flat, gsl_flat = _print_max_diffs(diagnostics)
    assert np.allclose(jax_flat, gsl_flat, atol=1e-10, rtol=0)


def test_jax_single_point_evaluator_matches_legacy_per_axis_kernel():
    diagnostics = []
    for dim, nodes, _, jax_interp in _single_chain_interpolant_cases():
        points = jax.numpy.asarray(_single_chain_interior_points(nodes), dtype=jax.numpy.float64)
        # warmup: trigger JIT compilation before assertions
        jax_interp.TPInterpolationND(np.asarray(points[0]))
        new_values = np.asarray(
            jax.vmap(lambda p, interp=jax_interp: interp.TPInterpolationND(p))(points)
        )
        legacy_values = np.asarray(jax.vmap(jax_interp._TPInterpolationND_jax)(points))
        diagnostics.extend(
            (dim, np.asarray(points[i]), new_values[i], legacy_values[i])
            for i in range(points.shape[0])
        )

    new_flat, legacy_flat = _print_max_diffs(diagnostics)
    assert np.allclose(new_flat, legacy_flat, atol=1e-13, rtol=0)


def test_jax_single_point_evaluator_boundary_points_match_gsl():
    diagnostics = []
    for dim, nodes, gsl_interp, jax_interp in _single_chain_interpolant_cases():
        lows = np.array([node[0] for node in nodes], dtype=np.float64)
        highs = np.array([node[-1] for node in nodes], dtype=np.float64)
        interior = [
            np.array(
                [nodes[axis][1 + (j + axis) % (len(nodes[axis]) - 2)] for axis in range(dim)],
                dtype=np.float64,
            )
            for j in range(8)
        ]
        points = np.vstack([lows, highs] + interior)
        # warmup: trigger JIT compilation before assertions
        jax_interp.TPInterpolationND(points[0])
        jax_values = np.asarray(
            jax.vmap(lambda p, interp=jax_interp: interp.TPInterpolationND(p))(
                jax.numpy.asarray(points, dtype=jax.numpy.float64)
            )
        )
        gsl_values = np.asarray(
            [gsl_interp.TPInterpolationND(point) for point in points], dtype=np.float64
        )
        diagnostics.extend(
            (dim, points[i], jax_values[i], gsl_values[i]) for i in range(points.shape[0])
        )

    jax_flat, gsl_flat = _print_max_diffs(diagnostics)
    assert np.allclose(jax_flat, gsl_flat, atol=1e-10, rtol=0)


def test_jax_single_point_evaluator_grad_smoke_1d_2d_3d():
    for dim, nodes, _, jax_interp in _single_chain_interpolant_cases():
        if dim > 3:
            continue
        points = jax.numpy.asarray(
            _single_chain_interior_points(nodes, count=3), dtype=jax.numpy.float64
        )
        grad_fn = jax.vmap(jax.grad(lambda p, interp=jax_interp: interp.TPInterpolationND(p)))
        grads = np.asarray(grad_fn(points))
        print(f"dim={dim} grads:\n{grads}")
        assert grads.shape == (3, dim)
        assert np.all(np.isfinite(grads))


def test_jax_TPInterpolationND_range_errors_and_endpoint_acceptance():
    for dim, nodes, _, jax_interp in _single_chain_interpolant_cases():
        lows = np.array([node[0] for node in nodes], dtype=np.float64)
        highs = np.array([node[-1] for node in nodes], dtype=np.float64)

        # exact endpoints must be accepted by both scalar and batched paths
        endpoints = np.vstack([lows, highs])
        for point in endpoints:
            value = float(np.asarray(jax_interp.TPInterpolationND(point)))
            assert np.isfinite(value)
        batch_values = np.asarray(jax_interp.TPInterpolationND_batched(endpoints))
        assert np.all(np.isfinite(batch_values))

        # out-of-range on each axis must raise with the axis and bounds named
        for axis in range(dim):
            for bad_value, bound in ((lows[axis] - 0.5, lows[axis]), (highs[axis] + 0.5, highs[axis])):
                bad_point = 0.5 * (lows + highs)
                bad_point[axis] = bad_value
                with pytest.raises(ValueError, match=f"X\\[{axis}\\]"):
                    jax_interp.TPInterpolationND(bad_point)
                bad_batch = np.vstack([0.5 * (lows + highs), bad_point])
                with pytest.raises(ValueError, match=f"X\\[1, {axis}\\]"):
                    jax_interp.TPInterpolationND_batched(bad_batch)

        # shape errors are unchanged
        with pytest.raises(ValueError):
            jax_interp.TPInterpolationND(np.zeros(dim + 1))
        with pytest.raises(ValueError):
            jax_interp.TPInterpolationND_batched(np.zeros((2, dim + 1)))


def test_jax_BsplineBasis1D_range_errors_and_endpoint_acceptance():
    nodes = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    basis = TPI_jax.BsplineBasis1D(nodes)

    for x in (nodes[0], nodes[-1], 0.16):
        values = np.asarray(basis.EvaluateBsplines(x))
        assert np.all(np.isfinite(values))
        derivs = np.asarray(basis.EvaluateBsplines3rdDerivatives(x))
        assert np.all(np.isfinite(derivs))

    for x in (nodes[0] - 0.01, nodes[-1] + 0.01):
        with pytest.raises(ValueError, match="outside of knots"):
            basis.EvaluateBsplines(x)
        with pytest.raises(ValueError, match="outside of knots"):
            basis.EvaluateBsplines3rdDerivatives(x)


def _vector_case_2d():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    nodes = (xi, yi)
    component_functions = (
        lambda x, y: np.sin(x) * np.arccos(y),
        lambda x, y: np.cos(10.0 * x) * np.exp(y),
        lambda x, y: x * y + 0.5 * y * y,
    )
    xx, yy = np.meshgrid(xi, yi, indexing="ij")
    F = np.stack([fn(xx, yy) for fn in component_functions], axis=-1)
    return nodes, component_functions, F


def test_jax_vector_valued_interpolation_matches_per_component_gsl():
    nodes, component_functions, F = _vector_case_2d()
    ncomp = F.shape[-1]

    vector_interp = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,))
    coeffs = vector_interp.ComputeSplineCoefficientsND(F)
    assert coeffs.shape == tuple(len(node) + 2 for node in nodes) + (ncomp,)

    gsl_interps = []
    for comp in range(ncomp):
        gsl_interp = TPI.TP_Interpolant_ND(list(nodes))
        gsl_interp.ComputeSplineCoefficientsND(np.ascontiguousarray(F[..., comp]))
        gsl_interps.append(gsl_interp)
        comp_coeff_diff = float(
            np.max(np.abs(np.asarray(coeffs[..., comp]) - np.asarray(gsl_interp.GetSplineCoefficientsND())))
        )
        print(f"component {comp} coefficient max abs diff vs GSL: {comp_coeff_diff:.3e}")
        assert comp_coeff_diff < 1e-10

    points = _single_chain_interior_points(nodes)
    # warmup: trigger JIT compilation before assertions
    vector_interp.TPInterpolationND(points[0])
    jax_values = np.asarray(
        jax.vmap(lambda p: vector_interp.TPInterpolationND(p))(
            jax.numpy.asarray(points, dtype=jax.numpy.float64)
        )
    )
    assert jax_values.shape == (points.shape[0], ncomp)
    gsl_values = np.stack(
        [
            np.asarray([gsl_interps[comp].TPInterpolationND(point) for point in points])
            for comp in range(ncomp)
        ],
        axis=-1,
    )
    abs_diff = np.abs(jax_values - gsl_values)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    rel_diff = abs_diff / np.maximum(np.abs(gsl_values), np.finfo(np.float64).tiny)
    ridx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    print(f"max abs diff: {abs_diff[idx]:.3e} at point={points[idx[0]]}, component={idx[1]}")
    print(f"max rel diff: {rel_diff[ridx]:.3e} at point={points[ridx[0]]}, component={ridx[1]}")
    assert np.allclose(jax_values, gsl_values, atol=1e-10, rtol=0)


def test_jax_tensor_valued_interpolation_shapes_and_batched_and_jit():
    nodes, component_functions, F = _vector_case_2d()
    F_tensor = np.concatenate([F, 2.0 * F], axis=-1).reshape(F.shape[:-1] + (2, 3))

    tensor_interp = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(2, 3), F=F_tensor)
    coeffs = tensor_interp.GetSplineCoefficientsND()
    assert coeffs.shape == tuple(len(node) + 2 for node in nodes) + (2, 3)

    point = np.array([0.16, 0.28])
    single = np.asarray(tensor_interp.TPInterpolationND(point))
    assert single.shape == (2, 3)

    points = _single_chain_interior_points(nodes, count=7)
    batch = np.asarray(tensor_interp.TPInterpolationND_batched(points))
    assert batch.shape == (7, 2, 3)
    singles = np.stack([np.asarray(tensor_interp.TPInterpolationND(p)) for p in points])
    max_abs = float(np.max(np.abs(batch - singles)))
    print(f"tensor-valued batch vs scalar max abs diff: {max_abs:.3e}")
    assert np.allclose(batch, singles, atol=1e-14, rtol=0)

    jit_fn = jax.jit(tensor_interp.TPInterpolationND)
    jit_value = np.asarray(jit_fn(point))
    assert np.allclose(jit_value, single, atol=1e-14, rtol=0)

    # first three components duplicate the vector case, last three are scaled
    assert np.allclose(single[0], 0.5 * single[1], atol=1e-13, rtol=0)


def test_jax_vector_from_component_splines_matches_components():
    nodes, component_functions, F = _vector_case_2d()
    ncomp = F.shape[-1]

    scalar_interps = []
    for comp in range(ncomp):
        interp = TPI_jax.TP_Interpolant_ND(list(nodes))
        interp.ComputeSplineCoefficientsND(np.ascontiguousarray(F[..., comp]))
        scalar_interps.append(interp)

    # combine from coefficient arrays and from interpolant objects
    from_coeffs = TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(
        list(nodes), [np.asarray(interp.GetSplineCoefficientsND()) for interp in scalar_interps]
    )
    from_interps = TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), scalar_interps)

    points = _single_chain_interior_points(nodes, count=10)
    scalar_values = np.stack(
        [
            np.asarray([float(np.asarray(scalar_interps[comp].TPInterpolationND(p))) for p in points])
            for comp in range(ncomp)
        ],
        axis=-1,
    )
    for label, combined in (("coeff arrays", from_coeffs), ("interpolants", from_interps)):
        combined_values = np.asarray(combined.TPInterpolationND_batched(points))
        max_abs = float(np.max(np.abs(combined_values - scalar_values)))
        print(f"FromComponentSplines({label}) max abs diff vs scalar splines: {max_abs:.3e}")
        assert combined_values.shape == (10, ncomp)
        assert np.allclose(combined_values, scalar_values, atol=1e-14, rtol=0)

    with pytest.raises(ValueError):
        TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), [])
    with pytest.raises(ValueError):
        bad = np.zeros((3, 3))
        TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), [bad])


def test_jax_vector_valued_jacobian_smoke():
    nodes, component_functions, F = _vector_case_2d()
    ncomp = F.shape[-1]
    vector_interp = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,), F=F)

    points = jax.numpy.asarray(_single_chain_interior_points(nodes, count=3))
    jac_fn = jax.vmap(jax.jacfwd(lambda p: vector_interp.TPInterpolationND(p)))
    jacobians = np.asarray(jac_fn(points))
    print(f"vector-valued jacobians shape: {jacobians.shape}")
    assert jacobians.shape == (3, ncomp, len(nodes))
    assert np.all(np.isfinite(jacobians))


def test_jax_vector_valued_shape_errors():
    nodes, component_functions, F = _vector_case_2d()

    with pytest.raises(ValueError):
        TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(0,))

    interp = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,))
    with pytest.raises(ValueError):
        interp.ComputeSplineCoefficientsND(F[..., :2])
    with pytest.raises(ValueError):
        interp.ComputeSplineCoefficientsND(F[..., 0])
    with pytest.raises(ValueError):
        interp.SetSplineCoefficientsND(np.zeros(tuple(len(node) + 2 for node in nodes)))


def _vector_case_3d():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = (xi, yi, zi)
    component_functions = (
        lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
        lambda x, y, z: np.cos(10.0 * x) * y * z,
        lambda x, y, z: x * y + z * z,
    )
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = np.stack([fn(xx, yy, zz) for fn in component_functions], axis=-1)
    return nodes, component_functions, F


def test_gsl_vector_valued_interpolation_matches_per_component():
    for case in (_vector_case_2d(), _vector_case_3d()):
        nodes, component_functions, F = case
        ncomp = F.shape[-1]
        dim = len(nodes)

        vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,))
        coeffs = np.asarray(vector_interp.ComputeSplineCoefficientsND(F))
        assert coeffs.shape == tuple(len(node) + 2 for node in nodes) + (ncomp,)

        scalar_interps = []
        for comp in range(ncomp):
            scalar_interp = TPI.TP_Interpolant_ND(list(nodes))
            scalar_interp.ComputeSplineCoefficientsND(np.ascontiguousarray(F[..., comp]))
            scalar_interps.append(scalar_interp)
            comp_coeff_diff = float(
                np.max(np.abs(coeffs[..., comp] - np.asarray(scalar_interp.GetSplineCoefficientsND())))
            )
            print(f"{dim}D component {comp} coefficient max abs diff vs scalar GSL: {comp_coeff_diff:.3e}")
            assert comp_coeff_diff < 1e-13

        points = _single_chain_interior_points(nodes)
        vector_values = np.stack([np.asarray(vector_interp.TPInterpolationND(p)) for p in points])
        assert vector_values.shape == (points.shape[0], ncomp)
        scalar_values = np.stack(
            [
                np.asarray([scalar_interps[comp].TPInterpolationND(point) for point in points])
                for comp in range(ncomp)
            ],
            axis=-1,
        )
        abs_diff = np.abs(vector_values - scalar_values)
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        rel_diff = abs_diff / np.maximum(np.abs(scalar_values), np.finfo(np.float64).tiny)
        ridx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        print(f"{dim}D max abs diff: {abs_diff[idx]:.3e} at point={points[idx[0]]}, component={idx[1]}")
        print(f"{dim}D max rel diff: {rel_diff[ridx]:.3e} at point={points[ridx[0]]}, component={ridx[1]}")
        assert np.allclose(vector_values, scalar_values, atol=1e-14, rtol=0)


def test_gsl_vector_valued_interpolation_matches_jax_vector():
    nodes, component_functions, F = _vector_case_3d()
    ncomp = F.shape[-1]

    gsl_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,), F=F)
    jax_interp = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,), F=F)

    points = _single_chain_interior_points(nodes)
    # warmup: trigger JIT compilation before assertions
    jax_interp.TPInterpolationND(points[0])
    gsl_values = np.stack([np.asarray(gsl_interp.TPInterpolationND(p)) for p in points])
    jax_values = np.asarray(jax_interp.TPInterpolationND_batched(points))
    abs_diff = np.abs(gsl_values - jax_values)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    rel_diff = abs_diff / np.maximum(np.abs(jax_values), np.finfo(np.float64).tiny)
    ridx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    print(f"max abs diff: {abs_diff[idx]:.3e} at point={points[idx[0]]}, component={idx[1]}")
    print(f"max rel diff: {rel_diff[ridx]:.3e} at point={points[ridx[0]]}, component={ridx[1]}")
    assert np.allclose(gsl_values, jax_values, atol=1e-10, rtol=0)


def test_gsl_tensor_valued_interpolation_shapes_and_call():
    nodes, component_functions, F = _vector_case_2d()
    F_tensor = np.concatenate([F, 2.0 * F], axis=-1).reshape(F.shape[:-1] + (2, 3))

    tensor_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(2, 3), F=F_tensor)
    coeffs = np.asarray(tensor_interp.GetSplineCoefficientsND())
    assert coeffs.shape == tuple(len(node) + 2 for node in nodes) + (2, 3)

    point = np.array([0.16, 0.28])
    single = np.asarray(tensor_interp.TPInterpolationND(point))
    assert single.shape == (2, 3)
    called = np.asarray(tensor_interp(point))
    assert np.allclose(called, single, atol=0, rtol=0)

    # first three components duplicate the vector case, last three are scaled
    max_abs = float(np.max(np.abs(single[0] - 0.5 * single[1])))
    print(f"tensor-valued row consistency max abs diff: {max_abs:.3e}")
    assert np.allclose(single[0], 0.5 * single[1], atol=1e-13, rtol=0)


def test_gsl_vector_from_component_splines_matches_components():
    nodes, component_functions, F = _vector_case_2d()
    ncomp = F.shape[-1]

    gsl_scalars = []
    jax_scalars = []
    for comp in range(ncomp):
        F_comp = np.ascontiguousarray(F[..., comp])
        gsl_scalars.append(TPI.TP_Interpolant_ND(list(nodes), F=F_comp))
        jax_scalars.append(TPI_jax.TP_Interpolant_ND(list(nodes), F=F_comp))

    # combine from coefficient arrays, GSL interpolants, and JAX interpolants
    from_coeffs = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(
        list(nodes), [np.asarray(interp.GetSplineCoefficientsND()) for interp in gsl_scalars]
    )
    from_gsl = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), gsl_scalars)
    from_jax = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), jax_scalars)

    points = _single_chain_interior_points(nodes, count=10)
    scalar_values = np.stack(
        [
            np.asarray([gsl_scalars[comp].TPInterpolationND(p) for p in points])
            for comp in range(ncomp)
        ],
        axis=-1,
    )
    for label, combined, atol in (
        ("coeff arrays", from_coeffs, 1e-14),
        ("GSL interpolants", from_gsl, 1e-14),
        ("JAX interpolants", from_jax, 1e-10),
    ):
        combined_values = np.stack([np.asarray(combined.TPInterpolationND(p)) for p in points])
        max_abs = float(np.max(np.abs(combined_values - scalar_values)))
        print(f"FromComponentSplines({label}) max abs diff vs scalar splines: {max_abs:.3e}")
        assert combined_values.shape == (10, ncomp)
        assert np.allclose(combined_values, scalar_values, atol=atol, rtol=0)

    # values_shape reshaping of the stacked components
    from_shaped = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(
        list(nodes), gsl_scalars + gsl_scalars, values_shape=(2, 3)
    )
    shaped = np.asarray(from_shaped.TPInterpolationND(points[0]))
    assert shaped.shape == (2, 3)
    assert np.allclose(shaped[0], shaped[1], atol=0, rtol=0)
    assert np.allclose(shaped[0], scalar_values[0], atol=1e-14, rtol=0)

    with pytest.raises(ValueError):
        TPI.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), [])
    with pytest.raises(ValueError):
        TPI.TP_Interpolant_ND_Vector.FromComponentSplines(list(nodes), [np.zeros((3, 3))])
    with pytest.raises(ValueError):
        TPI.TP_Interpolant_ND_Vector.FromComponentSplines(
            list(nodes), gsl_scalars, values_shape=(2, 2)
        )


def test_gsl_vector_valued_shape_and_range_errors():
    nodes, component_functions, F = _vector_case_2d()

    with pytest.raises(ValueError):
        TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(0,))

    interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,))
    with pytest.raises(ValueError):
        interp.ComputeSplineCoefficientsND(F[..., :2])
    with pytest.raises(ValueError):
        interp.ComputeSplineCoefficientsND(F[..., 0])
    with pytest.raises(ValueError):
        interp.SetSplineCoefficientsND(np.zeros(tuple(len(node) + 2 for node in nodes)))

    interp.ComputeSplineCoefficientsND(F)
    with pytest.raises(ValueError, match="outside of knots"):
        interp.TPInterpolationND(np.array([0.5, 0.0]))  # x outside [0.1, 0.25]
    with pytest.raises(ValueError):
        interp(np.array([0.16, 0.28, 0.0]))  # wrong dimension


def test_gsl_vector_valued_boundary_points_match_per_component():
    nodes, component_functions, F = _vector_case_2d()
    ncomp = F.shape[-1]

    vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,), F=F)
    scalar_interps = [
        TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., comp]))
        for comp in range(ncomp)
    ]

    xi, yi = nodes
    # left endpoint, right endpoint, and a sample of interior nodes (10 points)
    points = np.array(
        [
            [xi[0], yi[0]],
            [xi[-1], yi[-1]],
            [xi[0], yi[-1]],
            [xi[-1], yi[0]],
            [xi[3], yi[5]],
            [xi[5], yi[2]],
            [xi[7], yi[9]],
            [xi[2], yi[11]],
            [xi[8], yi[7]],
            [xi[4], yi[4]],
        ],
        dtype=np.float64,
    )
    vector_values = np.stack([np.asarray(vector_interp.TPInterpolationND(p)) for p in points])
    scalar_values = np.stack(
        [
            np.asarray([scalar_interps[comp].TPInterpolationND(point) for point in points])
            for comp in range(ncomp)
        ],
        axis=-1,
    )
    abs_diff = np.abs(vector_values - scalar_values)
    idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    rel_diff = abs_diff / np.maximum(np.abs(scalar_values), np.finfo(np.float64).tiny)
    ridx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
    print(f"max abs diff: {abs_diff[idx]:.3e} at point={points[idx[0]]}, component={idx[1]}")
    print(f"max rel diff: {rel_diff[ridx]:.3e} at point={points[ridx[0]]}, component={ridx[1]}")
    assert np.allclose(vector_values, scalar_values, atol=1e-14, rtol=0)


def test_gsl_batched_matches_scalar_loop():
    # The batched C path performs the identical per-point arithmetic in the
    # identical order, only with the workspace allocations hoisted out of the
    # point loop, so the results should agree to the last bit.
    for dim, nodes, gsl_interp, jax_interp in _single_chain_interpolant_cases():
        points = _single_chain_interior_points(nodes, count=512)
        loop_values = np.array([gsl_interp.TPInterpolationND(p) for p in points])
        batch_values = np.asarray(gsl_interp.TPInterpolationND_batched(points))
        assert batch_values.shape == (points.shape[0],)
        max_abs = float(np.max(np.abs(loop_values - batch_values)))
        print(f"{dim}D batched vs per-point loop max abs diff: {max_abs:.3e}")
        assert np.allclose(batch_values, loop_values, atol=1e-15, rtol=0)


def test_gsl_batched_matches_jax_batched():
    diagnostics = []
    for dim, nodes, gsl_interp, jax_interp in _single_chain_interpolant_cases():
        points = _single_chain_interior_points(nodes, count=64)
        # warmup: trigger JIT compilation before assertions
        jax_interp.TPInterpolationND_batched(points[:1])
        gsl_values = np.asarray(gsl_interp.TPInterpolationND_batched(points))
        jax_values = np.asarray(jax_interp.TPInterpolationND_batched(points))
        for point, gsl_value, jax_value in zip(points, gsl_values, jax_values):
            diagnostics.append((dim, point, gsl_value, jax_value))
    actual, expected = _print_max_diffs(diagnostics)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def test_gsl_vector_batched_matches_per_point_and_components():
    for case in (_vector_case_2d(), _vector_case_3d()):
        nodes, component_functions, F = case
        ncomp = F.shape[-1]
        dim = len(nodes)

        vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(ncomp,), F=F)
        points = _single_chain_interior_points(nodes, count=128)

        batch_values = np.asarray(vector_interp.TPInterpolationND_batched(points))
        assert batch_values.shape == (points.shape[0], ncomp)

        # identical arithmetic to the per-point vector evaluation
        loop_values = np.stack([np.asarray(vector_interp.TPInterpolationND(p)) for p in points])
        max_abs = float(np.max(np.abs(batch_values - loop_values)))
        print(f"{dim}D vector batched vs per-point loop max abs diff: {max_abs:.3e}")
        assert np.allclose(batch_values, loop_values, atol=1e-15, rtol=0)

        # per-component scalar batched evaluations
        scalar_values = np.stack(
            [
                np.asarray(
                    TPI.TP_Interpolant_ND(
                        list(nodes), F=np.ascontiguousarray(F[..., comp])
                    ).TPInterpolationND_batched(points)
                )
                for comp in range(ncomp)
            ],
            axis=-1,
        )
        max_abs = float(np.max(np.abs(batch_values - scalar_values)))
        print(f"{dim}D vector batched vs scalar batched max abs diff: {max_abs:.3e}")
        assert np.allclose(batch_values, scalar_values, atol=1e-14, rtol=0)

    # tensor-valued output keeps the trailing value axes per point
    nodes, component_functions, F = _vector_case_2d()
    F_tensor = np.concatenate([F, 2.0 * F], axis=-1).reshape(F.shape[:-1] + (2, 3))
    tensor_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(2, 3), F=F_tensor)
    points = _single_chain_interior_points(nodes, count=16)
    tensor_values = np.asarray(tensor_interp.TPInterpolationND_batched(points))
    assert tensor_values.shape == (points.shape[0], 2, 3)
    single_values = np.stack([np.asarray(tensor_interp.TPInterpolationND(p)) for p in points])
    assert np.allclose(tensor_values, single_values, atol=1e-15, rtol=0)


def test_gsl_batched_boundary_points_match_per_point():
    for dim, nodes, gsl_interp, jax_interp in _single_chain_interpolant_cases():
        lows = np.array([node[0] for node in nodes], dtype=np.float64)
        highs = np.array([node[-1] for node in nodes], dtype=np.float64)
        mids = np.array([node[len(node) // 2] for node in nodes], dtype=np.float64)
        # both corners, a grid node, and points on single-axis boundaries
        points = [lows, highs, mids]
        for axis in range(dim):
            low_edge = mids.copy()
            low_edge[axis] = lows[axis]
            high_edge = mids.copy()
            high_edge[axis] = highs[axis]
            points.extend([low_edge, high_edge])
        points = np.array(points, dtype=np.float64)
        loop_values = np.array([gsl_interp.TPInterpolationND(p) for p in points])
        batch_values = np.asarray(gsl_interp.TPInterpolationND_batched(points))
        max_abs = float(np.max(np.abs(loop_values - batch_values)))
        print(f"{dim}D batched boundary points max abs diff: {max_abs:.3e}")
        assert np.allclose(batch_values, loop_values, atol=1e-15, rtol=0)


def test_gsl_batched_shape_and_range_errors():
    nodes, component_functions, F = _vector_case_2d()
    scalar_interp = TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,), F=F)
    points = _single_chain_interior_points(nodes, count=8)

    for interp in (scalar_interp, vector_interp):
        with pytest.raises(ValueError, match="two-dimensional"):
            interp.TPInterpolationND_batched(points[0])
        with pytest.raises(ValueError, match="two-dimensional"):
            interp.TPInterpolationND_batched(points[:, :, np.newaxis])
        with pytest.raises(ValueError, match="shape"):
            interp.TPInterpolationND_batched(points[:, :1])

        # the error names the first offending point in batch order and its axis
        bad = points.copy()
        bad[5, 1] = 3.0  # y outside [-1, 1]
        bad[6, 0] = 0.5  # x outside [0.1, 0.25]
        with pytest.raises(ValueError, match=r"X\[5, 1\]"):
            interp.TPInterpolationND_batched(bad)

    unset = TPI.TP_Interpolant_ND(list(nodes))
    with pytest.raises(ValueError, match="coefficients"):
        unset.TPInterpolationND_batched(points)


def test_gsl_batched_edge_cases_and_noncontiguous_input():
    nodes, component_functions, F = _vector_case_2d()
    scalar_interp = TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,), F=F)
    points = _single_chain_interior_points(nodes, count=64)

    # empty batch
    assert np.asarray(scalar_interp.TPInterpolationND_batched(points[:0])).shape == (0,)
    assert np.asarray(vector_interp.TPInterpolationND_batched(points[:0])).shape == (0, 3)

    # single-point batch
    single = np.asarray(scalar_interp.TPInterpolationND_batched(points[:1]))
    assert single.shape == (1,)
    assert np.allclose(single[0], scalar_interp.TPInterpolationND(points[0]), atol=1e-15, rtol=0)

    # non-contiguous inputs are copied to contiguous storage internally
    reference = np.asarray(scalar_interp.TPInterpolationND_batched(points))
    strided = np.asarray(scalar_interp.TPInterpolationND_batched(points[::2]))
    assert np.allclose(strided, reference[::2], atol=1e-15, rtol=0)
    fortran = np.asarray(scalar_interp.TPInterpolationND_batched(np.asfortranarray(points)))
    assert np.allclose(fortran, reference, atol=1e-15, rtol=0)


def test_gsl_call_dispatches_single_point_and_batch():
    nodes, component_functions, F = _vector_case_2d()
    scalar_interp = TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    vector_interp = TPI.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,), F=F)
    points = _single_chain_interior_points(nodes, count=16)

    assert np.allclose(
        scalar_interp(points[0]), scalar_interp.TPInterpolationND(points[0]), atol=0, rtol=0
    )
    assert np.allclose(
        np.asarray(scalar_interp(points)),
        np.asarray(scalar_interp.TPInterpolationND_batched(points)),
        atol=0,
        rtol=0,
    )
    assert np.allclose(
        np.asarray(vector_interp(points)),
        np.asarray(vector_interp.TPInterpolationND_batched(points)),
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError):
        scalar_interp(points[:, :, np.newaxis])


def test_jax_call_dispatches_single_point_and_batch():
    nodes, component_functions, F = _vector_case_2d()
    jax_scalar = TPI_jax.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    jax_vector = TPI_jax.TP_Interpolant_ND_Vector(list(nodes), values_shape=(3,), F=F)
    gsl_scalar = TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    points = _single_chain_interior_points(nodes, count=16)

    # warmup: trigger JIT compilation before assertions
    jax_scalar.TPInterpolationND_batched(points[:1])
    jax_vector.TPInterpolationND_batched(points[:1])

    assert np.allclose(
        np.asarray(jax_scalar(points)),
        np.asarray(jax_scalar.TPInterpolationND_batched(points)),
        atol=0,
        rtol=0,
    )
    assert np.asarray(jax_vector(points)).shape == (points.shape[0], 3)
    assert np.allclose(
        np.asarray(jax_vector(points)),
        np.asarray(jax_vector.TPInterpolationND_batched(points)),
        atol=0,
        rtol=0,
    )
    # the two backends' __call__ agree on the same batch
    max_abs = float(np.max(np.abs(np.asarray(jax_scalar(points)) - np.asarray(gsl_scalar(points)))))
    print(f"__call__ batch max abs diff GSL vs JAX: {max_abs:.3e}")
    assert max_abs < 1e-10
    with pytest.raises(ValueError):
        jax_scalar(points[:, :, np.newaxis])


def test_gsl_batched_faster_than_python_loop():
    # The batched path exists to amortize the Python call overhead and the
    # per-call workspace allocations; with 10k points it must beat the loop.
    nodes, component_functions, F = _vector_case_3d()
    interp = TPI.TP_Interpolant_ND(list(nodes), F=np.ascontiguousarray(F[..., 0]))
    points = _single_chain_interior_points(nodes, count=10000)

    loop_times_ms = []
    batch_times_ms = []
    for _ in range(3):
        start = time.perf_counter()
        loop_values = np.array([interp.TPInterpolationND(p) for p in points])
        loop_times_ms.append((time.perf_counter() - start) * 1e3)

        start = time.perf_counter()
        batch_values = np.asarray(interp.TPInterpolationND_batched(points))
        batch_times_ms.append((time.perf_counter() - start) * 1e3)

    assert np.allclose(batch_values, loop_values, atol=1e-15, rtol=0)
    loop_ms = float(np.median(loop_times_ms))
    batch_ms = float(np.median(batch_times_ms))
    print(
        f"3D eval of {points.shape[0]} points: Python loop {loop_ms:.1f} ms, "
        f"batched {batch_ms:.1f} ms ({loop_ms / batch_ms:.1f}x)"
    )
    assert batch_ms < loop_ms


def test_jax_batched_public_path_is_jit_cached_and_tracks_coefficients():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]
    rng = np.random.default_rng(42)
    coeff_shape = tuple(len(node) + 2 for node in nodes)
    coeffs_a = rng.standard_normal(coeff_shape)
    coeffs_b = rng.standard_normal(coeff_shape)

    TPint = TPI_jax.TP_Interpolant_ND(nodes)
    TPint.SetSplineCoefficientsND(coeffs_a)

    lows = np.array([node[0] for node in nodes])
    highs = np.array([node[-1] for node in nodes])
    points = lows + rng.uniform(0.05, 0.95, size=(64, 3)) * (highs - lows)

    # warmup: trigger JIT compilation before assertions
    np.asarray(TPint.TPInterpolationND_batched(points))

    times_ms = []
    for _ in range(10):
        start = time.perf_counter()
        result = TPint.TPInterpolationND_batched(points)
        jax.block_until_ready(result)
        times_ms.append((time.perf_counter() - start) * 1e3)
    median_ms = float(np.median(times_ms))
    print(f"public batched call median: {median_ms:.3f} ms over 10 calls (64 points)")
    # A retrace on every call costs hundreds of ms; a cached call is well under 20.
    assert median_ms < 20.0

    values_a = np.asarray(TPint.TPInterpolationND_batched(points))
    TPint.SetSplineCoefficientsND(coeffs_b)
    values_b = np.asarray(TPint.TPInterpolationND_batched(points))
    scalar_b = np.array(
        [float(np.asarray(TPint.TPInterpolationND(point))) for point in points]
    )
    max_abs = float(np.max(np.abs(values_b - scalar_b)))
    print(f"batched-vs-scalar after coefficient swap max abs diff: {max_abs:.3e}")
    assert not np.allclose(values_a, values_b, atol=1e-12, rtol=0)
    assert np.allclose(values_b, scalar_b, atol=1e-14, rtol=0)


def test_jax_construction_reuses_compiled_setup_across_instances():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    nodes = [xi, yi, zi]
    f = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
    F = f(xx, yy, zz)

    # warmup: first construction compiles the setup path for these grid shapes
    reference = TPI_jax.TP_Interpolant_ND(nodes, F=F)

    times_ms = []
    values = []
    point = np.array([0.1692602, 0.2827312351474, -0.26624193])
    for _ in range(3):
        start = time.perf_counter()
        TPint = TPI_jax.TP_Interpolant_ND(nodes, F=F)
        times_ms.append((time.perf_counter() - start) * 1e3)
        values.append(float(np.asarray(TPint.TPInterpolationND(point))))
    median_ms = float(np.median(times_ms))
    print(f"repeat construction median: {median_ms:.3f} ms over 3 constructions")
    # Retracing the spline-matrix assembly costs hundreds of ms per construction.
    assert median_ms < 150.0

    expected = float(np.asarray(reference.TPInterpolationND(point)))
    assert np.allclose(values, expected, atol=1e-14, rtol=0)


def test_SetSplineCoefficientsND_roundtrip_interpolation_matches_compute():
    for dim, nodes, gsl_interp, _ in _single_chain_interpolant_cases():
        points = _single_chain_interior_points(nodes, count=10)
        computed_values = np.asarray(
            [gsl_interp.TPInterpolationND(point) for point in points], dtype=np.float64
        )

        transferred = TPI.TP_Interpolant_ND(list(nodes))
        transferred.SetSplineCoefficientsND(gsl_interp.GetSplineCoefficientsND())
        transferred_values = np.asarray(
            [transferred.TPInterpolationND(point) for point in points], dtype=np.float64
        )

        max_abs = float(np.max(np.abs(transferred_values - computed_values)))
        print(f"dim={dim} set/compute roundtrip max abs diff: {max_abs:.3e}")
        assert np.allclose(transferred_values, computed_values, atol=1e-14, rtol=0)


def test_TPInterpolationND_large_grid_eval_latency_independent_of_coefficient_size():
    # Guards against re-copying the full coefficient tensor on every evaluation:
    # with a ~7.5M-element coefficient tensor a per-call flatten costs tens of
    # milliseconds, while the evaluation itself only touches a 4^6 block.
    rng = np.random.default_rng(42)
    nodes = [np.sort(rng.uniform(0.0, 1.0, 12)) for _ in range(6)]
    TPint = TPI.TP_Interpolant_ND(nodes)
    coeffs = rng.standard_normal(tuple(len(node) + 2 for node in nodes))
    TPint.SetSplineCoefficientsND(coeffs)

    point = np.array([0.5 * (node[0] + node[-1]) for node in nodes], dtype=np.float64)
    value = TPint.TPInterpolationND(point)
    assert np.isfinite(value)

    times_ms = []
    for _ in range(20):
        start = time.perf_counter()
        TPint.TPInterpolationND(point)
        times_ms.append((time.perf_counter() - start) * 1e3)
    median_ms = float(np.median(times_ms))
    print(f"6D eval with {coeffs.size} coefficients: median {median_ms:.3f} ms over 20 calls")
    assert median_ms < 8.0


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


def _dense_reference_coefficients(nodes, F, values_ndim=0):
    """Reference coefficient solve using the dense assembled spline matrices.

    Solves axis-by-axis with np.linalg.solve on the full (n+2, n+2) matrices,
    so it reproduces the pre-banded implementation up to solver roundoff.
    """
    d = len(nodes)
    F0 = np.pad(F, [(1, 1)] * d + [(0, 0)] * values_ndim, "constant")
    result = F0
    for axis in range(d):
        A, _ = TPI.BsplineBasis1D(nodes[axis]).AssembleSplineMatrix()
        moved = np.moveaxis(result, axis, 0)
        solved = np.linalg.solve(A, moved.reshape(moved.shape[0], -1))
        result = np.moveaxis(solved.reshape(moved.shape), 0, axis)
    return result


def _print_coefficient_diffs(actual, expected):
    abs_diff = np.abs(actual - expected).reshape(-1)
    rel_den = np.maximum(np.abs(expected).reshape(-1), np.finfo(np.float64).tiny)
    rel_diff = abs_diff / rel_den
    max_abs_idx = int(np.argmax(abs_diff))
    max_rel_idx = int(np.argmax(rel_diff))
    max_abs_index = np.unravel_index(max_abs_idx, actual.shape)
    max_rel_index = np.unravel_index(max_rel_idx, actual.shape)
    print(f"max abs diff: {abs_diff[max_abs_idx]:.3e} at index={max_abs_index}")
    print(f"max rel diff: {rel_diff[max_rel_idx]:.3e} at index={max_rel_index}")


def _standard_grids_1d_to_4d():
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    wi = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])
    functions = {
        1: lambda x: np.cos(10.0 * x),
        2: lambda x, y: np.sin(x) * np.arccos(y),
        3: lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
        4: lambda x, y, z, w: np.sin(x) * np.arccos(y) * np.exp(z) * np.cos(w),
    }
    node_sets = {1: (xi,), 2: (xi, yi), 3: (xi, yi, zi), 4: (xi, yi, zi, wi)}
    for dim, nodes in node_sets.items():
        mesh = np.meshgrid(*nodes, indexing="ij")
        F = np.asarray(functions[dim](*mesh), dtype=np.float64)
        yield dim, list(nodes), F


def test_gsl_ComputeSplineCoefficientsND_matches_dense_reference():
    """Banded coefficient solve must match the dense-matrix solve for 1D-4D grids."""
    for dim, nodes, F in _standard_grids_1d_to_4d():
        TPint = TPI.TP_Interpolant_ND(nodes)
        TPint.ComputeSplineCoefficientsND(F)
        actual = np.asarray(TPint.GetSplineCoefficientsND())
        expected = _dense_reference_coefficients(nodes, F)
        assert actual.shape == expected.shape
        print(f"dim={dim}")
        _print_coefficient_diffs(actual, expected)
        assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def test_gsl_vector_ComputeSplineCoefficientsND_matches_dense_reference():
    """Vector-valued banded coefficient solve must match the dense-matrix solve."""
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    nodes = [xi, yi]
    values_shape = (2, 3)
    xx, yy = np.meshgrid(xi, yi, indexing="ij")
    base = np.sin(xx) * np.arccos(yy)
    scales = np.arange(1.0, 7.0).reshape(values_shape)
    F = base[..., None, None] * scales

    TPint = TPI.TP_Interpolant_ND_Vector(nodes, values_shape)
    TPint.ComputeSplineCoefficientsND(F)
    actual = np.asarray(TPint.GetSplineCoefficientsND())
    expected = _dense_reference_coefficients(nodes, F, values_ndim=len(values_shape))
    assert actual.shape == expected.shape
    _print_coefficient_diffs(actual, expected)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def test_banded_notaknot_boundary_rows_match_gsl():
    """TPI_banded's NumPy boundary rows must match the GSL dense matrix rows."""
    import TPI_banded

    grids = [
        np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),
        np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]),
        np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5]),
        np.sort(np.random.default_rng(42).uniform(0.0, 10.0, 30)),
    ]
    for x in grids:
        A, _ = TPI.BsplineBasis1D(x).AssembleSplineMatrix()
        row_first, row_last = TPI_banded.notaknot_boundary_rows(x)
        actual = np.stack((row_first, row_last))
        expected = np.stack((A[0], A[-1]))
        print(f"n={len(x)}")
        _print_coefficient_diffs(actual, expected)
        # 3rd-derivative rows scale like 1/h^3 (up to ~1e9 on fine grids), so
        # agreement is asserted relative to the entry magnitude.
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_jax_ComputeSplineCoefficientsND_large_1d_grid():
    """JAX setup and coefficient solve on a 200k-point 1D grid must be O(n) memory.

    Before the banded solve, TPInterpolationSetupND dense-LU-factored the
    (n+2)^2 spline matrix, which attempted a ~320 GB allocation here.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    n = 200_000
    x = np.sort(rng.uniform(0.0, 100.0, n))
    x[0] = 0.0
    x[-1] = 100.0
    F = np.sin(x) * np.exp(-0.01 * x)

    TPint = TPI_jax.TP_Interpolant_ND([x], F=F)

    xq = np.sort(rng.uniform(0.0, 100.0, 30))
    actual = np.asarray(TPint.TPInterpolationND_batched(xq[:, None]))
    expected = CubicSpline(x, F, bc_type="not-a-knot")(xq)

    _print_coefficient_diffs(actual, expected)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def test_gsl_ComputeSplineCoefficientsND_large_1d_grid():
    """A 200k-point 1D grid must not assemble dense (n+2)^2 matrices.

    Before the banded solve this attempted a ~320 GB allocation; now the
    whole solve is O(n) memory. Accuracy is checked against scipy's
    not-a-knot cubic spline.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    n = 200_000
    x = np.sort(rng.uniform(0.0, 100.0, n))
    x[0] = 0.0
    x[-1] = 100.0
    F = np.sin(x) * np.exp(-0.01 * x)

    TPint = TPI.TP_Interpolant_ND([x], F=F)

    xq = np.sort(rng.uniform(0.0, 100.0, 30))
    actual = TPint.TPInterpolationND_batched(xq[:, None])
    expected = CubicSpline(x, F, bc_type="not-a-knot")(xq)

    _print_coefficient_diffs(actual, expected)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)


def _spline1d_case_grids():
    rng = np.random.default_rng(42)
    for n in (10, 137, 2000):
        x = np.sort(rng.uniform(0.0, 10.0, n))
        x[0] = 0.0
        x[-1] = 10.0
        x = np.unique(x)
        F = np.sin(3.0 * x) * np.exp(-0.1 * x)
        yield x, F


def test_jax_Spline1D_matches_general_path_and_scipy():
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    for x, F in _spline1d_case_grids():
        spline = TPI_jax.Spline1D(x, F=F)
        general = TPI.TP_Interpolant_ND([x], F=F)
        reference = CubicSpline(x, F, bc_type="not-a-knot")

        xq = np.sort(rng.uniform(x[0], x[-1], 30))
        actual = np.asarray(spline(xq))
        expected_general = general.TPInterpolationND_batched(xq[:, None])
        expected_scipy = reference(xq)

        print(f"n={len(x)} vs general path:")
        _print_coefficient_diffs(actual, expected_general)
        print(f"n={len(x)} vs scipy CubicSpline:")
        _print_coefficient_diffs(actual, expected_scipy)
        assert np.allclose(actual, expected_general, atol=1e-10, rtol=0)
        assert np.allclose(actual, expected_scipy, atol=1e-10, rtol=0)


def test_jax_Spline1D_boundary_and_node_points():
    x, F = next(_spline1d_case_grids())
    spline = TPI_jax.Spline1D(x, F=F)
    # endpoints, plus a sample of interior nodes: 10 points total
    idx = np.linspace(0, len(x) - 1, 10).astype(int)
    actual = np.asarray(spline(x[idx]))
    _print_coefficient_diffs(actual, F[idx])
    assert np.allclose(actual, F[idx], atol=1e-13, rtol=0)
    # scalar input returns a scalar-shaped result
    single = np.asarray(spline(float(x[3])))
    assert single.shape == ()
    assert np.allclose(single, F[3], atol=1e-13, rtol=0)


def test_jax_Spline1D_stress_geometric_grid_reproduces_cubic():
    x = np.unique(np.concatenate(([0.0], np.geomspace(1e-6, 1.0, 199))))
    f = lambda t: 2.0 - 3.0 * t + 4.0 * t ** 2 - 5.0 * t ** 3
    spline = TPI_jax.Spline1D(x, F=f(x))
    xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))
    actual = np.asarray(spline(xq))
    expected = f(xq)
    _print_coefficient_diffs(actual, expected)
    assert np.allclose(actual, expected, atol=1e-12, rtol=0)


def test_jax_Spline1D_jit_vmap_grad():
    from scipy.interpolate import CubicSpline

    x, F = next(_spline1d_case_grids())
    spline = TPI_jax.Spline1D(x, F=F)
    reference = CubicSpline(x, F, bc_type="not-a-knot")
    xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))

    # jit of the functional construction, with traced F and traced x
    build = jax.jit(TPI_jax.spline_1d_hermite)
    # warmup: trigger JIT compilation before assertions
    jit_poly = build(jnp_x := jax.numpy.asarray(x), jax.numpy.asarray(F))
    eager_poly = TPI_jax.spline_1d_hermite(jnp_x, jax.numpy.asarray(F))
    for jit_c, eager_c in zip(jit_poly, eager_poly):
        assert np.allclose(np.asarray(jit_c), np.asarray(eager_c), atol=1e-12, rtol=0)

    # evaluation under jit + vmap over query points
    eval_one = lambda q: spline(q)
    batched = jax.jit(jax.vmap(eval_one))
    # warmup: trigger JIT compilation before assertions
    batched(xq)
    actual = np.asarray(batched(xq))
    assert np.allclose(actual, np.asarray(spline(xq)), atol=1e-14, rtol=0)

    # gradient w.r.t. the query point: finite and matches scipy's derivative
    grad_fn = jax.jit(jax.vmap(jax.grad(eval_one)))
    # warmup: trigger JIT compilation before assertions
    grad_fn(xq)
    grads = np.asarray(grad_fn(xq))
    expected_grads = reference.derivative()(xq)
    assert np.isfinite(grads).all()
    _print_coefficient_diffs(grads, expected_grads)
    assert np.allclose(grads, expected_grads, atol=1e-8, rtol=0)


def test_jax_Spline1D_coefficient_interop():
    x, F = next(_spline1d_case_grids())
    spline = TPI_jax.Spline1D(x, F=F)
    general = TPI.TP_Interpolant_ND([x], F=F)

    # to_coefficients: standard TPI format, matching the general path
    coeffs = np.asarray(spline.to_coefficients())
    expected = np.asarray(general.GetSplineCoefficientsND())
    assert coeffs.shape == (len(x) + 2,)
    _print_coefficient_diffs(coeffs, expected)
    assert np.allclose(coeffs, expected, atol=1e-10, rtol=0)

    # the exported coefficients round-trip through the general interpolant
    xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))
    roundtrip = TPI.TP_Interpolant_ND([x], coeffs=coeffs)
    assert np.allclose(
        roundtrip.TPInterpolationND_batched(xq[:, None]),
        np.asarray(spline(xq)),
        atol=1e-12,
        rtol=0,
    )

    # from_coefficients: loading old saved data, including coefficients that
    # did not come from a not-a-knot solve (arbitrary C^2 spline in B-form)
    rng = np.random.default_rng(7)
    arbitrary = rng.standard_normal(len(x) + 2)
    loaded = TPI_jax.Spline1D(x, coeffs=arbitrary)
    reference = TPI.TP_Interpolant_ND([x], coeffs=arbitrary)
    actual = np.asarray(loaded(xq))
    expected_eval = reference.TPInterpolationND_batched(xq[:, None])
    _print_coefficient_diffs(actual, expected_eval)
    assert np.allclose(actual, expected_eval, atol=1e-12, rtol=0)


def test_jax_Spline1D_error_behavior():
    x = np.linspace(0.0, 1.0, 10)
    F = np.sin(x)

    with pytest.raises(ValueError):
        TPI_jax.Spline1D(np.array([0.0, 1.0, 2.0]), F=np.zeros(3))  # n < 4
    with pytest.raises(ValueError):
        TPI_jax.Spline1D(np.array([0.0, 1.0, 1.0, 2.0]), F=np.zeros(4))  # not strictly increasing
    with pytest.raises(ValueError):
        TPI_jax.Spline1D(x, F=np.zeros(11))  # wrong data length
    with pytest.raises(ValueError):
        TPI_jax.Spline1D(x, coeffs=np.zeros(13))  # wrong coefficient length

    spline = TPI_jax.Spline1D(x, F=F)
    with pytest.raises(ValueError):
        spline(np.array([-0.5]))
    with pytest.raises(ValueError):
        spline(1.5)
    # endpoints are accepted
    values = np.asarray(spline(np.array([0.0, 1.0])))
    assert np.allclose(values, [F[0], F[-1]], atol=1e-13, rtol=0)


def test_gsl_Spline1D_matches_general_path_and_scipy():
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    for x, F in _spline1d_case_grids():
        spline = TPI.Spline1D(x, F=F)
        general = TPI.TP_Interpolant_ND([x], F=F)
        reference = CubicSpline(x, F, bc_type="not-a-knot")

        xq = np.sort(rng.uniform(x[0], x[-1], 30))
        actual = np.asarray(spline(xq))
        expected_general = general.TPInterpolationND_batched(xq[:, None])
        expected_scipy = reference(xq)

        print(f"n={len(x)} vs general path:")
        _print_coefficient_diffs(actual, expected_general)
        print(f"n={len(x)} vs scipy CubicSpline:")
        _print_coefficient_diffs(actual, expected_scipy)
        assert np.allclose(actual, expected_general, atol=1e-10, rtol=0)
        assert np.allclose(actual, expected_scipy, atol=1e-10, rtol=0)


def test_Spline1D_cross_backend_parity():
    for x, F in _spline1d_case_grids():
        cpu = TPI.Spline1D(x, F=F)
        jax_spline = TPI_jax.Spline1D(x, F=F)
        xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))
        actual = np.asarray(cpu(xq))
        expected = np.asarray(jax_spline(xq))
        print(f"n={len(x)} CPU vs JAX Spline1D:")
        _print_coefficient_diffs(actual, expected)
        assert np.allclose(actual, expected, atol=1e-12, rtol=0)


def test_gsl_Spline1D_sorted_walk_edge_cases():
    x, F = next(_spline1d_case_grids())
    spline = TPI.Spline1D(x, F=F)
    rng = np.random.default_rng(42)

    # duplicate query points, endpoints included, node hits
    xq = np.sort(np.concatenate((
        [x[0], x[0], x[-1], x[-1]], x[3:6], [x[4], x[4]],
        rng.uniform(x[0], x[-1], 21),
    )))
    sorted_result = np.asarray(spline(xq))

    # all queries inside one interval
    inner = np.linspace(x[2], np.nextafter(x[3], x[2]), 10)
    one_interval = np.asarray(spline(inner))
    from scipy.interpolate import CubicSpline
    reference = CubicSpline(x, F, bc_type="not-a-knot")
    assert np.allclose(one_interval, reference(inner), atol=1e-12, rtol=0)

    # unsorted queries take the fallback path and must agree with the walk
    perm = rng.permutation(len(xq))
    unsorted_result = np.asarray(spline(xq[perm]))
    _print_coefficient_diffs(unsorted_result, sorted_result[perm])
    assert np.array_equal(unsorted_result, sorted_result[perm])
    assert np.allclose(sorted_result, reference(xq), atol=1e-12, rtol=0)


def test_gsl_Spline1D_coefficient_interop():
    x, F = next(_spline1d_case_grids())
    spline = TPI.Spline1D(x, F=F)
    general = TPI.TP_Interpolant_ND([x], F=F)

    coeffs = np.asarray(spline.to_coefficients())
    expected = np.asarray(general.GetSplineCoefficientsND())
    assert coeffs.shape == (len(x) + 2,)
    _print_coefficient_diffs(coeffs, expected)
    assert np.allclose(coeffs, expected, atol=1e-10, rtol=0)

    xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))
    rng = np.random.default_rng(7)
    arbitrary = rng.standard_normal(len(x) + 2)
    loaded = TPI.Spline1D(x, coeffs=arbitrary)
    reference = TPI.TP_Interpolant_ND([x], coeffs=arbitrary)
    actual = np.asarray(loaded(xq))
    expected_eval = reference.TPInterpolationND_batched(xq[:, None])
    _print_coefficient_diffs(actual, expected_eval)
    assert np.allclose(actual, expected_eval, atol=1e-12, rtol=0)


def test_gsl_Spline1D_error_behavior():
    x = np.linspace(0.0, 1.0, 10)
    F = np.sin(x)

    with pytest.raises(ValueError):
        TPI.Spline1D(np.array([0.0, 1.0, 2.0]), F=np.zeros(3))
    with pytest.raises(ValueError):
        TPI.Spline1D(np.array([0.0, 1.0, 1.0, 2.0]), F=np.zeros(4))
    with pytest.raises(ValueError):
        TPI.Spline1D(x, F=np.zeros(11))
    with pytest.raises(ValueError):
        TPI.Spline1D(x, coeffs=np.zeros(13))

    spline = TPI.Spline1D(x, F=F)
    with pytest.raises(ValueError):
        spline(np.array([-0.5]))
    with pytest.raises(ValueError):
        spline(1.5)
    values = np.asarray(spline(np.array([0.0, 1.0])))
    assert np.allclose(values, [F[0], F[-1]], atol=1e-13, rtol=0)
    single = np.asarray(spline(float(x[3])))
    assert single.shape == ()
    assert np.allclose(single, F[3], atol=1e-13, rtol=0)


def test_Spline1D_recompute_coefficients_on_fixed_grid():
    """ComputeSplineCoefficients(F) refits new data reusing the instance.

    The refit must match a freshly constructed Spline1D exactly, in both
    backends, including after the instance was loaded from coefficients.
    """
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0.0, 10.0, 137))
    x[0], x[-1] = 0.0, 10.0
    F1 = np.sin(3.0 * x) * np.exp(-0.1 * x)
    F2 = np.cos(2.0 * x) + 0.1 * x
    xq = np.sort(rng.uniform(0.0, 10.0, 30))

    for module in (TPI, TPI_jax):
        spline = module.Spline1D(x, F=F1)
        before = np.asarray(spline(xq))

        result = spline.ComputeSplineCoefficients(F2)
        assert result is None
        actual = np.asarray(spline(xq))
        expected = np.asarray(module.Spline1D(x, F=F2)(xq))
        print(f"{module.__name__} refit vs fresh instance:")
        _print_coefficient_diffs(actual, expected)
        assert np.array_equal(actual, expected)
        assert not np.allclose(actual, before, atol=1e-6, rtol=0)

        # refit works on an instance that was loaded from coefficients
        loaded = module.Spline1D(x, coeffs=module.Spline1D(x, F=F1).to_coefficients())
        loaded.ComputeSplineCoefficients(F2)
        assert np.array_equal(np.asarray(loaded(xq)), expected)

        with pytest.raises(ValueError):
            spline.ComputeSplineCoefficients(np.zeros(len(x) + 1))


def test_gsl_Spline1D_pivoting_stress_grid():
    """Adversarial node spacings exercise the pivoted tridiagonal solve.

    Log-uniform spacings spanning ~11 orders of magnitude. The not-a-knot
    boundary rows are not diagonally dominant, so a non-pivoting Thomas
    solve loses digits here (measured ~4e-8 relative error on this grid);
    a partial-pivoting solve (same algorithm class as scipy's dgtsv) must
    stay at scipy-parity accuracy.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(96)
    n = int(rng.integers(6, 400))
    span = float(rng.uniform(4, 14))
    dx = 10.0 ** rng.uniform(-span, 0.0, n - 1)
    x = np.concatenate(([0.0], np.cumsum(dx)))
    assert np.all(np.diff(x) > 0)
    F = rng.standard_normal(n)
    spline = TPI.Spline1D(x, F=F)
    reference = CubicSpline(x, F, bc_type="not-a-knot")

    xq = np.sort(rng.uniform(x[0], x[-1], 50))
    actual = np.asarray(spline(xq))
    expected = reference(xq)
    scale = max(1.0, np.max(np.abs(expected)))
    _print_coefficient_diffs(actual, expected)
    assert np.max(np.abs(actual - expected)) / scale < 1e-10

    # refitting on the stress grid matches a fresh instance exactly
    F2 = rng.standard_normal(n)
    spline.ComputeSplineCoefficients(F2)
    fresh = TPI.Spline1D(x, F=F2)
    assert np.array_equal(np.asarray(spline(xq)), np.asarray(fresh(xq)))


def test_gsl_Spline1D_large_n_scipy_parity():
    from scipy.interpolate import CubicSpline

    x = np.unique(np.concatenate(([0.0], np.geomspace(1e-6, 1.0, 50_000))))
    F = np.sin(20.0 * x) * np.exp(-x)
    spline = TPI.Spline1D(x, F=F)
    reference = CubicSpline(x, F, bc_type="not-a-knot")

    xq = np.sort(np.random.default_rng(42).uniform(x[0], x[-1], 30))
    actual = np.asarray(spline(xq))
    expected = reference(xq)
    _print_coefficient_diffs(actual, expected)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)

    coeffs = np.asarray(spline.to_coefficients())
    assert coeffs.shape == (len(x) + 2,)
    roundtrip = TPI.Spline1D(x, coeffs=coeffs)
    rt = np.asarray(roundtrip(xq))
    _print_coefficient_diffs(rt, actual)
    assert np.allclose(rt, actual, atol=1e-10, rtol=0)


def test_gsl_Spline1D_node_validation_messages():
    """Exact validation errors on the F construction path.

    Finiteness is reported before monotonicity when both are violated, and
    NaN in the data values (as opposed to the nodes) is not validated.
    """
    x = np.linspace(0.0, 1.0, 10)

    with pytest.raises(ValueError, match="strictly increasing"):
        TPI.Spline1D(np.array([0.0, 1.0, 0.5, 2.0]), F=np.zeros(4))
    with pytest.raises(ValueError, match="must be finite"):
        TPI.Spline1D(np.array([0.0, 1.0, np.nan, 2.0]), F=np.zeros(4))
    with pytest.raises(ValueError, match="must be finite"):
        TPI.Spline1D(np.array([0.0, np.inf, 1.0, 0.5]), F=np.zeros(4))
    with pytest.raises(ValueError, match="one-dimensional"):
        TPI.Spline1D(np.zeros((4, 2)), F=np.zeros(8))
    with pytest.raises(ValueError, match="at least four"):
        TPI.Spline1D(np.array([0.0, 1.0, 2.0]), F=np.zeros(3))

    # NaN in F does not raise; it propagates like in the previous solver
    nan_F = np.sin(x)
    nan_F[3] = np.nan
    TPI.Spline1D(x, F=nan_F)


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
