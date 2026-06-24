# coding: utf-8

"""JAX backend helpers for TPI.

JAX defaults to 32-bit floating point. The GSL/Cython backend uses double
precision, so x64 is enabled before any JAX arrays are constructed.
"""

import numpy as np

import jax
from jax import core as jax_core

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _as_valid_nodes(nodes):
    if nodes is None:
        raise ValueError("Missing input nodes.")

    try:
        nodes_array = np.asarray(nodes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input nodes must be a one-dimensional real array.") from exc

    if nodes_array.ndim != 1:
        raise ValueError("Input nodes must be one-dimensional.")
    if nodes_array.shape[0] < 2:
        raise ValueError("Require at least two input nodes.")
    if np.iscomplexobj(nodes_array):
        raise ValueError("Input nodes must be real.")

    nodes_array = nodes_array.astype(np.float64)

    if not np.isfinite(nodes_array).all():
        raise ValueError("Input nodes must be finite.")
    if (np.diff(nodes_array) < 0).any():
        raise ValueError("Input nodes must be non-decreasing.")

    return jnp.asarray(nodes_array, dtype=jnp.float64)


def _construct_knots_jax(nodes):
    start = jnp.repeat(nodes[:1], 3)
    end = jnp.repeat(nodes[-1:], 3)
    return jnp.concatenate((start, nodes, end))


def _evaluate_cubic_bspline_basis_jax(knots, x):
    """Evaluate the full cubic B-spline basis on a clamped knot vector."""
    knots = jnp.asarray(knots, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    nbasis = knots.shape[0] - 4
    span_count = knots.shape[0] - 7
    span_idx = jnp.arange(span_count, dtype=jnp.int64)

    # Degree-0 basis is defined over the positive-width knot spans.
    # The final span is closed on the right to encode the clamped B-spline boundary convention.
    span_left = knots[3:-4]
    span_right = knots[4:-3]
    span_mask = (span_left <= x) & (
        (x < span_right) | ((span_idx == span_idx[-1]) & (x <= span_right))
    )
    span = jnp.sum(span_idx * span_mask.astype(span_idx.dtype)) + 3

    lefts = jnp.stack(
        (
            x - knots[span],
            x - knots[span - 1],
            x - knots[span - 2],
        )
    )
    rights = jnp.stack(
        (
            knots[span + 1] - x,
            knots[span + 2] - x,
            knots[span + 3] - x,
        )
    )

    basis = jnp.array((1.0, 0.0, 0.0, 0.0), dtype=jnp.float64)

    def degree_body(j, carry):
        basis_vec = carry

        def inner_body(r, inner_carry):
            basis_inner, saved = inner_carry
            denom = rights[r] + lefts[j - r]
            temp = jnp.where(denom != 0.0, basis_inner[r] / denom, 0.0)
            basis_inner = basis_inner.at[r].set(saved + rights[r] * temp)
            saved = lefts[j - r] * temp
            return basis_inner, saved

        basis_vec, saved = jax.lax.fori_loop(0, j + 1, inner_body, (basis_vec, 0.0))
        basis_vec = basis_vec.at[j + 1].set(saved)
        return basis_vec

    basis = jax.lax.fori_loop(0, 3, degree_body, basis)
    result = jnp.zeros(nbasis, dtype=jnp.float64)
    result = jax.lax.dynamic_update_slice(result, basis, (span - 3,))
    return result


def _evaluate_cubic_bspline_3rd_derivatives_jax(knots, x):
    """Evaluate the full cubic B-spline 3rd derivative basis on a clamped knot vector."""
    knots = jnp.asarray(knots, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    nbasis = knots.shape[0] - 4

    span_count = knots.shape[0] - 7
    span_idx = jnp.arange(span_count, dtype=jnp.int64)
    span_left = knots[3:-4]
    span_right = knots[4:-3]
    span_mask = (span_left <= x) & (
        (x < span_right) | ((span_idx == span_idx[-1]) & (x <= span_right))
    )
    span = jnp.sum(span_idx * span_mask.astype(span_idx.dtype)) + 3

    p = 3
    ndu = jnp.zeros((p + 1, p + 1), dtype=jnp.float64)
    ndu = ndu.at[0, 0].set(1.0)
    left = jnp.zeros(p + 1, dtype=jnp.float64)
    right = jnp.zeros(p + 1, dtype=jnp.float64)

    for j in range(1, p + 1):
        left = left.at[j].set(x - knots[span + 1 - j])
        right = right.at[j].set(knots[span + j] - x)
        saved = 0.0
        for r in range(j):
            ndu = ndu.at[j, r].set(right[r + 1] + left[j - r])
            denom = ndu[j, r]
            temp = jnp.where(denom != 0.0, ndu[r, j - 1] / denom, 0.0)
            ndu = ndu.at[r, j].set(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
        ndu = ndu.at[j, j].set(saved)

    ders = jnp.zeros((p + 1, p + 1), dtype=jnp.float64)
    ders = ders.at[0].set(ndu[:, p])

    for r in range(p + 1):
        a = jnp.zeros((2, p + 1), dtype=jnp.float64)
        a = a.at[0, 0].set(1.0)
        s1 = 0
        s2 = 1

        for k in range(1, p + 1):
            d = 0.0
            rk = r - k
            pk = p - k

            if r >= k:
                denom = ndu[pk + 1, rk]
                value = jnp.where(denom != 0.0, a[s1, 0] / denom, 0.0)
                a = a.at[s2, 0].set(value)
                d = value * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                denom = ndu[pk + 1, rk + j]
                value = jnp.where(denom != 0.0, (a[s1, j] - a[s1, j - 1]) / denom, 0.0)
                a = a.at[s2, j].set(value)
                d = d + value * ndu[rk + j, pk]

            if r <= pk:
                denom = ndu[pk + 1, r]
                value = jnp.where(denom != 0.0, -a[s1, k - 1] / denom, 0.0)
                a = a.at[s2, k].set(value)
                d = d + value * ndu[r, pk]

            ders = ders.at[k, r].set(d)
            s1, s2 = s2, s1

    ders = ders.at[1].set(ders[1] * 3.0)
    ders = ders.at[2].set(ders[2] * 6.0)
    ders = ders.at[3].set(ders[3] * 6.0)

    result = jnp.zeros(nbasis, dtype=jnp.float64)
    result = jax.lax.dynamic_update_slice(result, ders[3], (span - 3,))
    return result


class BsplineBasis1D:
    """JAX-side 1D cubic B-spline basis helper."""

    def __init__(self, xvec_in):
        self.xi = _as_valid_nodes(xvec_in)
        self.knots = _construct_knots_jax(self.xi)
        self.nbasis = int(self.knots.shape[0] - 4)

    def EvaluateBsplines(self, x):
        if isinstance(x, jax_core.Tracer):
            return _evaluate_cubic_bspline_basis_jax(self.knots, x)

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 0:
            raise ValueError("Evaluation point x must be scalar.")

        x_val = float(x_arr)
        x_min = float(np.asarray(self.xi[0]))
        x_max = float(np.asarray(self.xi[-1]))
        if x_val < x_min or x_val > x_max:
            raise ValueError(
                f"Error: Bspline_basis_1D(): x: {x_val} is outside of knots "
                f"vector with bounds [{x_min}, {x_max}]!"
            )

        return _evaluate_cubic_bspline_basis_jax(self.knots, x_val)

    def EvaluateBsplines3rdDerivatives(self, x):
        if isinstance(x, jax_core.Tracer):
            return _evaluate_cubic_bspline_3rd_derivatives_jax(self.knots, x)

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 0:
            raise ValueError("Evaluation point x must be scalar.")

        x_val = float(x_arr)
        x_min = float(np.asarray(self.xi[0]))
        x_max = float(np.asarray(self.xi[-1]))
        if x_val < x_min or x_val > x_max:
            raise ValueError(
                f"Error: Bspline_basis_3rd_derivative_1D(): x: {x_val} is outside of knots "
                f"vector with bounds [{x_min}, {x_max}]!"
            )

        return _evaluate_cubic_bspline_3rd_derivatives_jax(self.knots, x_val)

    def AssembleSplineMatrix(self):
        """Assemble the cubic spline matrix with not-a-knot boundary conditions."""
        return _assemble_spline_matrix_jax(self.xi)


def construct_knots(nodes):
    """Construct the cubic B-spline knot vector matching GSL's convention."""
    nodes_array = _as_valid_nodes(nodes)
    return _construct_knots_jax(nodes_array)


def _assemble_spline_matrix_jax(nodes):
    """Assemble the cubic spline matrix for a validated 1D node array."""
    knots = _construct_knots_jax(nodes)
    phi_internal = jax.vmap(lambda x: _evaluate_cubic_bspline_basis_jax(knots, x))(nodes)

    xi12mean = 0.5 * (nodes[0] + nodes[1])
    xi23mean = 0.5 * (nodes[1] + nodes[2])
    xim32mean = 0.5 * (nodes[-3] + nodes[-2])
    xim21mean = 0.5 * (nodes[-2] + nodes[-1])

    first_row = (
        _evaluate_cubic_bspline_3rd_derivatives_jax(knots, xi12mean)
        - _evaluate_cubic_bspline_3rd_derivatives_jax(knots, xi23mean)
    )
    last_row = (
        _evaluate_cubic_bspline_3rd_derivatives_jax(knots, xim32mean)
        - _evaluate_cubic_bspline_3rd_derivatives_jax(knots, xim21mean)
    )

    phi = jnp.vstack((first_row, phi_internal, last_row))
    return phi, knots


def _solve_axis_system(matrix, tensor, axis):
    rhs = jnp.moveaxis(tensor, axis, 0)
    leading = rhs.shape[0]
    solved = jnp.linalg.solve(matrix, rhs.reshape((leading, -1)))
    solved = solved.reshape(rhs.shape)
    return jnp.moveaxis(solved, 0, axis)


def compute_spline_coefficients_nd(nodes, F):
    """Compute tensor-product spline coefficients for validated nodes and data."""
    nodes = tuple(jnp.asarray(node, dtype=jnp.float64) for node in nodes)
    F = jnp.asarray(F, dtype=jnp.float64)

    dims = tuple(int(node.shape[0]) for node in nodes)
    if F.shape != dims:
        raise ValueError(f"Data on TP grid should have shape {list(dims)}")

    coeffs = jnp.pad(F, [(1, 1)] * len(nodes), mode="constant")
    for axis in range(len(nodes) - 1, -1, -1):
        phi, _ = _assemble_spline_matrix_jax(nodes[axis])
        coeffs = _solve_axis_system(phi, coeffs, axis)
    return coeffs


class TP_Interpolant_ND:
    """JAX-side tensor-product spline interpolant helper."""

    def __init__(self, nodes, coeffs=None, F=None):
        if nodes is None:
            raise ValueError("Missing input nodes.")
        if not isinstance(nodes, (list, tuple)):
            raise TypeError("Expected list of numpy.ndarrays.")
        if not np.array(list(map(lambda x: isinstance(x, np.ndarray), nodes))).all():
            raise TypeError("Expected list of numpy.ndarrays.")

        self.nodes = tuple(_as_valid_nodes(node) for node in nodes)
        self.n = len(self.nodes)
        self.c = None
        self.knots_list = None
        self.bases = None

        self.TPInterpolationSetupND()
        if coeffs is not None:
            self.SetSplineCoefficientsND(coeffs)
        if F is not None:
            self.ComputeSplineCoefficientsND(F)

    def TPInterpolationSetupND(self):
        self.bases = tuple(BsplineBasis1D(np.asarray(node)) for node in self.nodes)
        self.knots_list = tuple(base.knots for base in self.bases)

    def ComputeSplineCoefficientsND(self, F):
        coeffs = compute_spline_coefficients_nd(self.nodes, F)
        self.c = coeffs
        return coeffs

    def GetSplineCoefficientsND(self):
        return self.c

    def SetSplineCoefficientsND(self, coeffs):
        dims = tuple(len(node) + 2 for node in self.nodes)
        coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        if coeffs.shape != dims:
            raise ValueError(f"Spline coefficients should have shape {list(dims)}")
        self.c = coeffs
