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


def construct_knots(nodes):
    """Construct the cubic B-spline knot vector matching GSL's convention."""
    nodes_array = _as_valid_nodes(nodes)
    return _construct_knots_jax(nodes_array)
