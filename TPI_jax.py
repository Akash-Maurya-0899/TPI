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


def construct_knots(nodes):
    """Construct the cubic B-spline knot vector matching GSL's convention."""
    nodes_array = _as_valid_nodes(nodes)
    return _construct_knots_jax(nodes_array)
