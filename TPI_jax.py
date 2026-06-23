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
    idx = jnp.arange(knots.shape[0] - 1)
    endpoint = x == knots[-1]

    # Degree-0 basis functions live on the knot spans.
    # The final span is closed on the right so x == knots[-1] maps to the last basis.
    span_mask = (knots[:-1] <= x) & (
        (x < knots[1:]) | ((idx == idx[-1]) & (x <= knots[1:]))
    )
    basis = jnp.where(span_mask, 1.0, 0.0)

    def body_fun(degree, basis_vec):
        left_knots = jnp.take(knots, idx, mode="clip")
        left_knots_d = jnp.take(knots, idx + degree, mode="clip")
        right_knots = jnp.take(knots, idx + 1, mode="clip")
        right_knots_d = jnp.take(knots, idx + degree + 1, mode="clip")

        left_denom = left_knots_d - left_knots
        right_denom = right_knots_d - right_knots
        active = idx < (basis_vec.shape[0] - degree)

        left = jnp.where(
            left_denom != 0.0,
            (x - left_knots) / left_denom * basis_vec,
            0.0,
        )
        right = jnp.where(
            right_denom != 0.0,
            (right_knots_d - x) / right_denom * jnp.take(basis_vec, idx + 1, mode="clip"),
            0.0,
        )
        next_basis = jnp.where(active, left + right, 0.0)
        endpoint_basis = jnp.where(idx == (basis_vec.shape[0] - degree - 1), 1.0, 0.0)
        next_basis = jnp.where(endpoint, endpoint_basis, next_basis)
        return next_basis

    basis = jax.lax.fori_loop(1, 4, body_fun, basis)
    return basis[:nbasis]


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
