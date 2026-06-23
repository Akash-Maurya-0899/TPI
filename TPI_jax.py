# coding: utf-8

"""JAX backend helpers for TPI.

JAX defaults to 32-bit floating point. The GSL/Cython backend uses double
precision, so x64 is enabled before any JAX arrays are constructed.
"""

import numpy as np

import jax

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


def construct_knots(nodes):
    """Construct the cubic B-spline knot vector matching GSL's convention."""
    nodes_array = _as_valid_nodes(nodes)
    return _construct_knots_jax(nodes_array)
