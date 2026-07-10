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

import TPI_banded


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


def _find_active_cubic_bspline_span_jax(knots, x):
    """Return the 4 active cubic basis values and the first active index."""
    knots = jnp.asarray(knots, dtype=jnp.float64)
    x = jnp.asarray(x, dtype=jnp.float64)
    span = jnp.searchsorted(knots[3:-3], x, side="right") + 2
    span = jnp.clip(span, 3, knots.shape[0] - 5)

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
    return basis, span - 3


def _pack_axis_eval_arrays(knots_list):
    """Pack per-axis knot vectors into padded matrices for the single-point evaluator.

    Axes may have different knot counts, so rows are padded with +inf. Padded
    entries never affect the span search (inf <= x is always false for in-range x)
    and are never gathered because the span is clipped per axis before indexing.
    """
    knots_np = [np.asarray(knots, dtype=np.float64) for knots in knots_list]
    inner = [knots[3:-3] for knots in knots_np]
    inner_pad = np.full((len(knots_np), max(len(v) for v in inner)), np.inf, dtype=np.float64)
    for i, v in enumerate(inner):
        inner_pad[i, : len(v)] = v
    knots_pad = np.full((len(knots_np), max(len(k) for k in knots_np)), np.inf, dtype=np.float64)
    for i, k in enumerate(knots_np):
        knots_pad[i, : len(k)] = k
    span_max = np.array([len(k) - 5 for k in knots_np], dtype=np.int64)
    return jnp.asarray(inner_pad), jnp.asarray(knots_pad), jnp.asarray(span_max)


def _evaluate_cubic_bspline_basis_jax(knots, x):
    """Evaluate the full cubic B-spline basis on a clamped knot vector."""
    basis, start = _find_active_cubic_bspline_span_jax(knots, x)
    nbasis = jnp.asarray(knots).shape[0] - 4
    result = jnp.zeros(nbasis, dtype=jnp.float64)
    result = jax.lax.dynamic_update_slice(result, basis, (start,))
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
        # Cached as Python floats: reading them per evaluation would force a
        # device sync on every call.
        self._x_min = float(np.asarray(self.xi[0]))
        self._x_max = float(np.asarray(self.xi[-1]))

    def EvaluateBsplines(self, x):
        if isinstance(x, jax_core.Tracer):
            return _evaluate_cubic_bspline_basis_jax(self.knots, x)

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 0:
            raise ValueError("Evaluation point x must be scalar.")

        x_val = float(x_arr)
        if x_val < self._x_min or x_val > self._x_max:
            raise ValueError(
                f"Error: Bspline_basis_1D(): x: {x_val} is outside of knots "
                f"vector with bounds [{self._x_min}, {self._x_max}]!"
            )

        return _evaluate_cubic_bspline_basis_jax(self.knots, x_val)

    def EvaluateBsplines3rdDerivatives(self, x):
        if isinstance(x, jax_core.Tracer):
            return _evaluate_cubic_bspline_3rd_derivatives_jax(self.knots, x)

        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.ndim != 0:
            raise ValueError("Evaluation point x must be scalar.")

        x_val = float(x_arr)
        if x_val < self._x_min or x_val > self._x_max:
            raise ValueError(
                f"Error: Bspline_basis_3rd_derivative_1D(): x: {x_val} is outside of knots "
                f"vector with bounds [{self._x_min}, {self._x_max}]!"
            )

        return _evaluate_cubic_bspline_3rd_derivatives_jax(self.knots, x_val)

    def AssembleSplineMatrix(self):
        """Assemble the cubic spline matrix with not-a-knot boundary conditions."""
        return _assemble_spline_matrix_jax(self.xi)


def construct_knots(nodes):
    """Construct the cubic B-spline knot vector matching GSL's convention."""
    nodes_array = _as_valid_nodes(nodes)
    return _construct_knots_jax(nodes_array)


@jax.jit
def _assemble_spline_matrix_jax(nodes):
    """Assemble the cubic spline matrix for a validated 1D node array.

    jit-compiled so repeated construction on grids with the same axis lengths
    reuses the compiled assembly instead of retracing it per instance.
    """
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


def _banded_spline_factors(nodes):
    """Factored collocation matrices for concrete node arrays, host-side."""
    return tuple(
        TPI_banded.factor_collocation_matrix(np.asarray(node, dtype=np.float64))
        for node in nodes
    )


def _banded_solve_host(factors, F_np, ngrid, values_ndim):
    """Pad the grid axes and run the banded solve along each grid axis."""
    coeffs = np.pad(F_np, [(1, 1)] * ngrid + [(0, 0)] * values_ndim, mode="constant")
    for axis in range(ngrid - 1, -1, -1):
        coeffs = TPI_banded.solve_collocation_axis(factors[axis], coeffs, axis)
    return coeffs


def _compute_spline_coefficients_nd(nodes, F, spline_matrix_factors=None, values_shape=()):
    """Compute tensor-product spline coefficients for validated nodes and data.

    The implementation performs sequential banded 1D solves along each axis
    (the collocation matrix has bandwidths (4, 4)), in O(n) memory and time
    per axis instead of the O(n^2)/O(n^3) of a dense factorization. The solve
    runs host-side via scipy; traced inputs (e.g. under jit) are routed
    through jax.pure_callback with identical numerics. The callback path is
    not differentiable w.r.t. F -- coefficient autodiff is out of scope.

    For vector/tensor-valued data, F carries trailing value axes of shape
    values_shape. The 1D solves only run over the grid axes; value axes ride
    along as extra right-hand sides.
    """
    nodes = tuple(jnp.asarray(node, dtype=jnp.float64) for node in nodes)
    F = jnp.asarray(F, dtype=jnp.float64)
    values_shape = tuple(values_shape)

    dims = tuple(int(node.shape[0]) for node in nodes)
    if F.shape != dims + values_shape:
        raise ValueError(f"Data on TP grid should have shape {list(dims + values_shape)}")

    ngrid = len(nodes)
    values_ndim = len(values_shape)

    if not any(isinstance(arr, jax_core.Tracer) for arr in nodes + (F,)):
        if spline_matrix_factors is None:
            spline_matrix_factors = _banded_spline_factors(nodes)
        coeffs = _banded_solve_host(spline_matrix_factors, np.asarray(F), ngrid, values_ndim)
        return jnp.asarray(coeffs)

    factors = spline_matrix_factors

    def _callback(*args):
        nodes_np = [np.asarray(node, dtype=np.float64) for node in args[:-1]]
        F_np = np.asarray(args[-1], dtype=np.float64)
        cb_factors = factors if factors is not None else _banded_spline_factors(nodes_np)
        return _banded_solve_host(cb_factors, F_np, ngrid, values_ndim)

    out_shape = tuple(dim + 2 for dim in dims) + values_shape
    return jax.pure_callback(
        _callback,
        jax.ShapeDtypeStruct(out_shape, jnp.float64),
        *nodes,
        F,
        vmap_method="sequential",
    )


def compute_spline_coefficients_nd(nodes, F):
    return _compute_spline_coefficients_nd(nodes, F)


def _contract_tensor_product_jax(bases, coeff_block):
    result = coeff_block
    for basis in reversed(bases):
        result = jnp.tensordot(result, basis, axes=((-1,), (0,)))
    return result


def _contract_tensor_product_values_jax(bases, coeff_block, values_shape):
    """Contract the grid axes of a (4,)*n + values_shape block, keeping value axes."""
    n = len(bases)
    result = coeff_block.reshape((4,) * n + (-1,))
    for basis in reversed(bases):
        result = jnp.sum(result * basis[:, None], axis=-2)
    return result.reshape(values_shape)


class TP_Interpolant_ND:
    """JAX-side tensor-product spline interpolant helper."""

    def __init__(self, nodes, coeffs=None, F=None):
        if nodes is None:
            raise ValueError("Missing input nodes.")
        if not isinstance(nodes, (list, tuple)):
            raise TypeError("Expected list of numpy.ndarrays.")
        if not np.array(list(map(lambda x: isinstance(x, np.ndarray), nodes))).all():
            raise TypeError("Expected list of numpy.ndarrays.")

        # Scalar-valued by default; TP_Interpolant_ND_Vector sets _values_shape
        # before delegating here.
        if not hasattr(self, "_values_shape"):
            self._values_shape = ()

        self.nodes = tuple(_as_valid_nodes(node) for node in nodes)
        self.n = len(self.nodes)
        self.c = None
        self.knots_list = None
        self.bases = None
        self.spline_matrix_factors = None

        self.TPInterpolationSetupND()
        if coeffs is not None:
            self.SetSplineCoefficientsND(coeffs)
        if F is not None:
            self.ComputeSplineCoefficientsND(F)

    def TPInterpolationSetupND(self):
        self.bases = tuple(BsplineBasis1D(np.asarray(node)) for node in self.nodes)
        self.knots_list = tuple(base.knots for base in self.bases)
        self.spline_matrix_factors = _banded_spline_factors(self.nodes)
        # Domain bounds cached host-side: reading node endpoints per evaluation
        # would force a device sync on every call.
        self._lows = np.array([base._x_min for base in self.bases], dtype=np.float64)
        self._highs = np.array([base._x_max for base in self.bases], dtype=np.float64)
        self._jit_eval = self._build_evaluator()
        # Batched evaluator, jit-cached once here: vmapping on every call would
        # retrace the kernel per batch (hundreds of ms). Coefficients are an
        # explicit argument so SetSplineCoefficientsND updates take effect
        # without recompilation.
        self._jit_eval_batched = jax.jit(jax.vmap(self._eval_pointwise, in_axes=(None, 0)))

    def _build_evaluator(self):
        n = self.n
        values_shape = self._values_shape
        inner_pad, knots_pad, span_max = _pack_axis_eval_arrays(self.knots_list)
        offsets = jnp.arange(-2, 4, dtype=jnp.int64)

        def _evaluate(c, X):
            # All axes are processed as one batched op chain. Keeping the traced
            # graph a single dependency chain matters for single-point latency:
            # independent per-axis subgraphs get scheduled across threads by the
            # XLA CPU runtime, and the cross-thread synchronization dominates the
            # actual arithmetic by an order of magnitude.
            X = jnp.asarray(X, dtype=jnp.float64)

            # Vectorized span search over all axes; the count of inner knots <= x
            # equals searchsorted(knots[3:-3], x, side="right") per axis.
            count = jnp.sum(inner_pad <= X[:, None], axis=1)
            span = jnp.clip(count + 2, 3, span_max)

            # The 6 knots around each axis span, then the cubic de Boor recurrence
            # unrolled on length-n vectors. 0/0 is guarded to 0 exactly as in
            # _find_active_cubic_bspline_span_jax.
            k = jnp.take_along_axis(knots_pad, span[:, None] + offsets[None, :], axis=1)
            l1 = X - k[:, 2]
            l2 = X - k[:, 1]
            l3 = X - k[:, 0]
            r1 = k[:, 3] - X
            r2 = k[:, 4] - X
            r3 = k[:, 5] - X

            def _safe_div(num, den):
                return jnp.where(den != 0.0, num / den, 0.0)

            b0 = _safe_div(r1, r1 + l1)
            b1 = _safe_div(l1, r1 + l1)
            q0 = r1 * _safe_div(b0, r1 + l2)
            q1 = l2 * _safe_div(b0, r1 + l2) + r2 * _safe_div(b1, r2 + l1)
            q2 = l1 * _safe_div(b1, r2 + l1)
            d0 = r1 * _safe_div(q0, r1 + l3)
            d1 = l3 * _safe_div(q0, r1 + l3) + r2 * _safe_div(q1, r2 + l2)
            d2 = l2 * _safe_div(q1, r2 + l2) + r3 * _safe_div(q2, r3 + l1)
            d3 = l1 * _safe_div(q2, r3 + l1)
            basis = jnp.stack((d0, d1, d2, d3), axis=1)

            starts = span - 3
            if values_shape:
                zero = jnp.zeros((), dtype=starts.dtype)
                coeff_block = jax.lax.dynamic_slice(
                    c,
                    tuple(starts[axis] for axis in range(n)) + (zero,) * len(values_shape),
                    (4,) * n + values_shape,
                )
                result = coeff_block.reshape((4,) * n + (-1,))
                for axis in range(n - 1, -1, -1):
                    result = jnp.sum(result * basis[axis][:, None], axis=-2)
                return result.reshape(values_shape)
            coeff_block = jax.lax.dynamic_slice(
                c, tuple(starts[axis] for axis in range(n)), (4,) * n
            )
            result = coeff_block
            for axis in range(n - 1, -1, -1):
                result = jnp.sum(result * basis[axis], axis=-1)
            return result

        return jax.jit(_evaluate)

    def ComputeSplineCoefficientsND(self, F):
        coeffs = _compute_spline_coefficients_nd(
            self.nodes, F, self.spline_matrix_factors, values_shape=self._values_shape
        )
        self.c = coeffs
        return coeffs

    def GetSplineCoefficientsND(self):
        return self.c

    def SetSplineCoefficientsND(self, coeffs):
        dims = tuple(len(node) + 2 for node in self.nodes) + self._values_shape
        coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
        if coeffs.shape != dims:
            raise ValueError(f"Spline coefficients should have shape {list(dims)}")
        self.c = coeffs

    def TPInterpolationND(self, X):
        if self.c is None:
            raise ValueError("Spline coefficients have not been set.")

        if isinstance(X, jax_core.Tracer):
            X_arr = jnp.asarray(X, dtype=jnp.float64)
            return self._jit_eval(self.c, X_arr)

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 1:
            raise ValueError("Evaluation point X is more than one-dimensional!")
        if X_arr.shape[0] != self.n:
            raise ValueError(
                f"Expected X to be array of length {self.n}, but got length {X_arr.shape[0]}"
            )

        if np.any(X_arr < self._lows) or np.any(X_arr > self._highs):
            for axis in range(self.n):
                if X_arr[axis] < self._lows[axis] or X_arr[axis] > self._highs[axis]:
                    raise ValueError(
                        f"TP_Interpolation_ND: X[{axis}] = {X_arr[axis]} is outside of "
                        f"knots vector [{self._lows[axis]}, {self._highs[axis]}]!"
                    )

        return self._jit_eval(self.c, X_arr)

    def _eval_pointwise(self, c, X):
        """Per-axis evaluation kernel with coefficients as an explicit argument."""
        bases = []
        starts = []
        for axis in range(self.n):
            basis, start = _find_active_cubic_bspline_span_jax(self.knots_list[axis], X[axis])
            bases.append(basis)
            starts.append(start)

        if self._values_shape:
            zero = jnp.zeros((), dtype=starts[0].dtype)
            coeff_block = jax.lax.dynamic_slice(
                c,
                tuple(starts) + (zero,) * len(self._values_shape),
                (4,) * self.n + self._values_shape,
            )
            return _contract_tensor_product_values_jax(bases, coeff_block, self._values_shape)
        coeff_block = jax.lax.dynamic_slice(c, tuple(starts), (4,) * self.n)
        return _contract_tensor_product_jax(bases, coeff_block)

    def _TPInterpolationND_jax(self, X):
        return self._eval_pointwise(self.c, X)

    def TPInterpolationND_batched(self, X):
        if self.c is None:
            raise ValueError("Spline coefficients have not been set.")

        if isinstance(X, jax_core.Tracer):
            X_arr = jnp.asarray(X, dtype=jnp.float64)
            return self._TPInterpolationND_batched_jax(X_arr)

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError("Evaluation batch X must be two-dimensional!")
        if X_arr.shape[1] != self.n:
            raise ValueError(
                f"Expected X to have shape (M, {self.n}), but got shape {X_arr.shape}"
            )

        if np.any(X_arr < self._lows) or np.any(X_arr > self._highs):
            for row_index, point in enumerate(X_arr):
                for axis in range(self.n):
                    if point[axis] < self._lows[axis] or point[axis] > self._highs[axis]:
                        raise ValueError(
                            f"TP_Interpolation_ND: X[{row_index}, {axis}] = {point[axis]} "
                            f"is outside of knots vector [{self._lows[axis]}, {self._highs[axis]}]!"
                        )

        return self._TPInterpolationND_batched_jax(jnp.asarray(X_arr, dtype=jnp.float64))

    def _TPInterpolationND_batched_jax(self, X):
        return self._jit_eval_batched(self.c, X)

    def __call__(self, X):
        X_arr = np.atleast_1d(np.asarray(X, dtype=np.float64))
        if X_arr.ndim == 2:
            return self.TPInterpolationND_batched(X_arr)
        if X_arr.ndim != 1:
            raise ValueError(
                "Evaluation point X must be one-dimensional (a single point) "
                "or two-dimensional (a batch of points)!"
            )
        if X_arr.shape[0] != self.n:
            raise ValueError(
                f"Expected X to be array of length {self.n}, but got length {X_arr.shape[0]}"
            )
        return self.TPInterpolationND(X_arr)


def _validate_spline1d_nodes(x):
    x_np = np.asarray(x, dtype=np.float64)
    if x_np.ndim != 1:
        raise ValueError("Input nodes must be one-dimensional.")
    if x_np.shape[0] < 4:
        raise ValueError("Require at least four input nodes for Spline1D.")
    if not np.isfinite(x_np).all():
        raise ValueError("Input nodes must be finite.")
    if (np.diff(x_np) <= 0.0).any():
        raise ValueError("Input nodes must be strictly increasing for Spline1D.")
    return x_np


def spline_1d_hermite(x, F):
    """Per-interval cubic coefficients of the 1D not-a-knot spline, pure JAX.

    Solves the classic tridiagonal not-a-knot system for the node derivatives
    with jax.lax.linalg.tridiagonal_solve, so construction is jit-compatible
    and runs entirely on the JAX device (GPU-capable, no host callback).
    Sortedness of x is exploited throughout: no search is performed anywhere,
    interval i is known by position.

    Returns (c0, c1, c2, c3), each of length n - 1: the spline on
    [x[i], x[i+1]] is c0[i] + t*(c1[i] + t*(c2[i] + t*c3[i])), t = xq - x[i].
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    F = jnp.asarray(F, dtype=jnp.float64)
    dx = jnp.diff(x)
    slope = jnp.diff(F) / dx

    # Not-a-knot boundary rows reduced to tridiagonal form (scipy CubicSpline
    # formulation); interior rows are the standard C^2 continuity conditions.
    d_left = x[2] - x[0]
    d_right = x[-1] - x[-3]
    dl = jnp.concatenate((jnp.zeros(1, x.dtype), dx[1:], d_right[None]))
    diag = jnp.concatenate((dx[1:2], 2.0 * (dx[:-1] + dx[1:]), dx[-2:-1]))
    du = jnp.concatenate((d_left[None], dx[:-1], jnp.zeros(1, x.dtype)))
    b_left = (
        (dx[0] + 2.0 * d_left) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]
    ) / d_left
    b_right = (
        dx[-1] ** 2 * slope[-2] + (2.0 * d_right + dx[-1]) * dx[-2] * slope[-1]
    ) / d_right
    b = jnp.concatenate(
        (b_left[None], 3.0 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:]), b_right[None])
    )

    s = jax.lax.linalg.tridiagonal_solve(dl, diag, du, b[:, None])[:, 0]

    c0 = F[:-1]
    c1 = s[:-1]
    c2 = (3.0 * slope - 2.0 * s[:-1] - s[1:]) / dx
    c3 = (s[:-1] + s[1:] - 2.0 * slope) / dx ** 2
    return c0, c1, c2, c3


def spline_1d_evaluate(x, poly, xq):
    """Evaluate per-interval cubic pieces at query points, pure JAX.

    Differentiable w.r.t. xq and vmap/jit-compatible. Accepts scalar or
    1D xq; the span search is a vectorized searchsorted on the node array.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    xq = jnp.asarray(xq, dtype=jnp.float64)
    c0, c1, c2, c3 = poly
    i = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, x.shape[0] - 2)
    t = xq - x[i]
    return c0[i] + t * (c1[i] + t * (c2[i] + t * c3[i]))


_spline_1d_hermite_jit = jax.jit(spline_1d_hermite)


class Spline1D:
    """Dedicated fast 1D cubic spline (not-a-knot) in native Hermite form.

    Construction solves the classic tridiagonal system for the node
    derivatives entirely on the JAX device (GPU-capable, no host callback)
    and stores per-interval polynomial coefficients; evaluation is a
    vectorized span search plus Horner evaluation, jit/vmap-compatible and
    differentiable w.r.t. the query points. Nodes must be strictly
    increasing with at least 4 points.

    Interop with the general TP_Interpolant_ND classes and previously saved
    data: to_coefficients() exports the standard TPI coefficient vector of
    shape (n + 2,), and the coeffs= constructor argument loads one (any
    B-form cubic spline on these nodes, not only not-a-knot interpolants).
    """

    def __init__(self, x, F=None, coeffs=None):
        x_np = _validate_spline1d_nodes(x)
        self.x = jnp.asarray(x_np)
        self.n = int(x_np.shape[0])
        # Host-side bounds: reading device arrays per call would force a sync.
        self._x_min = float(x_np[0])
        self._x_max = float(x_np[-1])
        self.poly = None
        self._jit_eval = jax.jit(
            lambda poly, xq: spline_1d_evaluate(self.x, poly, xq)
        )
        if F is not None and coeffs is not None:
            raise ValueError("Pass either F or coeffs, not both.")
        if F is not None:
            F_np = np.asarray(F, dtype=np.float64)
            if F_np.shape != (self.n,):
                raise ValueError(f"Data should have shape [{self.n}]")
            self.poly = _spline_1d_hermite_jit(self.x, jnp.asarray(F_np))
        elif coeffs is not None:
            coeffs_np = np.asarray(coeffs, dtype=np.float64)
            if coeffs_np.shape != (self.n + 2,):
                raise ValueError(
                    f"Spline coefficients should have shape [{self.n + 2}]"
                )
            f, s = TPI_banded.bspline_to_hermite(x_np, coeffs_np)
            pieces = TPI_banded.hermite_polynomial_pieces(x_np, f, s)
            self.poly = tuple(jnp.asarray(c) for c in pieces)

    def to_coefficients(self):
        """Standard TPI coefficient vector (shape (n + 2,)) of this spline."""
        if self.poly is None:
            raise ValueError("Spline coefficients have not been set.")
        c0, c1, c2, c3 = (np.asarray(c) for c in self.poly)
        return TPI_banded.hermite_to_bspline_coefficients(
            np.asarray(self.x), c0, c1, c2, c3
        )

    def __call__(self, xq):
        if self.poly is None:
            raise ValueError("Spline coefficients have not been set.")

        if isinstance(xq, jax_core.Tracer):
            return spline_1d_evaluate(self.x, self.poly, xq)

        xq_np = np.asarray(xq, dtype=np.float64)
        if xq_np.ndim > 1:
            raise ValueError("Query points must be scalar or one-dimensional!")
        if np.any(xq_np < self._x_min) or np.any(xq_np > self._x_max):
            bad = np.atleast_1d(xq_np)
            bad_value = bad[(bad < self._x_min) | (bad > self._x_max)][0]
            raise ValueError(
                f"Spline1D: x = {bad_value} is outside of knots vector "
                f"[{self._x_min}, {self._x_max}]!"
            )
        return self._jit_eval(self.poly, xq_np)


class TP_Interpolant_ND_Vector(TP_Interpolant_ND):
    """Tensor-product spline interpolant for vector/tensor-valued data.

    Behaves like TP_Interpolant_ND, except grid data F carries trailing value
    axes of shape values_shape and evaluation returns arrays of that shape.
    All components share the grid, knot vectors, and LU-factored spline
    matrices; the coefficient solve and the compiled evaluator are batched
    over the value axes, so interpolating an M-component field costs far less
    than M scalar interpolants.

    Shapes:
      * F passed to ComputeSplineCoefficientsND: grid dims + values_shape
      * coefficients: tuple(len(nodes_i) + 2) + values_shape
      * TPInterpolationND(X): values_shape
      * TPInterpolationND_batched(X) with X of shape (M, n): (M,) + values_shape
    """

    def __init__(self, nodes, values_shape, coeffs=None, F=None):
        try:
            shape_tuple = tuple(int(s) for s in np.atleast_1d(values_shape))
        except (TypeError, ValueError) as exc:
            raise ValueError("values_shape must be a shape of positive integers.") from exc
        if len(shape_tuple) == 0 or any(s < 1 for s in shape_tuple):
            raise ValueError("values_shape must be a shape of positive integers.")
        self._values_shape = shape_tuple
        super().__init__(nodes, coeffs=coeffs, F=F)

    @classmethod
    def FromComponentSplines(cls, nodes, components, values_shape=None):
        """Combine per-component splines sharing one grid into one vector-valued spline.

        Arguments:
          * nodes: list of 1D node arrays, as for the constructor. Every
            component must have been built on exactly these nodes.
          * components: list whose entries are either per-component spline
            coefficient arrays of shape tuple(len(nodes_i) + 2), or objects
            exposing GetSplineCoefficientsND() (TPI or TPI_jax interpolants).
          * values_shape: optional shape for the value axes; defaults to
            (len(components),). Its product must equal len(components); the
            components fill the value axes in C (row-major) order.

        Returns a TP_Interpolant_ND_Vector with the stacked coefficients set.
        """
        coeff_arrays = []
        for component in components:
            if hasattr(component, "GetSplineCoefficientsND"):
                component = component.GetSplineCoefficientsND()
            coeff_arrays.append(np.asarray(component, dtype=np.float64))
        if not coeff_arrays:
            raise ValueError("Expected at least one component spline.")

        expected = tuple(len(np.asarray(node)) + 2 for node in nodes)
        for index, coeff in enumerate(coeff_arrays):
            if coeff.shape != expected:
                raise ValueError(
                    f"Component {index} coefficients have shape {list(coeff.shape)}, "
                    f"expected {list(expected)}; all components must share the same nodes."
                )

        stacked = np.stack(coeff_arrays, axis=-1)
        if values_shape is None:
            values_shape = (len(coeff_arrays),)
        else:
            values_shape = tuple(int(s) for s in np.atleast_1d(values_shape))
            if int(np.prod(values_shape)) != len(coeff_arrays):
                raise ValueError(
                    f"values_shape {list(values_shape)} does not hold "
                    f"{len(coeff_arrays)} components."
                )
            stacked = stacked.reshape(expected + values_shape)
        return cls(nodes, values_shape, coeffs=stacked)
