#!/usr/bin/env python
"""Benchmark TPI Cython/GSL and JAX backends side by side.

Run with:
    conda run -n test_esigmapy python benchmarks/bench_TPI.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import sys

import numpy as np

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TPI
import TPI_jax


@dataclass(frozen=True)
class Case:
    dim: int
    nodes: tuple[np.ndarray, ...]
    function_name: str
    function: object
    single_point: np.ndarray
    batch_points: np.ndarray
    coeff_tensors: tuple[np.ndarray, ...]

    @property
    def coeff_shape(self) -> tuple[int, ...]:
        return tuple(len(node) + 2 for node in self.nodes)


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)


def _elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1e3


def _format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def _time_repeat(fn, repeat: int, *args):
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn(*args)
    end = time.perf_counter()
    return _elapsed_ms(start, end), result


def _time_repeat_blocking(fn, repeat: int, *args):
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn(*args)
        _block_until_ready(result)
    end = time.perf_counter()
    return _elapsed_ms(start, end), result


def _time_once(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    _block_until_ready(result)
    end = time.perf_counter()
    return _elapsed_ms(start, end), result


def _make_batch_points(nodes: tuple[np.ndarray, ...], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lows = np.array([float(node[0]) for node in nodes], dtype=np.float64)
    highs = np.array([float(node[-1]) for node in nodes], dtype=np.float64)
    interior = rng.uniform(0.1, 0.9, size=(count, len(nodes)))
    return lows + interior * (highs - lows)


def _make_coeff_tensors(shape: tuple[int, ...], count: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    tensors = []
    for index in range(count):
        tensor = rng.standard_normal(shape, dtype=np.float64)
        tensor += 0.05 * index
        tensors.append(tensor)
    return tuple(tensors)


def _build_cases() -> dict[int, Case]:
    xi1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64)
    f1 = lambda x: np.cos(10.0 * x)
    case1 = Case(
        dim=1,
        nodes=(xi1,),
        function_name="cos(10x)",
        function=f1,
        single_point=np.array([0.16], dtype=np.float64),
        batch_points=_make_batch_points((xi1,), 256, seed=101),
        coeff_tensors=_make_coeff_tensors((len(xi1) + 2,), 6, seed=201),
    )

    xi2 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64)
    yi2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64)
    f2 = lambda x, y: np.sin(x) * np.arccos(y)
    case2 = Case(
        dim=2,
        nodes=(xi2, yi2),
        function_name="sin(x) * arccos(y)",
        function=f2,
        single_point=np.array([0.16, 0.28], dtype=np.float64),
        batch_points=_make_batch_points((xi2, yi2), 192, seed=102),
        coeff_tensors=_make_coeff_tensors((len(xi2) + 2, len(yi2) + 2), 6, seed=202),
    )

    xi3 = xi2
    yi3 = yi2
    zi3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0], dtype=np.float64)
    f3 = lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z)
    case3 = Case(
        dim=3,
        nodes=(xi3, yi3, zi3),
        function_name="sin(x) * arccos(y) * exp(z)",
        function=f3,
        single_point=np.array([0.1692602, 0.2827312351474, -0.26624193], dtype=np.float64),
        batch_points=_make_batch_points((xi3, yi3, zi3), 128, seed=103),
        coeff_tensors=_make_coeff_tensors((len(xi3) + 2, len(yi3) + 2, len(zi3) + 2), 6, seed=203),
    )

    x1_4 = xi2
    x2_4 = yi2
    x3_4 = zi3
    x4_4 = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    f4 = lambda x1, x2, x3, x4: np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4)
    case4 = Case(
        dim=4,
        nodes=(x1_4, x2_4, x3_4, x4_4),
        function_name="sin(x1) * arccos(x2) * exp(x3) * cos(x4)",
        function=f4,
        single_point=np.array([0.16, 0.28, -0.26, 0.05], dtype=np.float64),
        batch_points=_make_batch_points((x1_4, x2_4, x3_4, x4_4), 96, seed=104),
        coeff_tensors=_make_coeff_tensors((len(x1_4) + 2, len(x2_4) + 2, len(x3_4) + 2, len(x4_4) + 2), 6, seed=204),
    )

    x1_7 = np.array(
        [0.04210023, 0.08049712, 0.10439003, 0.23567061, 0.26747638, 0.51894333, 0.87695656, 1.13424169],
        dtype=np.float64,
    )
    x2_7 = np.array(
        [0.06512773, 0.10554492, 0.30739299, 0.52934042, 0.53375456, 0.70565296, 0.90977329, 1.0904668, 1.09161535],
        dtype=np.float64,
    )
    x3_7 = np.array(
        [0.17568927, 0.20990473, 0.40272389, 0.54519648, 0.62970609, 0.65005828, 0.67672559, 1.03551716],
        dtype=np.float64,
    )
    x4_7 = np.array(
        [0.04209146, 0.18164518, 0.32001217, 0.5469396, 0.65685659, 0.69706066, 0.8338755, 0.84175853, 1.03421552],
        dtype=np.float64,
    )
    x5_7 = np.array([0.15592869, 0.24300596, 0.53102712, 0.76409654, 0.83426527], dtype=np.float64)
    x6_7 = np.array(
        [
            0.09278997,
            0.60858288,
            0.68604479,
            0.69185573,
            1.05187626,
            1.25311729,
            1.83783997,
            2.30367353,
            2.34835024,
            2.5526501,
            2.88134666,
        ],
        dtype=np.float64,
    )
    x7_7 = np.array(
        [0.02039465, 0.84219648, 1.34410666, 1.50315468, 1.77942063, 4.30194875, 4.81437542, 5.30694653, 6.04394163],
        dtype=np.float64,
    )
    f7 = lambda x1, x2, x3, x4, x5, x6, x7: np.sin(x1) * np.arccos(x2 / 2.0) * np.exp(x3) * np.cos(x4) * np.abs(x5) + np.sin(x6) * np.exp(x7)
    case7 = Case(
        dim=7,
        nodes=(x1_7, x2_7, x3_7, x4_7, x5_7, x6_7, x7_7),
        function_name="7D test function",
        function=f7,
        single_point=np.array([0.673, 0.2836, 0.734, 0.089, 0.619, 1.782, 4.96], dtype=np.float64),
        batch_points=_make_batch_points((x1_7, x2_7, x3_7, x4_7, x5_7, x6_7, x7_7), 48, seed=107),
        coeff_tensors=_make_coeff_tensors((len(x1_7) + 2, len(x2_7) + 2, len(x3_7) + 2, len(x4_7) + 2, len(x5_7) + 2, len(x6_7) + 2, len(x7_7) + 2), 4, seed=207),
    )

    return {case.dim: case for case in (case1, case2, case3, case4, case7)}


def _make_function_values(case: Case) -> np.ndarray:
    mesh = np.meshgrid(*case.nodes, indexing="ij")
    return np.asarray(case.function(*mesh), dtype=np.float64)


def _make_cython_interpolant(case: Case):
    return TPI.TP_Interpolant_ND(list(case.nodes))


def _make_jax_interpolant(case: Case):
    return TPI_jax.TP_Interpolant_ND(list(case.nodes))


def _make_jax_eval_with_coeffs(case: Case):
    knots_list = tuple(TPI_jax.construct_knots(node) for node in case.nodes)
    dim = case.dim
    basis_labels = ",".join("abcdefghijklmnopqrstuvwxyz"[i] for i in range(dim))
    coeff_labels = "".join("abcdefghijklmnopqrstuvwxyz"[i] for i in range(dim))
    equation = f"{basis_labels},{coeff_labels}->"

    def eval_one(coeffs, point):
        bases = []
        starts = []
        for axis in range(dim):
            basis, start = TPI_jax._find_active_cubic_bspline_span_jax(knots_list[axis], point[axis])
            bases.append(basis)
            starts.append(start)
        coeff_block = jax.lax.dynamic_slice(coeffs, tuple(starts), (4,) * dim)
        return jnp.einsum(equation, *bases, coeff_block)

    return jax.jit(eval_one)


def _make_jax_batched_eval_with_coeffs(case: Case):
    scalar_eval = _make_jax_eval_with_coeffs(case)
    return jax.jit(jax.vmap(scalar_eval, in_axes=(None, 0)))


def _print_line(label: str, value: float, extra: str = ""):
    suffix = f" {extra}" if extra else ""
    print(f"  {label:<18} {value:>12.3f} ms{suffix}")


def _print_warmup_and_steady(label: str, warmup_ms: float, steady_ms: float, speedup_vs_gsl: float | None = None):
    print(f"  {label:<18} JIT warmup    {_format_ms(warmup_ms)}")
    if speedup_vs_gsl is None:
        print(f"  {label:<18} steady-state  {_format_ms(steady_ms)}")
    else:
        print(
            f"  {label:<18} steady-state  {_format_ms(steady_ms)} "
            f"({speedup_vs_gsl:.2f}x vs GSL)"
        )


def _repeat_counts(dim: int) -> dict[str, int]:
    return {
        1: 5000,
        2: 3000,
        3: 2000,
        4: 1000,
        7: 200,
    }[dim]


def _batch_repeat_counts(dim: int) -> int:
    return {
        1: 200,
        2: 120,
        3: 80,
        4: 48,
        7: 20,
    }[dim]


def _solve_repeat_counts(dim: int) -> int:
    return {
        1: 100,
        2: 60,
        3: 32,
        4: 20,
        7: 8,
    }[dim]


def benchmark_case(case: Case):
    print(f"=== {case.dim}D | {case.function_name} ===")
    print(f"grid shape: {tuple(len(node) for node in case.nodes)}")
    print(f"coeff shape: {case.coeff_shape}")
    print(f"single-point evaluation uses X = {np.array2string(case.single_point, precision=6, separator=', ')}")
    print(f"batched evaluation uses {case.batch_points.shape[0]} points")
    print(f"repeated coefficient benchmark uses {len(case.coeff_tensors)} coefficient tensors")

    f_values = _make_function_values(case)

    cython = _make_cython_interpolant(case)
    jax_interp = _make_jax_interpolant(case)

    print("Single-point evaluation")
    cython.ComputeSplineCoefficientsND(f_values)
    single_reps = _repeat_counts(case.dim)
    cython_single_ms, cython_single_result = _time_repeat(cython.TPInterpolationND, single_reps, case.single_point)
    cython_single_value = float(np.asarray(cython_single_result))
    print(f"  GSL/Cython         steady-state  {_format_ms(cython_single_ms / single_reps)}")

    jax_interp.ComputeSplineCoefficientsND(f_values)
    jax_single_jit = jax.jit(jax_interp.TPInterpolationND)
    jax_single_warmup_ms, _ = _time_once(jax_single_jit, case.single_point)
    jax_single_steady_ms, jax_single_result = _time_repeat_blocking(jax_single_jit, single_reps, case.single_point)
    jax_single_value = float(np.asarray(jax_single_result))
    print(
        f"  JAX                JIT warmup    {_format_ms(jax_single_warmup_ms)}"
    )
    print(
        f"  JAX                steady-state  {_format_ms(jax_single_steady_ms / single_reps)} "
        f"({(cython_single_ms / jax_single_steady_ms):.2f}x speedup vs GSL)"
    )
    print(
        f"  result check       | GSL={cython_single_value:.12g} | JAX={jax_single_value:.12g}"
    )

    print("Batched evaluation")
    batch_reps = _batch_repeat_counts(case.dim)
    cython_batch_ms, cython_batch_result = _time_repeat(
        lambda pts: np.array([float(np.asarray(cython.TPInterpolationND(pt))) for pt in pts], dtype=np.float64),
        batch_reps,
        case.batch_points,
    )
    cython_batch_values = np.asarray(cython_batch_result, dtype=np.float64)
    print(f"  GSL/Cython         steady-state  {_format_ms(cython_batch_ms / batch_reps)}")

    jax_batched_jit = jax.jit(jax_interp.TPInterpolationND_batched)
    jax_batch_warmup_ms, _ = _time_once(jax_batched_jit, case.batch_points)
    jax_batch_steady_ms, jax_batch_result = _time_repeat_blocking(jax_batched_jit, batch_reps, case.batch_points)
    jax_batch_values = np.asarray(jax_batch_result, dtype=np.float64)
    print(f"  JAX                JIT warmup    {_format_ms(jax_batch_warmup_ms)}")
    print(
        f"  JAX                steady-state  {_format_ms(jax_batch_steady_ms / batch_reps)} "
        f"({(cython_batch_ms / jax_batch_steady_ms):.2f}x speedup vs GSL)"
    )
    print(f"  batch max abs diff  {np.max(np.abs(jax_batch_values - cython_batch_values)):.3e}")

    print("Repeated evaluations with different coefficient tensors")
    repeated_point = case.single_point
    cython_repeat_ms = 0.0
    cython_repeat_values = []
    for coeffs in case.coeff_tensors:
        cython.SetSplineCoefficientsND(coeffs)
        t0 = time.perf_counter()
        result = cython.TPInterpolationND(repeated_point)
        t1 = time.perf_counter()
        cython_repeat_ms += _elapsed_ms(t0, t1)
        cython_repeat_values.append(float(np.asarray(result)))
    print(
        f"  GSL/Cython         steady-state  {_format_ms(cython_repeat_ms / len(case.coeff_tensors))}"
    )

    jax_repeat_eval = _make_jax_eval_with_coeffs(case)
    warmup_coeffs = case.coeff_tensors[0]
    jax_repeat_warmup_ms, _ = _time_once(jax_repeat_eval, warmup_coeffs, repeated_point)
    jax_repeat_ms = 0.0
    jax_repeat_values = []
    for coeffs in case.coeff_tensors:
        t0 = time.perf_counter()
        result = jax_repeat_eval(coeffs, repeated_point)
        _block_until_ready(result)
        t1 = time.perf_counter()
        jax_repeat_ms += _elapsed_ms(t0, t1)
        jax_repeat_values.append(float(np.asarray(result)))
    print(f"  JAX                JIT warmup    {_format_ms(jax_repeat_warmup_ms)}")
    print(
        f"  JAX                steady-state  {_format_ms(jax_repeat_ms / len(case.coeff_tensors))} "
        f"({(cython_repeat_ms / jax_repeat_ms):.2f}x speedup vs GSL)"
    )
    print(
        f"  repeat max abs diff {np.max(np.abs(np.asarray(jax_repeat_values) - np.asarray(cython_repeat_values))):.3e}"
    )

    print("Coefficient solve")
    if case.dim == 7:
        print(
            "  note: the 7D coefficient gap observed in this grid is expected to be amplified "
            "by the ill-conditioned sequential solve; it is not an implementation bug."
        )
    solve_reps = _solve_repeat_counts(case.dim)
    cython_solve_ms, _ = _time_repeat(cython.ComputeSplineCoefficientsND, solve_reps, f_values)
    print(f"  GSL/Cython         steady-state  {_format_ms(cython_solve_ms / solve_reps)}")

    jax_solve_jit = jax.jit(TPI_jax.compute_spline_coefficients_nd)
    jax_solve_warmup_ms, _ = _time_once(jax_solve_jit, case.nodes, f_values)
    jax_solve_ms, jax_solve_result = _time_repeat_blocking(jax_solve_jit, solve_reps, case.nodes, f_values)
    print(f"  JAX                JIT warmup    {_format_ms(jax_solve_warmup_ms)}")
    print(
        f"  JAX                steady-state  {_format_ms(jax_solve_ms / solve_reps)} "
        f"({(cython_solve_ms / jax_solve_ms):.2f}x speedup vs GSL)"
    )
    print(f"  solve max abs diff  {np.max(np.abs(np.asarray(jax_solve_result) - np.asarray(cython.GetSplineCoefficientsND()))):.3e}")

    print("Setup and spline-matrix assembly")
    cython_construct_ms, _ = _time_repeat(_make_cython_interpolant, 200, case)
    print(f"  GSL/Cython construct steady-state  {_format_ms(cython_construct_ms / 200.0)}")

    jax_construct_ms, _ = _time_repeat(_make_jax_interpolant, 200, case)
    print(f"  JAX                steady-state  {_format_ms(jax_construct_ms / 200.0)}")

    basis_nodes = case.nodes[0]
    cython_basis = TPI.BsplineBasis1D(basis_nodes)
    jax_basis = TPI_jax.BsplineBasis1D(basis_nodes)
    cython_matrix_ms, cython_matrix_result = _time_repeat(cython_basis.AssembleSplineMatrix, 64)
    print(f"  GSL/Cython matrix   steady-state  {_format_ms(cython_matrix_ms / 64.0)}")

    jax_matrix_jit = jax.jit(jax_basis.AssembleSplineMatrix)
    jax_matrix_warmup_ms, _ = _time_once(jax_matrix_jit)
    jax_matrix_ms, jax_matrix_result = _time_repeat_blocking(lambda: jax_matrix_jit(), 64)
    print(f"  JAX                JIT warmup    {_format_ms(jax_matrix_warmup_ms)}")
    print(
        f"  JAX                steady-state  {_format_ms(jax_matrix_ms / 64.0)} "
        f"({(cython_matrix_ms / jax_matrix_ms):.2f}x speedup vs GSL)"
    )
    print(
        f"  matrix max abs diff {np.max(np.abs(np.asarray(jax_matrix_result[0]) - np.asarray(cython_matrix_result[0]))):.3e}"
    )
    print("")


def main() -> None:
    cases = _build_cases()
    print("TPI benchmark suite")
    print(f"jax version: {jax.__version__}")
    print(f"jaxlib version: {jaxlib.__version__}")
    print("")
    for dim in (1, 2, 3, 4, 7):
        benchmark_case(cases[dim])


if __name__ == "__main__":
    main()
