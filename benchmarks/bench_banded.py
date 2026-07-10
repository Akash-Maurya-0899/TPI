#!/usr/bin/env python
"""Benchmark the banded coefficient solve against the legacy dense solve.

Coefficient-computation timings for the standard 1D-4D grids (legacy dense
inverse replicated inline vs the shipped banded path, both backends) and
1D construction timings on large grids with scipy.interpolate.CubicSpline
as an external reference.

Run with:
    conda run -n test_esigmapy python benchmarks/bench_banded.py
"""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TPI
import TPI_jax


def _median_ms(fn, repeat=20):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _standard_grids():
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


def _legacy_dense_coefficients(nodes, F):
    """The pre-banded implementation: dense assembly, explicit inverse, tensordot."""
    d = len(nodes)
    inv_1d_matrices = []
    for i in range(d):
        A, _ = TPI.BsplineBasis1D(nodes[i]).AssembleSplineMatrix()
        inv_1d_matrices.append(np.linalg.inv(A))
    tmp = np.pad(F, 1, "constant")
    for minv in inv_1d_matrices[::-1]:
        tmp = np.tensordot(minv, tmp, (1, d - 1))
    return tmp


def bench_small_grids():
    print("Coefficient computation, standard grids (median of 20, steady-state):")
    print("dim | legacy dense (ms) | GSL banded (ms) | JAX banded (ms)")
    for dim, nodes, F in _standard_grids():
        gsl = TPI.TP_Interpolant_ND(nodes)
        jaxi = TPI_jax.TP_Interpolant_ND(nodes)
        # warmup: trigger any lazy setup before timing
        gsl.ComputeSplineCoefficientsND(F)
        jaxi.ComputeSplineCoefficientsND(F)
        _legacy_dense_coefficients(nodes, F)

        t_legacy = _median_ms(lambda: _legacy_dense_coefficients(nodes, F))
        t_gsl = _median_ms(lambda: gsl.ComputeSplineCoefficientsND(F))
        t_jax = _median_ms(lambda: jaxi.ComputeSplineCoefficientsND(F))
        print(f"{dim:3d} | {t_legacy:17.3f} | {t_gsl:15.3f} | {t_jax:15.3f}")


def bench_large_1d():
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(42)
    print()
    print("1D construction on large grids (median of 20, steady-state):")
    print("      n | GSL banded: setup+solve (ms) | JAX banded: setup+solve (ms) "
          "| scipy CubicSpline (ms)")
    for n in (100_000, 500_000):
        x = np.sort(rng.uniform(0.0, 100.0, n))
        x[0] = 0.0
        x[-1] = 100.0
        F = np.sin(x) * np.exp(-0.01 * x)

        t_gsl = _median_ms(lambda: TPI.TP_Interpolant_ND([x], F=F))
        t_jax = _median_ms(lambda: TPI_jax.TP_Interpolant_ND([x], F=F))
        t_scipy = _median_ms(lambda: CubicSpline(x, F, bc_type="not-a-knot"))
        print(f"{n:7d} | {t_gsl:28.1f} | {t_jax:28.1f} | {t_scipy:22.1f}")


def main():
    print("Environment:")
    jax.print_environment_info()
    print()
    bench_small_grids()
    bench_large_1d()


if __name__ == "__main__":
    main()
