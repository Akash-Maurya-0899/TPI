#!/usr/bin/env python
"""Benchmark the dedicated 1D fast path (Spline1D) on large grids.

Construction, fixed-grid refit, and sorted-batch evaluation timings for
both backends, with scipy.interpolate.CubicSpline as the not-a-knot
reference and, when installed, pygsl_lite's cspline as an external GSL
reference (natural boundary conditions -- different math, reference only).

Run with:
    conda run -n test_esigmapy python benchmarks/bench_spline1d.py
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

try:
    from pygsl_lite import spline as pygsl_spline
except ImportError:
    pygsl_spline = None

N_NODES = 500_000
N_QUERIES = 2_000_000


def _median_ms(fn, repeat=20):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _median_ms_blocking(fn, repeat=20):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _grids():
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0.0, 100.0, N_NODES))
    x[0] = 0.0
    x[-1] = 100.0
    x = np.unique(x)
    F = np.sin(x) * np.exp(-0.01 * x)
    xq = np.sort(rng.uniform(0.0, 100.0, N_QUERIES))
    return x, F, xq


def _build_pygsl(x, F):
    s = pygsl_spline.cspline(len(x))
    s.init(x, F)
    return s


def bench_construction(x, F):
    print(f"Construction, n = {len(x)} (median of 20, steady-state):")

    t_gsl = _median_ms(lambda: TPI.Spline1D(x, F=F))
    print(f"  TPI.Spline1D (C, not-a-knot):          {t_gsl:8.2f} ms")

    warm = TPI.Spline1D(x, F=F)
    t_refit = _median_ms(lambda: warm.ComputeSplineCoefficients(F))
    print(f"  TPI.Spline1D refit (cached LU):        {t_refit:8.2f} ms")

    t_jax_warmup = _median_ms_blocking(lambda: TPI_jax.Spline1D(x, F=F), repeat=1)
    t_jax = _median_ms_blocking(lambda: TPI_jax.Spline1D(x, F=F))
    print(f"  TPI_jax.Spline1D ({jax.default_backend()}):{'':15s}{t_jax:8.2f} ms"
          f"   (first call incl. jit: {t_jax_warmup:.2f} ms)")

    from scipy.interpolate import CubicSpline

    t_scipy = _median_ms(lambda: CubicSpline(x, F, bc_type="not-a-knot"))
    print(f"  scipy CubicSpline (not-a-knot):        {t_scipy:8.2f} ms")

    if pygsl_spline is not None:
        t_pygsl = _median_ms(lambda: _build_pygsl(x, F))
        print(f"  pygsl_lite cspline (natural BC,")
        print(f"    different math, reference only):    {t_pygsl:8.2f} ms")
    else:
        print("  pygsl_lite not available - skipped")


def bench_evaluation(x, F, xq):
    from scipy.interpolate import CubicSpline

    print()
    print(f"Sorted-batch evaluation, M = {len(xq)} (median of 20, steady-state):")

    gsl = TPI.Spline1D(x, F=F)
    t_gsl = _median_ms(lambda: gsl(xq))
    print(f"  TPI.Spline1D (C monotone walk):        {t_gsl:8.2f} ms")

    jax_spline = TPI_jax.Spline1D(x, F=F)
    jax_spline(xq)  # warmup: trigger JIT compilation before timing
    t_jax = _median_ms_blocking(lambda: jax_spline(xq))
    print(f"  TPI_jax.Spline1D ({jax.default_backend()}):{'':15s}{t_jax:8.2f} ms")

    reference = CubicSpline(x, F, bc_type="not-a-knot")
    t_scipy = _median_ms(lambda: reference(xq))
    print(f"  scipy CubicSpline (PPoly):             {t_scipy:8.2f} ms")

    if pygsl_spline is not None:
        pygsl = _build_pygsl(x, F)
        t_pygsl = _median_ms(lambda: pygsl.eval_vector(xq))
        print(f"  pygsl_lite eval_vector (natural BC):   {t_pygsl:8.2f} ms")
    else:
        print("  pygsl_lite not available - skipped")

    err_gsl = np.max(np.abs(np.asarray(gsl(xq)) - reference(xq)))
    err_jax = np.max(np.abs(np.asarray(jax_spline(xq)) - reference(xq)))
    print()
    print("Accuracy vs scipy CubicSpline (not-a-knot) on the full query batch:")
    print(f"  max |TPI - scipy|:     {err_gsl:.3e}")
    print(f"  max |TPI_jax - scipy|: {err_jax:.3e}")


def bench_vector(x, xq, p=3):
    """Vector-valued Spline1D (trailing value axes) vs p scalar splines."""
    from scipy.interpolate import CubicSpline

    Fv = np.stack(
        [np.sin((k + 1) * 0.05 * x) * np.exp(-0.01 * (k + 1) * x) for k in range(p)],
        axis=-1,
    )

    print()
    print(f"Vector-valued splines, n = {len(x)}, values_shape = ({p},) "
          "(median of 20, steady-state):")

    t_con = _median_ms(lambda: TPI.Spline1D(x, F=Fv))
    t_con_scalar = _median_ms(
        lambda: [TPI.Spline1D(x, F=Fv[:, k]) for k in range(p)])
    print(f"  construction  TPI.Spline1D vector:    {t_con:8.2f} ms"
          f"   ({p} scalar splines: {t_con_scalar:.2f} ms)")

    warm = TPI.Spline1D(x, F=Fv)
    t_refit = _median_ms(lambda: warm.ComputeSplineCoefficients(Fv))
    print(f"  refit         TPI.Spline1D vector:    {t_refit:8.2f} ms")

    t_ev = _median_ms(lambda: warm(xq))
    scalars = [TPI.Spline1D(x, F=Fv[:, k]) for k in range(p)]
    t_ev_scalar = _median_ms(lambda: [s(xq) for s in scalars])
    print(f"  eval M={len(xq)}  TPI.Spline1D vector:    {t_ev:8.2f} ms"
          f"   ({p} scalar splines: {t_ev_scalar:.2f} ms)")

    jax_vec = TPI_jax.Spline1D(x, F=Fv)
    jax_vec(xq)  # warmup: trigger JIT compilation before timing
    t_jax_con = _median_ms_blocking(lambda: TPI_jax.Spline1D(x, F=Fv).poly)
    t_jax_ev = _median_ms_blocking(lambda: jax_vec(xq))
    print(f"  construction  TPI_jax.Spline1D vector ({jax.default_backend()}): {t_jax_con:8.2f} ms")
    print(f"  eval          TPI_jax.Spline1D vector ({jax.default_backend()}): {t_jax_ev:8.2f} ms")

    reference = CubicSpline(x, Fv, bc_type="not-a-knot", axis=0)
    t_scipy_ev = _median_ms(lambda: reference(xq))
    print(f"  eval          scipy CubicSpline (PPoly, axis=0): {t_scipy_ev:6.2f} ms")

    err_gsl = np.max(np.abs(np.asarray(warm(xq)) - reference(xq)))
    err_jax = np.max(np.abs(np.asarray(jax_vec(xq)) - reference(xq)))
    print(f"  accuracy: max |TPI - scipy| = {err_gsl:.3e}, "
          f"max |TPI_jax - scipy| = {err_jax:.3e}")


def main():
    print("Environment:")
    jax.print_environment_info()
    print()
    x, F, xq = _grids()
    bench_construction(x, F)
    bench_evaluation(x, F, xq)
    bench_vector(x, xq)


if __name__ == "__main__":
    main()
