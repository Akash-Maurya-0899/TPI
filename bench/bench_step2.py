from __future__ import annotations

import pathlib
import statistics
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_ROOT = pathlib.Path("/home/akash-x16/src/TPI")

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TPI
import TPI_jax


def _block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def _time_once(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    _block_until_ready(result)
    return (time.perf_counter() - start) * 1e3, result


def _time_median_ms(fn, repeat, *args):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args)
        _block_until_ready(result)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), result


def _make_values(nodes, fn):
    mesh = np.meshgrid(*nodes, indexing="ij")
    return np.asarray(fn(*mesh), dtype=np.float64)


def _make_batch_points(nodes, count=64):
    rng = np.random.default_rng(42)
    lows = np.array([node[0] for node in nodes], dtype=np.float64)
    highs = np.array([node[-1] for node in nodes], dtype=np.float64)
    return rng.uniform(lows, highs, size=(count, len(nodes))).astype(np.float64)


def _gsl_batch_eval(interp, batch_points):
    return np.asarray([interp.TPInterpolationND(point) for point in batch_points], dtype=np.float64)


CASES = [
    (
        "1D",
        (np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64),),
        lambda x: np.cos(10.0 * x),
        np.array([0.16], dtype=np.float64),
    ),
    (
        "2D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64),
        ),
        lambda x, y: np.sin(x) * np.arccos(y),
        np.array([0.16, 0.28], dtype=np.float64),
    ),
    (
        "3D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64),
            np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0], dtype=np.float64),
        ),
        lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
        np.array([0.1692602, 0.2827312351474, -0.26624193], dtype=np.float64),
    ),
    (
        "4D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25], dtype=np.float64),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0], dtype=np.float64),
            np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0], dtype=np.float64),
            np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5], dtype=np.float64),
        ),
        lambda x1, x2, x3, x4: np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4),
        np.array([0.16, 0.28, -0.26, 0.05], dtype=np.float64),
    ),
]


def _print_row(dim, single_gsl, single_warmup, single_steady, batch_gsl, batch_warmup, batch_steady):
    print(
        f"{dim:>3} | {single_gsl:16.3f} | {single_warmup:20.3f} | {single_steady:19.3f} | "
        f"{batch_gsl:14.3f} | {batch_warmup:19.3f} | {batch_steady:18.3f}"
    )


def main():
    jax.print_environment_info()

    print()
    print("Canonical Step 2 latency table")
    print(
        "dim | single GSL (ms) | single JAX warmup (ms) | single JAX steady (ms) "
        "| batch GSL (ms)  | batch JAX warmup (ms)  | batch JAX steady (ms)"
    )

    for dim, nodes, fn, single_point in CASES:
        values = _make_values(nodes, fn)
        batch_points = _make_batch_points(nodes)

        gsl = TPI.TP_Interpolant_ND(list(nodes))
        gsl.TPInterpolationSetupND()
        gsl.ComputeSplineCoefficientsND(values)
        single_gsl_ms, _ = _time_median_ms(gsl.TPInterpolationND, 20, single_point)
        batch_gsl_ms, _ = _time_median_ms(_gsl_batch_eval, 20, gsl, batch_points)

        jax_interp = TPI_jax.TP_Interpolant_ND(list(nodes))
        jax_interp.TPInterpolationSetupND()
        jax_interp.ComputeSplineCoefficientsND(values)

        single_jit = jax.jit(jax_interp.TPInterpolationND)
        single_warmup_ms, _ = _time_once(single_jit, single_point)
        single_steady_ms, _ = _time_median_ms(single_jit, 20, single_point)

        batch_jit = jax.jit(jax_interp.TPInterpolationND_batched)
        batch_warmup_ms, _ = _time_once(batch_jit, batch_points)
        batch_steady_ms, _ = _time_median_ms(batch_jit, 20, batch_points)

        _print_row(
            dim,
            single_gsl_ms,
            single_warmup_ms,
            single_steady_ms,
            batch_gsl_ms,
            batch_warmup_ms,
            batch_steady_ms,
        )


if __name__ == "__main__":
    main()
