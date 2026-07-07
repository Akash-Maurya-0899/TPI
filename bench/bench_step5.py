"""Step 5 benchmark: single-chain batched-axes evaluator vs HEAD kernel and GSL.

Run with:
    conda run -n test_esigmapy python bench/bench_step5.py

Loads the pre-step-5 TPI_jax from git HEAD for the before/after comparison, so run
this before committing step 5 (or change the git revision below).
"""

from __future__ import annotations

import importlib.util
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TPI
import TPI_jax

OLD_REVISION = "HEAD"

old_path = pathlib.Path("/tmp/TPI_jax_step5_old.py")
with old_path.open("wb") as fh:
    subprocess.run(
        ["git", "show", f"{OLD_REVISION}:TPI_jax.py"], check=True, stdout=fh, cwd=REPO_ROOT
    )

spec = importlib.util.spec_from_file_location("TPI_jax_old", old_path)
TPI_jax_old = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(TPI_jax_old)


def _block(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def time_once_ms(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    _block(result)
    return (time.perf_counter() - start) * 1e3, result


def median_ms(fn, repeat, *args):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args)
        _block(result)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), result


cases = [
    (
        "1D",
        (np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),),
        lambda x: np.cos(10.0 * x),
        np.array([0.16]),
    ),
    (
        "2D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]),
        ),
        lambda x, y: np.sin(x) * np.arccos(y),
        np.array([0.16, 0.28]),
    ),
    (
        "3D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]),
            np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0]),
        ),
        lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
        np.array([0.1692602, 0.2827312351474, -0.26624193]),
    ),
    (
        "4D",
        (
            np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25]),
            np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]),
            np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0]),
            np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5]),
        ),
        lambda x1, x2, x3, x4: np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4),
        np.array([0.16, 0.28, -0.26, 0.05]),
    ),
    (
        "7D",
        tuple(np.sort(np.random.default_rng(i).uniform(0.0, 1.0, 8)) for i in range(7)),
        None,
        None,
    ),
]


def make_batch(nodes, count=64):
    rng = np.random.default_rng(42)
    lows = np.array([node[0] for node in nodes], dtype=np.float64)
    highs = np.array([node[-1] for node in nodes], dtype=np.float64)
    return lows + rng.uniform(0.05, 0.95, size=(count, len(nodes))) * (highs - lows)


print("Step 5 benchmark (median of 30 single calls / 20 batch calls, 64-point batches)")
header = (
    "dim | single GSL (ms) | single JAX warmup (ms) | single JAX steady (ms) | "
    "single old-JAX steady (ms) | batch GSL (ms) | batch JAX warmup (ms) | "
    "batch JAX steady (ms) | batch old-JAX steady (ms) | max abs diff"
)
print(header)
for label, nodes, fn, point in cases:
    if fn is not None:
        mesh = np.meshgrid(*nodes, indexing="ij")
        F = np.asarray(fn(*mesh), dtype=np.float64)
    else:
        rng = np.random.default_rng(0)
        F = rng.standard_normal(tuple(len(node) for node in nodes))
        point = np.array([float(np.mean(node)) for node in nodes])
    batch = make_batch(nodes)

    gsl = TPI.TP_Interpolant_ND(list(nodes))
    gsl.ComputeSplineCoefficientsND(F)
    gsl_single_ms, gsl_value = median_ms(gsl.TPInterpolationND, 30, point)
    gsl_batch_ms, _ = median_ms(
        lambda pts: np.array([gsl.TPInterpolationND(p) for p in pts]), 20, batch
    )

    # Single-point latency is measured through jax.jit(TPInterpolationND) as in
    # the step 1-4 benchmarks, so kernel changes are compared on equal footing.
    new = TPI_jax.TP_Interpolant_ND(list(nodes))
    new.ComputeSplineCoefficientsND(F)
    new_single_jit = jax.jit(new.TPInterpolationND)
    new_warm_ms, new_value = time_once_ms(new_single_jit, point)
    new_single_ms, _ = median_ms(new_single_jit, 30, point)
    new_batch_jit = jax.jit(new.TPInterpolationND_batched)
    new_batch_warm_ms, _ = time_once_ms(new_batch_jit, batch)
    new_batch_ms, new_batch_values = median_ms(new_batch_jit, 20, batch)

    old = TPI_jax_old.TP_Interpolant_ND(list(nodes))
    old.ComputeSplineCoefficientsND(F)
    old_single_jit = jax.jit(old.TPInterpolationND)
    old_single_jit(point)
    old_single_ms, old_value = median_ms(old_single_jit, 30, point)
    old_batch_jit = jax.jit(old.TPInterpolationND_batched)
    old_batch_jit(batch)
    old_batch_ms, old_batch_values = median_ms(old_batch_jit, 20, batch)

    max_diff = max(
        abs(float(np.asarray(new_value)) - float(np.asarray(gsl_value))),
        float(np.max(np.abs(np.asarray(new_batch_values) - np.asarray(old_batch_values)))),
    )
    print(
        f"{label:>3} | {gsl_single_ms:15.6f} | {new_warm_ms:22.3f} | {new_single_ms:22.6f} | "
        f"{old_single_ms:26.6f} | {gsl_batch_ms:14.6f} | {new_batch_warm_ms:21.3f} | "
        f"{new_batch_ms:21.6f} | {old_batch_ms:25.6f} | {max_diff:.3e}"
    )
