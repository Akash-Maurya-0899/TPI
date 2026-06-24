from __future__ import annotations

import importlib.util
import pathlib
import statistics
import subprocess
import sys
import time

import jax
import numpy as np

REPO_ROOT = pathlib.Path("/home/akash-x16/src/TPI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import TPI
import TPI_jax

jax.print_environment_info()

old_path = pathlib.Path("/tmp/TPI_jax_old.py")
with old_path.open("wb") as fh:
    subprocess.run(["git", "show", "HEAD^:TPI_jax.py"], check=True, stdout=fh)

spec = importlib.util.spec_from_file_location("TPI_jax_old", old_path)
TPI_jax_old = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(TPI_jax_old)

cases = [
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


def make_values(nodes, fn):
    mesh = np.meshgrid(*nodes, indexing="ij")
    return np.asarray(fn(*mesh), dtype=np.float64)


def block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()


def time_once(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    block_until_ready(result)
    end = time.perf_counter()
    return (end - start) * 1e3, result


def median_ms(fn, repeat, *args):
    samples = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args)
        block_until_ready(result)
        end = time.perf_counter()
        samples.append((end - start) * 1e3)
    return statistics.median(samples), result


rows = []
for label, nodes, fn, point in cases:
    values = make_values(nodes, fn)

    gsl = TPI.TP_Interpolant_ND(list(nodes))
    gsl.ComputeSplineCoefficientsND(values)
    gsl_ms, _ = median_ms(gsl.TPInterpolationND, 20, point)

    new_interp = TPI_jax.TP_Interpolant_ND(list(nodes))
    coeffs = np.asarray(new_interp.ComputeSplineCoefficientsND(values), dtype=np.float64)
    new_jit = jax.jit(new_interp.TPInterpolationND)
    new_warmup_ms, _ = time_once(new_jit, point)
    new_ms, _ = median_ms(new_jit, 20, point)

    old_interp = TPI_jax_old.TP_Interpolant_ND(list(nodes))
    old_interp.SetSplineCoefficientsND(coeffs)
    old_jit = jax.jit(old_interp.TPInterpolationND)
    old_warmup_ms, _ = time_once(old_jit, point)
    old_ms, _ = median_ms(old_jit, 20, point)

    rows.append((label, gsl_ms, old_warmup_ms, old_ms, new_warmup_ms, new_ms))

print()
print("Single-point latency table")
print("dim | single GSL (ms) | old JAX warmup (ms) | old JAX steady (ms) | new JAX warmup (ms) | new JAX steady (ms)")
for row in rows:
    print(f"{row[0]:>3} | {row[1]:15.3f} | {row[2]:19.3f} | {row[3]:18.3f} | {row[4]:19.3f} | {row[5]:18.3f}")
