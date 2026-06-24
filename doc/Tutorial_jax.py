# %%
# JAX backend tutorial for TPI.
#
# This script mirrors the existing Cython/GSL usage walkthrough and adds a
# comparison section for accuracy and speed against the legacy backend.

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jax

jax.config.update("jax_enable_x64", True)

import TPI
import TPI_jax


def _time_repeat(fn, repeat, *args):
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn(*args)
    end = time.perf_counter()
    return end - start, result


def _time_repeat_blocking(fn, repeat, *args):
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
    end = time.perf_counter()
    return end - start, result


def _make_mesh(nodes):
    return np.meshgrid(*nodes, indexing="ij")


def _plot_basis_examples():
    # %%
    # ## B-spline basis functions
    #
    # Plot the cubic B-spline basis functions for a non-uniform knot vector.
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    b = TPI_jax.BsplineBasis1D(x1)

    x = np.linspace(x1[0], x1[-1], 200)
    Bspl = np.array([b.EvaluateBsplines(xi) for xi in x])

    for i in range(Bspl.shape[1]):
        plt.plot(x, Bspl[:, i], label=i)
    plt.xlim([x1[0], x1[-1]])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\mathcal{B}_3^n(x)$")
    plt.legend(loc="best")
    plt.show()


def _plot_spline_matrix():
    # %%
    # ## Knots and spline matrix with not-a-knot boundary conditions
    x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12.0])
    b = TPI_jax.BsplineBasis1D(x1)
    b_phi, b_knots = b.AssembleSplineMatrix()

    print("Knots:")
    print(np.asarray(b_knots))
    print()
    print("Spline matrix:")
    print(pd.DataFrame(np.asarray(b_phi)))

    plt.imshow(np.asarray(b_phi))
    plt.colorbar()
    plt.title("JAX spline matrix")
    plt.show()


def _one_dimensional_interpolation():
    # %%
    # ## 1D spline interpolation
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    X = [xi]

    def f(x):
        return np.cos(40.0 * x)

    xx = np.linspace(xi[0], xi[-1], 200)
    yy = f(xx)

    fI = TPI_jax.TP_Interpolant_ND(X, F=f(xi))
    fIv = np.vectorize(fI)

    plt.plot(xi, f(xi), "o", label="data")
    plt.plot(xx, yy, "k-", label=r"$f(x)$")
    plt.plot(xx, fIv(xx), "r--", label=r"$I[f](x)$")
    plt.legend(loc=3)
    plt.show()


def _two_dimensional_interpolation():
    # %%
    # ## 2D TP spline interpolation
    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    X = [xi, yi]

    def f(x, y):
        return np.sin(x) * np.arccos(y)

    xx, yy = _make_mesh(X)
    F = f(xx, yy)

    fI = TPI_jax.TP_Interpolant_ND(X, F=F)

    print("Interpolant at a point:")
    print(fI([0.134, -0.28]))

    XX, YY = np.meshgrid(np.linspace(xi[0], xi[-1], 50), np.linspace(yi[0], yi[-1], 50), indexing="ij")
    FF = f(XX, YY)
    ZZ = np.array([fI([XX[i][j], YY[i][j]]) for i in np.arange(XX.shape[0]) for j in np.arange(XX.shape[1])]).reshape(XX.shape)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
    im = ax1.imshow(np.rot90(FF), cmap=plt.cm.gist_earth_r, extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    plt.colorbar(im, ax=ax1, label="$f(x,y)$")

    im = ax2.imshow(np.rot90(ZZ), cmap=plt.cm.gist_earth_r, extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$y$")
    plt.colorbar(im, ax=ax2, label="$I[f](x,y)$")

    im = ax3.imshow(np.rot90(FF - ZZ), cmap=plt.cm.gist_earth_r, extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$y$")
    plt.colorbar(im, ax=ax3, label="abs error")

    im = ax4.imshow(np.rot90((FF - ZZ) / FF), cmap=plt.cm.gist_earth_r, extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect="auto")
    ax4.set_xlabel(r"$x$")
    ax4.set_ylabel(r"$y$")
    plt.colorbar(im, ax=ax4, label="rel error")

    plt.tight_layout()
    plt.show()


def _four_dimensional_interpolation():
    # %%
    # ## 4D TP spline interpolation
    x1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    x2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    x3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    x4 = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])
    X = [x1, x2, x3, x4]

    def f(x1, x2, x3, x4):
        return np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4)

    F = f(*_make_mesh(X))
    fI = TPI_jax.TP_Interpolant_ND(X, F=F)

    y = (0.238, -0.97, 0.93, -0.251)
    print("4D example:")
    print("f(y)   =", f(*y))
    print("I[f](y)=", fI(y))
    print("relative error =", (f(*y) - fI(y)) / f(*y))


def _accuracy_and_speed_comparison():
    # %%
    # ## Accuracy and speed comparisons
    #
    # Compare the JAX backend against the Cython/GSL backend using the same
    # node sets and functions from the original tutorial.
    comparisons = []

    cases = []

    xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
    yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
    wi = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])

    cases.append(
        (
            "1D",
            [xi],
            lambda x: np.cos(40.0 * x),
            np.array([0.16]),
            np.linspace(xi[0], xi[-1], 200)[:, None],
        )
    )
    cases.append(
        (
            "2D",
            [xi, yi],
            lambda x, y: np.sin(x) * np.arccos(y),
            np.array([0.134, -0.28]),
            np.array([[0.16, 0.28], [0.123, -0.2], [0.2, 0.0]]),
        )
    )
    cases.append(
        (
            "3D",
            [xi, yi, zi],
            lambda x, y, z: np.sin(x) * np.arccos(y) * np.exp(z),
            np.array([0.1692602, 0.2827312351474, -0.26624193]),
            np.array(
                [
                    [0.1692602, 0.2827312351474, -0.26624193],
                    [0.11, -0.2, 0.4],
                    [0.235, 0.6, 0.1],
                ]
            ),
        )
    )
    cases.append(
        (
            "4D",
            [xi, yi, zi, wi],
            lambda x1, x2, x3, x4: np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4),
            np.array([0.238, -0.97, 0.93, -0.251]),
            np.array(
                [
                    [0.238, -0.97, 0.93, -0.251],
                    [0.16, 0.28, -0.26, 0.05],
                    [0.11, -0.2, 0.4, 1.0],
                ]
            ),
        )
    )

    for label, nodes, func, point, batch_points in cases:
        mesh = _make_mesh(nodes)
        F = func(*mesh)

        gsl = TPI.TP_Interpolant_ND(nodes, F=F)
        jax_interp = TPI_jax.TP_Interpolant_ND(nodes, F=F)

        gsl_value = float(np.asarray(gsl.TPInterpolationND(point)))
        jax_value = float(np.asarray(jax_interp.TPInterpolationND(point)))

        gsl_single_time, _ = _time_repeat(gsl.TPInterpolationND, 200, point)
        jax_jit = jax.jit(jax_interp.TPInterpolationND)
        jax_warmup_time, _ = _time_repeat_blocking(jax_jit, 1, point)
        jax_single_time, _ = _time_repeat_blocking(jax_jit, 200, point)

        gsl_batch_time, _ = _time_repeat(lambda pts: np.array([float(np.asarray(gsl.TPInterpolationND(p))) for p in pts]), 20, batch_points)
        jax_batch_jit = jax.jit(jax_interp.TPInterpolationND_batched)
        jax_batch_warmup_time, _ = _time_repeat_blocking(jax_batch_jit, 1, batch_points)
        jax_batch_time, _ = _time_repeat_blocking(jax_batch_jit, 20, batch_points)

        comparisons.append(
            {
                "dim": label,
                "single max abs diff": abs(gsl_value - jax_value),
                "GSL single (ms)": 1e3 * gsl_single_time / 200.0,
                "JAX warmup single (ms)": 1e3 * jax_warmup_time,
                "JAX steady single (ms)": 1e3 * jax_single_time / 200.0,
                "GSL batch (ms)": 1e3 * gsl_batch_time / 20.0,
                "JAX warmup batch (ms)": 1e3 * jax_batch_warmup_time,
                "JAX steady batch (ms)": 1e3 * jax_batch_time / 20.0,
            }
        )

    df = pd.DataFrame(comparisons)
    print(df.to_string(index=False))

    print()
    print("Notes:")
    print("- JAX 64-bit mode is enabled at import time.")
    print("- JAX warmup includes the first JIT compilation call; steady-state timings exclude it.")
    print("- For very high-dimensional or ill-conditioned grids, tiny matrix differences can be amplified in coefficient solves.")


def main():
    _plot_basis_examples()
    _plot_spline_matrix()
    _one_dimensional_interpolation()
    _two_dimensional_interpolation()
    _four_dimensional_interpolation()
    _accuracy_and_speed_comparison()


if __name__ == "__main__":
    main()
