# %% [markdown]
# # TPI usage tutorial: Cython/GSL and JAX backends
#
# `TPI` performs cubic B-spline interpolation with not-a-knot boundary
# conditions on N-dimensional Cartesian product grids. This tutorial walks
# through everything a user needs day to day:
#
#  1. **Creating splines** with the Cython/GSL backend (`import TPI`) and the
#     JAX backend (`import TPI_jax`).
#  2. **Saving minimal spline data** and reconstructing the spline later.
#  3. **Evaluating splines** — single points, batches, and, on the JAX side,
#     how `jit`, `vmap`, and `grad` change the game.
#  4. **Vector/tensor-valued interpolation** (both backends), including how to
#     combine previously saved per-component splines into one vectorized spline.
#  5. **Benchmarks** for construction and evaluation, with plain-language
#     explanations of what each number actually measures.
#  6. **Large grids and the dedicated fast 1D path (`Spline1D`)** — the
#     coefficient solve is banded (O(n) memory, hundreds of thousands of
#     points are fine), and a specialized 1D class squeezes out the last
#     factors for construction and sorted-batch evaluation.
#
# The file is written in jupytext "percent" format: every `# %%` marker is a
# code cell and every `# %% [markdown]` marker is a markdown cell, so you can
# either run it directly as a script,
#
# ```bash
# conda run -n test_esigmapy python doc/Tutorial_usage.py
# ```
#
# or convert it to a notebook with
#
# ```bash
# jupytext --to notebook doc/Tutorial_usage.py
# ```
#
# **Choosing a backend in one sentence:** if you call the spline one point at
# a time from ordinary Python, the Cython/GSL backend has the lowest latency;
# if you evaluate batches, need derivatives, or want the spline inside a
# JAX-compiled model, use the JAX backend.

# %%
import os
import sys
import time
import statistics

import numpy as np

# Make the repository importable when running this file from the doc/ directory
# or from a notebook whose working directory is doc/.
try:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:  # __file__ does not exist inside a notebook
    _repo_root = os.path.abspath("..")
for _candidate in (_repo_root, os.path.abspath(".")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import TPI       # Cython + GSL backend

# %% [markdown]
# ## A note on JAX and 64-bit floats
#
# JAX computes in 32-bit floating point unless told otherwise, while the GSL
# backend (and most scientific interpolation work) uses 64-bit. `TPI_jax`
# enables 64-bit mode **at import time**, so simply importing it is enough —
# but JAX fixes this setting when the first array is created, so make sure
# `import TPI_jax` (or your own `jax.config.update("jax_enable_x64", True)`)
# happens before any other JAX arrays are built in your program.

# %%
import jax
import jax.numpy as jnp

import TPI_jax   # JAX backend; enables 64-bit mode on import

# %% [markdown]
# ## 1. Creating splines
#
# Both backends share the same recipe:
#
#  1. Choose the grid: one 1D array of strictly ordered nodes per dimension.
#     The full grid is the Cartesian product of these arrays — you never build
#     the N-dimensional mesh yourself, the interpolant only needs the 1D axes.
#  2. Tabulate your function on that grid with `indexing='ij'` convention, so
#     `F[i, j, k]` is the function at `(x[i], y[j], z[k])`.
#  3. Construct the interpolant. Passing `F=` computes the spline coefficients
#     immediately; alternatively construct first and call
#     `ComputeSplineCoefficientsND(F)` later.
#
# We use a 3D example throughout. Everything works identically in any
# dimension (the test suite exercises up to 7D).

# %%
xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
zi = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
nodes = [xi, yi, zi]

def f(x, y, z):
    return np.sin(x) * np.arccos(y) * np.exp(z)

xx, yy, zz = np.meshgrid(xi, yi, zi, indexing="ij")
F = f(xx, yy, zz)
print("grid shape:", F.shape)

# %% [markdown]
# ### Cython/GSL backend

# %%
fI_gsl = TPI.TP_Interpolant_ND(nodes, F=F)

point = np.array([0.1692602, 0.2827312351474, -0.26624193])
print("exact:       ", f(*point))
print("GSL spline:  ", fI_gsl(point))

# %% [markdown]
# ### JAX backend
#
# The public API is identical. Construction does slightly more work up front:
# it LU-factorizes the 1D spline matrices (so repeated coefficient solves on
# the same grid are cheap) and prepares a compiled evaluator. The evaluator is
# compiled by JAX on **first use**, not at construction — see the benchmark
# section for what that "warmup" costs.

# %%
fI_jax = TPI_jax.TP_Interpolant_ND(nodes, F=F)

print("JAX spline:  ", float(fI_jax(point)))
print("GSL vs JAX:  ", float(fI_jax(point)) - fI_gsl(point))

# %% [markdown]
# The two backends agree to ~1e-15 (floating-point roundoff); the underlying
# algorithm — not-a-knot cubic B-splines — is the same.
#
# ### Aside: the 1D building blocks
#
# `BsplineBasis1D` exposes the pieces the interpolant is made of, which is
# useful for debugging or for building custom solvers: the B-spline basis
# functions at a point, and the spline collocation matrix with not-a-knot
# boundary rows. Both backends provide it.

# %%
basis = TPI.BsplineBasis1D(xi)
phi, knots = basis.AssembleSplineMatrix()
print("basis values at x=0.16 (only 4 are ever nonzero):")
print(np.asarray(basis.EvaluateBsplines(0.16)))
print("spline matrix shape:", np.asarray(phi).shape, "| knots:", np.asarray(knots).shape)

# %% [markdown]
# ## 2. Saving minimal spline data and reconstructing later
#
# A spline is fully determined by two things:
#
#  * the list of 1D node arrays (`nodes`), and
#  * the coefficient tensor, of shape `tuple(len(n_i) + 2 for n_i in nodes)`.
#
# Everything else (knot vectors, factored spline matrices, the compiled
# evaluator) is deterministically rebuilt from the nodes. So "save the spline"
# means: save the nodes and the coefficients. `GetSplineCoefficientsND()`
# returns the coefficients; passing `coeffs=` to the constructor (or calling
# `SetSplineCoefficientsND`) restores them **without re-solving** the linear
# systems — reconstruction only pays the setup cost.

# %%
coeffs = np.asarray(fI_jax.GetSplineCoefficientsND())
print("coefficient tensor shape:", coeffs.shape, "= grid shape + 2 per axis")

save_path = "/tmp/tpi_spline_3d.npz"
np.savez(save_path, coeffs=coeffs, **{f"nodes_{i}": n for i, n in enumerate(nodes)})
print("saved:", save_path, f"({os.path.getsize(save_path)/1024:.1f} KiB)")

# %%
with np.load(save_path) as data:
    loaded_nodes = [data[f"nodes_{i}"] for i in range(3)]
    loaded_coeffs = data["coeffs"]

# Reconstruction works with either backend, from the same file — the
# coefficients are backend-agnostic numbers.
fI_gsl_restored = TPI.TP_Interpolant_ND(loaded_nodes, coeffs=loaded_coeffs)
fI_jax_restored = TPI_jax.TP_Interpolant_ND(loaded_nodes, coeffs=loaded_coeffs)

print("restored GSL:", fI_gsl_restored(point))
print("restored JAX:", float(fI_jax_restored(point)))

# %% [markdown]
# ## 3. Evaluating splines
#
# ### 3.1 Single points, from plain Python
#
# Both backends accept a length-N NumPy array (or via `__call__`, anything
# array-like). Points outside the grid raise a `ValueError` naming the
# offending axis.

# %%
print("GSL:", fI_gsl.TPInterpolationND(point))
print("JAX:", float(fI_jax.TPInterpolationND(point)))

try:
    fI_jax.TPInterpolationND(np.array([0.5, 0.0, 0.0]))  # x outside [0.1, 0.25]
except ValueError as exc:
    print("out of range ->", exc)

# %% [markdown]
# ### 3.2 Batches of points
#
# Suppose you have M points stored as an `(M, N)` array. **Both backends**
# provide `TPInterpolationND_batched`, which evaluates all M points in one
# call. `__call__` dispatches on the input shape, so `fI(points)` with an
# `(M, N)` array does the same thing. If any point is out of range, a
# `ValueError` names the offending point and axis — and nothing is evaluated.
#
# Under the hood the two backends get there differently:
#
#  * **Cython/GSL** runs the per-point C evaluation in a single C loop. The
#    workspace setup and the Python-call overhead are paid once per *batch*
#    instead of once per *point*, and the GIL is released while the loop runs,
#    so other Python threads can make progress.
#  * **JAX** uses **`jax.vmap`** ("vectorizing map"): JAX takes the
#    *single-point* evaluation code and mechanically transforms it into a
#    program that processes the whole batch in one compiled call — same math,
#    but vectorized across points. You never write the loop, and there is no
#    per-point Python overhead.

# %%
rng = np.random.default_rng(42)
lows = np.array([n[0] for n in nodes])
highs = np.array([n[-1] for n in nodes])
points_batch = lows + rng.uniform(0.05, 0.95, size=(512, 3)) * (highs - lows)

values_gsl = fI_gsl.TPInterpolationND_batched(points_batch)              # one C loop
values_jax = np.asarray(fI_jax.TPInterpolationND_batched(points_batch))  # one vmap call

print("batch shapes:", values_gsl.shape, values_jax.shape)
print("max |GSL - JAX| over 512 points:", np.max(np.abs(values_gsl - values_jax)))
print("__call__ dispatches on shape:", np.array_equal(fI_gsl(points_batch), values_gsl))

try:
    fI_gsl.TPInterpolationND_batched(np.array([[0.16, 0.0, 0.0], [0.5, 0.0, 0.0]]))
except ValueError as exc:
    print("out of range ->", exc)  # names point 1, axis 0 (x outside [0.1, 0.25])

# %% [markdown]
# ### 3.3 `jit`: where the JAX backend actually shines
#
# JAX programs are *traced and compiled*: the first time a `jax.jit`-decorated
# function runs, JAX records the operations and hands them to the XLA compiler,
# which produces fast machine code specialized to your array shapes. That first
# call is slow (tens to hundreds of milliseconds — the **warmup**); every call
# after that reuses the compiled program (**steady state**).
#
# Two consequences worth internalizing:
#
#  * A single call from Python always pays a fixed dispatch cost (~10-20 µs)
#    to hand control from Python to the compiled code. For one point in low
#    dimensions the Cython backend (~1-3 µs) therefore stays faster.
#  * **Inside** a jitted function, that dispatch cost vanishes: the spline
#    evaluation is inlined and fused into your surrounding computation. This
#    is the intended way to use the JAX backend in a JAX codebase — the
#    interpolant becomes just another differentiable operation in your model.
#
# The methods of `TP_Interpolant_ND` are traceable, so you can call them
# directly inside your own jitted functions:

# %%
@jax.jit
def my_pipeline(pts):
    """Toy 'downstream model': spline evaluation composed with other math."""
    values = jax.vmap(fI_jax.TPInterpolationND)(pts)
    return jnp.sum(jnp.sin(values) ** 2)

pts_dev = jnp.asarray(points_batch)
result = my_pipeline(pts_dev)          # first call: traces + compiles (warmup)
result = my_pipeline(pts_dev)          # subsequent calls: compiled fast path
print("pipeline result:", float(result))

# %% [markdown]
# **Caveat — range checking inside `jit`:** when the interpolant is called
# with traced values (inside `jit`/`vmap`/`grad`), Python cannot inspect the
# numbers, so the out-of-range `ValueError` above cannot be raised. Instead
# the evaluation clamps to the boundary spline segment, i.e. out-of-range
# points are extrapolated with the outermost cubic polynomial. Validate your
# inputs before entering jitted code if range errors matter to you.
#
# ### 3.4 `grad`: derivatives of the interpolant
#
# Because the JAX evaluation is built from differentiable operations, JAX can
# differentiate **through the spline** with respect to the evaluation point.
# `jax.grad(f)` returns a function computing ∂f/∂X — the length-N gradient
# vector — by automatic differentiation of the actual cubic polynomials (no
# finite-difference stencils, no extra spline construction). This is exact for
# the spline (not for the underlying function it approximates).

# %%
grad_fn = jax.grad(fI_jax.TPInterpolationND)
g = np.asarray(grad_fn(jnp.asarray(point)))
print("gradient at point:", g)

# Compare against a crude central finite difference of the spline itself:
eps = 1e-6
fd = np.array([
    (float(fI_jax(point + eps * np.eye(3)[k])) - float(fI_jax(point - eps * np.eye(3)[k]))) / (2 * eps)
    for k in range(3)
])
print("finite-difference check:", fd)

# Transformations compose: gradient at every point of a batch, in one call.
batch_grads = jax.vmap(jax.grad(fI_jax.TPInterpolationND))(pts_dev[:5])
print("batched gradients shape:", np.asarray(batch_grads).shape)

# %% [markdown]
# ## 4. Vector- and tensor-valued interpolation (both backends)
#
# Often you interpolate several quantities on the *same grid* — components of
# a vector field, a waveform's amplitude and phase, a tensor of couplings.
# Building one scalar interpolant per component works but repeats the span
# search and basis computation M times per evaluation.
#
# `TP_Interpolant_ND_Vector` interpolates all components at once, and exists
# in **both backends** with the same constructor. You give it a `values_shape`
# — the shape of your function's *output* — and grid data `F` with those value
# axes trailing the grid axes. All components share the grid, knots, and
# spline matrices; the coefficient solve treats the value axes as extra
# right-hand sides, and evaluation returns an array of shape `values_shape`.
#
# In the Cython/GSL backend the span search and the products of B-spline basis
# functions are computed once per evaluation point in C and shared by all
# components — only the final coefficient accumulation scales with M, so a
# vector evaluation costs barely more than a *single* scalar one (see the
# benchmarks in section 5.5).

# %%
def g1(x, y, z):
    return np.sin(x) * np.arccos(y) * np.exp(z)

def g2(x, y, z):
    return np.cos(10 * x) * y * z

def g3(x, y, z):
    return x * y + z ** 2

F_vec = np.stack([g(xx, yy, zz) for g in (g1, g2, g3)], axis=-1)
print("vector-valued grid data shape:", F_vec.shape, "= grid shape + (3,)")

fI_vec_gsl = TPI.TP_Interpolant_ND_Vector(nodes, values_shape=(3,), F=F_vec)
fI_vec = TPI_jax.TP_Interpolant_ND_Vector(nodes, values_shape=(3,), F=F_vec)

value = np.asarray(fI_vec.TPInterpolationND(point))
print("GSL vector value at point:", fI_vec_gsl.TPInterpolationND(point))
print("JAX vector value at point:", value)
print("component-wise reference: ", np.array([g(*point) for g in (g1, g2, g3)]))

# %% [markdown]
# Everything from section 3 carries over: with the GSL backend you evaluate
# point by point from Python (each call now returning a length-3 vector); with
# the JAX backend you additionally get batching, jit, and differentiation.
# For a vector-valued function the derivative is a **Jacobian** (one gradient
# row per output component), so use `jax.jacfwd` instead of `jax.grad`:

# %%
batch_vec = np.asarray(fI_vec.TPInterpolationND_batched(points_batch[:8]))
print("batched vector evaluation shape:", batch_vec.shape)  # (M,) + values_shape

jac = np.asarray(jax.jacfwd(fI_vec.TPInterpolationND)(jnp.asarray(point)))
print("Jacobian shape:", jac.shape)  # values_shape + (N,)
print(jac)

# %% [markdown]
# `values_shape` can be any shape, not just a vector — e.g. `(2, 3)` for a
# matrix-valued function; evaluation then returns `(2, 3)` arrays and the
# coefficients have shape `grid + (2, 3)`. This works in both backends.
#
# ### 4.1 Combining previously saved per-component splines
#
# If you already have spline data for each component — built and saved
# separately over the years, but all on the same grid — you do **not** need to
# recompute anything. `FromComponentSplines` stacks existing per-component
# coefficients into one vector-valued interpolant. It exists on both backends'
# `TP_Interpolant_ND_Vector` and accepts either raw coefficient arrays or
# interpolant objects — from *either* backend, in any mix. So a pile of old
# Cython splines can become one GSL vector spline, one JAX vector spline, or
# both:

# %%
# Pretend these were built independently (here: one per component, GSL backend)
gsl_components = []
for g in (g1, g2, g3):
    comp = TPI.TP_Interpolant_ND(nodes, F=g(xx, yy, zz))
    gsl_components.append(comp)

# Combine the existing splines -- no coefficient solve happens here.
fI_combined_gsl = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(nodes, gsl_components)
fI_combined = TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(nodes, gsl_components)

print("combined value (GSL):", fI_combined_gsl.TPInterpolationND(point))
print("combined value (JAX):", np.asarray(fI_combined.TPInterpolationND(point)))
print("difference vs F_vec: ",
      np.max(np.abs(np.asarray(fI_combined.TPInterpolationND(point)) - value)))

# The minimal saved data for a vector spline is nodes + stacked coefficients,
# and it is backend-agnostic, exactly as in section 2:
stacked = np.asarray(fI_combined.GetSplineCoefficientsND())
print("stacked coefficients shape:", stacked.shape)
fI_reloaded_gsl = TPI.TP_Interpolant_ND_Vector(nodes, values_shape=(3,), coeffs=stacked)
fI_reloaded = TPI_jax.TP_Interpolant_ND_Vector(nodes, values_shape=(3,), coeffs=stacked)

# %% [markdown]
# ## 5. Benchmarks
#
# Before the numbers, three concepts so the measurements make sense without
# any JAX background:
#
#  * **Warmup vs steady state.** The first call to a JAX function includes
#    compilation (tracing your Python, running the XLA optimizer, generating
#    machine code). That is a one-time cost per function *and* per input
#    shape. We report it separately as "warmup"; "steady" is the median of
#    repeated calls afterwards, which is what you pay in production loops.
#    Never judge JAX by its first call.
#  * **Asynchronous dispatch.** JAX calls return immediately with a promise of
#    the result while the computation may still be running. Timing naively
#    would measure how fast JAX *queues* work, not how fast it *does* work.
#    `jax.block_until_ready(...)` waits for completion, so all timings below
#    include the actual computation.
#  * **Median, not mean.** Operating-system hiccups make individual timings
#    spiky; the median of many repetitions is a robust "typical cost".

# %%
def median_ms(fn, *args, repeat=30):
    """Median wall-clock milliseconds per call, waiting for JAX completion."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)

def once_ms(fn, *args):
    """Wall-clock milliseconds of a single call (used for JIT warmup)."""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) * 1e3

# %% [markdown]
# ### 5.1 Spline construction
#
# **What this measures:** everything needed to go from raw grid data to a
# usable interpolant — building knot vectors and spline matrices, and solving
# the linear systems for the coefficients. You pay this once per dataset (or
# whenever the data on the grid changes), not per evaluation.
#
# The JAX row excludes compilation of the *evaluator* (that belongs to the
# first evaluation, measured next); but note that the very first JAX
# construction in a process also compiles the LU solve — shown as "first".

# %%
gsl_construct = median_ms(lambda: TPI.TP_Interpolant_ND(nodes, F=F), repeat=20)

jax_construct_first = once_ms(lambda: TPI_jax.TP_Interpolant_ND(nodes, F=F))
jax_construct = median_ms(lambda: TPI_jax.TP_Interpolant_ND(nodes, F=F), repeat=20)

print(f"construction, 3D grid {F.shape}:")
print(f"  GSL/Cython                {gsl_construct:8.3f} ms")
print(f"  JAX (first in process)    {jax_construct_first:8.3f} ms")
print(f"  JAX (steady)              {jax_construct:8.3f} ms")

# %% [markdown]
# If you rebuild coefficients repeatedly on a *fixed grid* (e.g. fitting many
# datasets), reuse the interpolant and call `ComputeSplineCoefficientsND(F)`
# — the factored spline matrices are cached, so only the solves rerun:

# %%
gsl_solve = median_ms(fI_gsl.ComputeSplineCoefficientsND, F, repeat=20)
jax_solve = median_ms(fI_jax.ComputeSplineCoefficientsND, F, repeat=20)
print(f"coefficient solve only:  GSL {gsl_solve:7.3f} ms | JAX {jax_solve:7.3f} ms")

# %% [markdown]
# ### 5.2 Single-point evaluation
#
# **What this measures:** the latency of asking for the spline value at one
# point, called from ordinary Python — the classic inner-loop scenario of
# legacy scientific code.
#
# Three variants:
#
#  * **GSL single** — the Cython call. Pure compiled C behind a thin wrapper;
#    this is the latency floor for one-at-a-time calls from Python.
#  * **JAX single (public)** — `fI_jax.TPInterpolationND(point)` from Python.
#    Includes input validation, range checks, and the fixed Python-to-compiled
#    dispatch cost. The compute inside is a few hundred nanoseconds; the
#    ~10-20 µs you see is almost entirely dispatch overhead.
#  * **JAX inside jit** — the same evaluation embedded in a compiled caller
#    (here amortized over a 512-point batch, i.e. the per-point cost when the
#    spline lives inside a jitted model). This is the number that matters in
#    a JAX codebase.
#
# One honest footnote about the "warmup" rows below: compiled programs are
# cached per function and input shape for the life of the process, and this
# tutorial already evaluated these splines in earlier sections — so the
# "warmup" rows here are cache *hits*, not real compilations. The gradient
# benchmark in section 5.4 uses a transformation not seen before in this
# process, so its warmup row shows a genuine first-time compilation cost.

# %%
gsl_single = median_ms(fI_gsl.TPInterpolationND, point, repeat=100)

jax_single_warmup = once_ms(fI_jax.TPInterpolationND, point)  # includes compile
jax_single = median_ms(fI_jax.TPInterpolationND, point, repeat=100)

batch_warmup = once_ms(fI_jax.TPInterpolationND_batched, points_batch)
jax_batch = median_ms(fI_jax.TPInterpolationND_batched, points_batch, repeat=50)
jax_per_point_in_jit = jax_batch / len(points_batch)

print("single-point evaluation, 3D:")
print(f"  GSL single                {gsl_single*1e3:9.2f} us")
print(f"  JAX single warmup         {jax_single_warmup:9.2f} ms   (cache hit; see note above)")
print(f"  JAX single (public call)  {jax_single*1e3:9.2f} us")
print(f"  JAX per point inside jit  {jax_per_point_in_jit*1e3:9.2f} us   (from 512-pt batch)")

# %% [markdown]
# **How to read this:** for isolated single points from Python, GSL wins —
# no amount of JAX tuning removes the Python-to-compiled dispatch floor. But
# the per-point cost *inside* compiled code is far below even the GSL single
# call, which is why batching (or embedding in jit) flips the comparison.
#
# ### 5.3 Batched evaluation
#
# **What this measures:** total time to evaluate the same 512 points three
# ways — looping in Python over the Cython backend (per-point overhead × 512),
# the Cython backend's batched call (one C loop, overhead paid once), and the
# vectorized JAX call.

# %%
gsl_loop = median_ms(
    lambda pts: np.array([fI_gsl.TPInterpolationND(p) for p in pts]), points_batch, repeat=20
)
gsl_batch = median_ms(fI_gsl.TPInterpolationND_batched, points_batch, repeat=50)
print(f"512-point batch, 3D:")
print(f"  GSL python loop           {gsl_loop:9.3f} ms  ({gsl_loop/512*1e3:6.2f} us/point)")
print(f"  GSL batched               {gsl_batch:9.3f} ms  ({gsl_batch/512*1e3:6.2f} us/point)")
print(f"  JAX warmup                {batch_warmup:9.3f} ms   (cache hit; see note above)")
print(f"  JAX vmap batch            {jax_batch:9.3f} ms  ({jax_batch/512*1e3:6.2f} us/point)")
print(f"  GSL batched vs loop       {gsl_loop/gsl_batch:9.1f}x")
print(f"  accuracy: max |GSL batched - JAX batch| = "
      f"{np.max(np.abs(np.asarray(fI_gsl.TPInterpolationND_batched(points_batch)) - np.asarray(fI_jax.TPInterpolationND_batched(points_batch)))):.2e}")

# %% [markdown]
# **How to read this:** never loop over points in Python — the batched GSL
# call does bit-for-bit the same arithmetic with the Python overhead and the
# per-call workspace allocations paid once per batch. That overhead is a
# fixed cost per point, so the gain is largest where the actual evaluation is
# cheapest: roughly 5x in 1D, shrinking towards ~1.3x by 4D where the 4^N
# coefficient contraction dominates. Between the two batched paths the winner
# depends on dimension, grid size, and batch size: the GSL loop has
# essentially zero fixed cost and wins small batches outright, while the
# compiled JAX kernel vectorizes *across* points and pulls ahead for large
# batches in higher dimensions. Both agree to ~1e-14, so from plain Python
# simply use the batched call of whichever backend you already hold.

# %% [markdown]
# ### 5.4 Gradients
#
# **What this measures:** the cost of a derivative. With the Cython backend
# your only option is finite differences — 2N extra spline evaluations per
# point, plus truncation error. JAX computes the exact spline gradient in one
# compiled call, batched over all points.

# %%
grad_batch_fn = jax.jit(jax.vmap(jax.grad(fI_jax.TPInterpolationND)))
grad_warmup = once_ms(grad_batch_fn, pts_dev)
grad_batch = median_ms(grad_batch_fn, pts_dev, repeat=50)

fd_batch = median_ms(
    lambda pts: np.array(
        [
            [
                (fI_gsl.TPInterpolationND(p + eps * np.eye(3)[k])
                 - fI_gsl.TPInterpolationND(p - eps * np.eye(3)[k])) / (2 * eps)
                for k in range(3)
            ]
            for p in pts
        ]
    ),
    points_batch,
    repeat=5,
)

print("gradients at 512 points, 3D:")
print(f"  GSL finite differences    {fd_batch:9.3f} ms  (approximate)")
print(f"  JAX grad warmup           {grad_warmup:9.3f} ms   (genuine first-time compile)")
print(f"  JAX vmap(grad) batch      {grad_batch:9.3f} ms  (exact)")

# %% [markdown]
# ### 5.5 Vector-valued evaluation
#
# **What this measures:** the saving from interpolating M components in one
# interpolant instead of M scalar interpolants. In both backends the span
# search and basis computation are shared; only the final coefficient
# accumulation scales with M.

# %%
gsl_scalar_components = [
    TPI.TP_Interpolant_ND(nodes, F=g(xx, yy, zz)) for g in (g1, g2, g3)
]
gsl_vec_single = median_ms(fI_vec_gsl.TPInterpolationND, point, repeat=100)
gsl_scalar_x3 = median_ms(
    lambda p: [comp.TPInterpolationND(p) for comp in gsl_scalar_components], point, repeat=100
)

vec_warmup = once_ms(fI_vec.TPInterpolationND, point)
vec_single = median_ms(fI_vec.TPInterpolationND, point, repeat=100)
scalar_x3 = 3 * jax_single

print("3-component vector spline, 3D, single point from Python:")
print(f"  GSL 1 vector eval         {gsl_vec_single*1e3:9.2f} us")
print(f"  GSL 3 scalar evals        {gsl_scalar_x3*1e3:9.2f} us")
print(f"  JAX 1 vector eval         {vec_single*1e3:9.2f} us")
print(f"  JAX 3 scalar evals        {scalar_x3*1e3:9.2f} us")

# %% [markdown]
# The GSL vector evaluation costs barely more than a *single* scalar call —
# the per-point bookkeeping dominates and is paid once. The saving grows with
# the number of components: below, the same comparison with a 100-component
# spline on the same grid.

# %%
ncomp = 100
F_many = np.stack([np.sin((k + 1) * xx) * np.exp(yy) + zz for k in range(ncomp)], axis=-1)
fI_many_gsl = TPI.TP_Interpolant_ND_Vector(nodes, values_shape=(ncomp,), F=F_many)
many_scalars_gsl = [
    TPI.TP_Interpolant_ND(nodes, F=np.ascontiguousarray(F_many[..., k])) for k in range(ncomp)
]

gsl_vec100 = median_ms(fI_many_gsl.TPInterpolationND, point, repeat=100)
gsl_scalar_x100 = median_ms(
    lambda p: [comp.TPInterpolationND(p) for comp in many_scalars_gsl], point, repeat=30
)
print(f"100-component vector spline, 3D (GSL):")
print(f"  1 vector eval             {gsl_vec100*1e3:9.2f} us")
print(f"  100 scalar evals          {gsl_scalar_x100*1e3:9.2f} us")
print(f"  speedup                   {gsl_scalar_x100/gsl_vec100:9.1f}x")

# %% [markdown]
# Vector-valued splines batch too — `TPInterpolationND_batched` returns
# `(M,) + values_shape`, sharing the span search across components *and* the
# setup across points:

# %%
gsl_vec_batch = median_ms(fI_vec_gsl.TPInterpolationND_batched, points_batch, repeat=50)
vec_batch_warmup = once_ms(fI_vec.TPInterpolationND_batched, points_batch)
jax_vec_batch = median_ms(fI_vec.TPInterpolationND_batched, points_batch, repeat=50)
print("3-component vector spline, 512-point batch:")
print(f"  GSL batched               {gsl_vec_batch:9.3f} ms  -> shape {np.asarray(fI_vec_gsl.TPInterpolationND_batched(points_batch)).shape}")
print(f"  JAX warmup                {vec_batch_warmup:9.3f} ms")
print(f"  JAX vmap batch            {jax_vec_batch:9.3f} ms")

# %% [markdown]
# ### 5.6 Takeaways
#
#  * **Construction** costs are comparable; both are one-time costs.
#  * **One-at-a-time Python calls:** Cython/GSL is the right tool
#    (microseconds vs tens of microseconds).
#  * **Batches:** never loop over points in Python — both backends provide
#    `TPInterpolationND_batched` (also reachable as `fI(points)`). The GSL
#    batch is a single GIL-releasing C loop and wins small-to-medium batches;
#    the JAX kernel vectorizes across points and pulls ahead for large
#    batches in higher dimensions.
#  * **Inside a JAX model:** embedding the interpolant in jitted code makes
#    its per-point cost sub-microsecond and gives you exact gradients for
#    free — this is the setting the JAX backend was built for.
#  * **First calls are slow by design** (compilation). If your process is
#    short-lived and calls the spline only a few times, that warmup may
#    dominate; long-running analyses amortize it to nothing.
#  * **Vector-valued splines** share all per-point bookkeeping across
#    components — prefer one vector interpolant over M scalar ones, in either
#    backend. For one-at-a-time vector values from Python, the GSL
#    `TP_Interpolant_ND_Vector` is the fastest option by a wide margin.
#  * **Purely 1D problems:** use `Spline1D` (section 6) — fastest
#    construction (tridiagonal solve, on-device in JAX) and fastest sorted
#    batch evaluation (monotone C walk in the GSL backend).

# %% [markdown]
# ## 6. Large grids and the dedicated fast 1D path (`Spline1D`)
#
# ### 6.1 Large grids just work now
#
# The not-a-knot collocation matrix of each axis is banded (bandwidths 4/4),
# and the coefficient solve now exploits that: assembly and solve are
# **O(n) in memory and time** per axis (`scipy.linalg.solve_banded`; small
# axes keep the historical dense-inverse behavior, which is faster for the
# many right-hand sides of N-D grids). Previously the matrix was assembled
# dense and inverted — a 1D spline on 500,000 points would have needed about
# **2 TB** of memory; it now needs a few hundred MB of process total and
# about a quarter of a second, in both backends and with unchanged results
# (coefficients agree with the old solver to ~1e-13).
#
# Nothing about the API changes on large grids — construct
# `TP_Interpolant_ND` exactly as before:

# %%
rng = np.random.default_rng(42)
n_large = 200_000
x_large = np.sort(rng.uniform(0.0, 100.0, n_large))
x_large[0], x_large[-1] = 0.0, 100.0
F_large = np.sin(x_large) * np.exp(-0.01 * x_large)

t0 = time.perf_counter()
fI_large = TPI.TP_Interpolant_ND([x_large], F=F_large)
print(f"GSL general path, n={n_large}: {(time.perf_counter()-t0)*1e3:.0f} ms")

t0 = time.perf_counter()
fI_large_jax = TPI_jax.TP_Interpolant_ND([x_large], F=F_large)
print(f"JAX general path, n={n_large}: {(time.perf_counter()-t0)*1e3:.0f} ms")

# %% [markdown]
# ### 6.2 `Spline1D`: the dedicated 1D class
#
# For purely 1D problems both backends additionally provide `Spline1D`,
# which is substantially faster than the general machinery because it
# exploits everything the 1D case offers:
#
#  * **Construction** solves the classic *tridiagonal* not-a-knot system for
#    the node derivatives — no search anywhere, the sortedness of the node
#    array is used to the fullest. In the JAX backend the solve is
#    `jax.lax.linalg.tridiagonal_solve`, so construction is jit-compatible
#    and runs **entirely on the JAX device (GPU-capable, no host callback)**.
#  * **Evaluation** works directly on the native per-interval (Hermite)
#    representation. The Cython backend evaluates *sorted* query batches with
#    a monotone span walk in C — O(M + n) instead of O(M log n) — and falls
#    back to a vectorized binary search for unsorted input. The JAX
#    evaluator is jit/vmap-compatible and differentiable w.r.t. the query
#    points.
#
# Requirements: strictly increasing nodes, at least 4 of them, scalar values.

# %%
s_gsl = TPI.Spline1D(x_large, F=F_large)
s_jax = TPI_jax.Spline1D(x_large, F=F_large)

xq_sorted = np.sort(rng.uniform(0.0, 100.0, 500_000))
print("GSL Spline1D :", s_gsl(xq_sorted[:3]))
print("JAX Spline1D :", np.asarray(s_jax(xq_sorted[:3])))
print("general path :", fI_large.TPInterpolationND_batched(xq_sorted[:3, None]))

# %% [markdown]
# How much faster? Construction and a 500k-point sorted evaluation on the
# 200k-node grid (medians; the first JAX call pays JIT compilation, see
# section 5):

# %%
con_gsl_1d = median_ms(lambda: TPI.Spline1D(x_large, F=F_large), repeat=10)
con_jax_1d = median_ms(lambda: TPI_jax.Spline1D(x_large, F=F_large).poly, repeat=10)
con_gsl_gen = median_ms(lambda: TPI.TP_Interpolant_ND([x_large], F=F_large), repeat=10)

ev_gsl_1d = median_ms(s_gsl, xq_sorted, repeat=10)
ev_jax_1d = median_ms(s_jax, xq_sorted, repeat=10)
# The general path costs tens of microseconds *per point* on grids this
# large, so it is timed on a 20k subset and scaled to the full batch size.
subset = xq_sorted[::25][:, None]
ev_gsl_gen = median_ms(fI_large.TPInterpolationND_batched, subset, repeat=3)
ev_gsl_gen_scaled = ev_gsl_gen * (len(xq_sorted) / len(subset))

print(f"construction n={n_large}:")
print(f"  Spline1D GSL {con_gsl_1d:8.1f} ms | Spline1D JAX {con_jax_1d:8.1f} ms "
      f"| general GSL {con_gsl_gen:8.1f} ms")
print(f"evaluation of {len(xq_sorted)} sorted points:")
print(f"  Spline1D GSL {ev_gsl_1d:8.1f} ms | Spline1D JAX {ev_jax_1d:8.1f} ms "
      f"| general GSL batched ~{ev_gsl_gen_scaled:8.0f} ms (scaled from 20k points)")

# %% [markdown]
# Accuracy is identical to the general path (same spline, different — and on
# pathological grids equally stable — factorization):

# %%
check = xq_sorted[::25]
diff = np.abs(s_gsl(check) - fI_large.TPInterpolationND_batched(check[:, None]))
print(f"max |Spline1D - general path| over {len(check)} points:", diff.max())

# %% [markdown]
# ### 6.3 Saving `Spline1D` data, and loading old saved splines
#
# A `Spline1D` is fully determined by `(x, F)` — saving those two arrays and
# reconstructing is exact and cheap. For interop with the general classes
# (and with data saved by *older* versions of TPI), `to_coefficients()`
# exports the standard `(n + 2,)` TPI coefficient vector, and the `coeffs=`
# constructor argument accepts one:

# %%
c_1d = s_gsl.to_coefficients()

# old-style saved data -> fast 1D path (works for any B-form coefficients
# on these nodes, whether or not they came from an interpolation solve)
s_from_old = TPI.Spline1D(x_large, coeffs=c_1d)
# fast 1D path -> general ND class (e.g. to embed in an N-D workflow)
fI_from_1d = TPI.TP_Interpolant_ND([x_large], coeffs=c_1d)

print("round-trip agreement:",
      np.abs(s_from_old(xq_sorted[:1000]) - s_gsl(xq_sorted[:1000])).max())

# %% [markdown]
# On the JAX side, construction itself can live inside `jit` (and on GPU)
# through the functional core: `spline_1d_hermite(x, F)` returns the
# per-interval coefficients, `spline_1d_evaluate(x, poly, xq)` evaluates
# them. Both are pure JAX functions:

# %%
@jax.jit
def build_and_eval(x, F, xq):
    poly = TPI_jax.spline_1d_hermite(x, F)
    return TPI_jax.spline_1d_evaluate(x, poly, xq)

# warmup: trigger JIT compilation before timing/inspection
_ = build_and_eval(x_large, F_large, xq_sorted[:10])
print("jitted build+eval:", np.asarray(_)[:3])

# %% [markdown]
# ## Quick reference
#
# ```python
# # --- build ---
# fI = TPI.TP_Interpolant_ND(nodes, F=F)          # Cython/GSL
# fI = TPI_jax.TP_Interpolant_ND(nodes, F=F)      # JAX
#
# # --- save / restore (backend-agnostic) ---
# c = np.asarray(fI.GetSplineCoefficientsND())
# fI2 = TPI_jax.TP_Interpolant_ND(nodes, coeffs=c)
#
# # --- evaluate ---
# y  = fI(x)                                      # single point
# ys = fI.TPInterpolationND_batched(xs)           # (M, N) -> (M,)   [both backends]
# ys = fI(xs)                                     # same, __call__ dispatches on shape
# g  = jax.grad(fI.TPInterpolationND)(x)          # dI/dx            [JAX]
# ys = jax.vmap(fI.TPInterpolationND)(xs)         # inside your own jit/vmap
#
# # --- vector/tensor-valued (both backends) ---
# fV = TPI.TP_Interpolant_ND_Vector(nodes, values_shape=(M,), F=F_vec)      # Cython/GSL
# fV = TPI_jax.TP_Interpolant_ND_Vector(nodes, values_shape=(M,), F=F_vec)  # JAX
# fV = TPI.TP_Interpolant_ND_Vector.FromComponentSplines(nodes, [fI_1, ..., fI_M])
# fV = TPI_jax.TP_Interpolant_ND_Vector.FromComponentSplines(nodes, [fI_1, ..., fI_M])
# v  = fV(x)                                      # shape (M,), either backend
# J  = jax.jacfwd(fV.TPInterpolationND)(x)        # Jacobian, shape (M, N)   [JAX]
#
# # --- dedicated 1D fast path (both backends; strictly increasing x, n >= 4) ---
# s  = TPI.Spline1D(x, F=F)                       # tridiagonal solve, O(n)
# s  = TPI_jax.Spline1D(x, F=F)                   # same, solve on JAX device
# ys = s(xq)                                      # sorted xq -> O(M+n) C walk [GSL]
# c  = s.to_coefficients()                        # standard (n+2,) TPI format
# s2 = TPI.Spline1D(x, coeffs=c)                  # load old saved coefficients
# poly = TPI_jax.spline_1d_hermite(x, F)          # pure-JAX build (jit/GPU-safe)
# ys = TPI_jax.spline_1d_evaluate(x, poly, xq)    # pure-JAX eval (grad w.r.t. xq)
# ```
