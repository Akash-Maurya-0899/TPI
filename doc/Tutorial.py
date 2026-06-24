# %%
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import TPI

# %% [markdown]
# ## B-spline basis functions

# %% [markdown]
# Plot the nth non-uniform B-spline basis function of degree 3 with knots x1

# %%
x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
b = TPI.BsplineBasis1D(x1)

x = np.linspace(x1[0], x1[-1], 200)
Bspl = np.array(map(b.EvaluateBsplines, x))

for i in range(len(Bspl.T)):
    plt.plot(x, Bspl.T[i], label=i)
plt.xlim([x1[0], x1[-1]]);
plt.xlabel(r'$x$')
plt.ylabel(r'$\mathcal{B}_3^n(x)$');

# %% [markdown]
# ## Knots and spline matrix with "not-a-knot" boundary conditions

# %%
x1 = np.array([1.1, 3.2, 5.1, 7.2, 9.3, 12])
b = TPI.BsplineBasis1D(x1)
b_phi, b_knots = b.AssembleSplineMatrix()

b_knots

# %% [markdown]
# The spline matrix is not tridiagonal near the boundaries due to the "not-a-knot" boundary conditions.

# %%
pd.DataFrame(b_phi)

# %%
plt.imshow(b_phi);
plt.colorbar();

# %% [markdown]
# ## 1D spline interpolation

# %%
xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
X = [xi]

def f(x):
    return np.cos(40*x)

fI = TPI.TP_Interpolant_ND(X, F=f(xi))
fIv = np.vectorize(fI)

# %%
x = np.linspace(xi[0], xi[-1])
plt.plot(xi, f(xi), 'o', label='data')
plt.plot(x, f(x), 'k-', label=r'$f(x)$')
plt.plot(x, fIv(x), 'r--', label=r'$I[f](x)$')
plt.legend(loc=3);

# %% [markdown]
# ## 2D TP spline interpolation

# %%
xi = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
yi = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
X = [xi, yi]

def f(x,y):
    return np.sin(x) * np.arccos(y)

xx, yy = np.meshgrid(xi, yi, indexing='ij')
F = f(xx, yy)

fI = TPI.TP_Interpolant_ND(X, F=F)

# %% [markdown]
# Evaluate the interpolant at a point within the 2D regular grid formed by the Cartesian product of `xi` and `yi`:

# %%
fI([0.134, -0.28])

# %% [markdown]
# Evaluate the interpolant and the original function on a finer grid:

# %%
XX, YY = np.meshgrid(np.linspace(xi[0], xi[-1], 50), np.linspace(yi[0], yi[-1], 50), indexing='ij')
FF = f(XX, YY)
ZZ = np.array([fI([XX[i][j], YY[i][j]]) 
               for i in np.arange(XX.shape[0]) 
               for j in np.arange(XX.shape[1])]
             ).reshape(XX.shape)

# %% [markdown]
# Plot the function, interpolant and errors.

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
im = ax1.imshow(np.rot90(FF), cmap=plt.cm.gist_earth_r,
           extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect='auto')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
plt.colorbar(im, ax=ax1, label='$f(x,y)$');

im = ax2.imshow(np.rot90(ZZ), cmap=plt.cm.gist_earth_r,
           extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect='auto')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$y$')
plt.colorbar(im, ax=ax2, label='$I[f](x,y)$');

im = ax3.imshow(np.rot90(FF - ZZ), cmap=plt.cm.gist_earth_r,
           extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect='auto')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$y$')
plt.colorbar(im, ax=ax3, label='abs error $f(x,y) - I[f](x,y)$');

im = ax4.imshow(np.rot90((FF - ZZ)/FF), cmap=plt.cm.gist_earth_r,
           extent=[xi[0], xi[-1], yi[0], yi[-1]], aspect='auto')
ax4.set_xlabel(r'$x$')
ax4.set_ylabel(r'$y$')
plt.colorbar(im, ax=ax4, label='rel error $(f(x,y) - I[f](x,y)) / f(x,y)$');

plt.tight_layout()

# %% [markdown]
# ## 4D TP spline interpolation

# %%
x1 = np.array([0.1, 0.11, 0.12, 0.15, 0.2, 0.23, 0.24, 0.248, 0.249, 0.25])
x2 = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
x3 = np.array([-1, -0.8, -0.6, -0.4, 0.0, 0.2, 0.4, 0.8, 1.0])
x4 = np.array([-0.8, -0.6, -0.4, 0.0, 0.5, 1.0, 1.5])
X = [x1, x2, x3, x4]

def f(x1,x2,x3,x4):
    return np.sin(x1) * np.arccos(x2) * np.exp(x3) * np.cos(x4)

xx1, xx2, xx3, xx4 = np.meshgrid(x1, x2, x3, x4, indexing='ij')
F = f(xx1, xx2, xx3, xx4)

fI = TPI.TP_Interpolant_ND(X, F=F)

# %% [markdown]
# While the code works for arbitrary dimensions, but it may be difficult to get good accuracy for reasonably sized grids when the number of dimensions is high.

# %%
y = (0.238, -0.97, 0.93, -0.251)
f(*y), fI(y)

# %% [markdown]
# For this example the relative error is already ~ 4%.

# %%
(f(*y) - fI(y)) / f(*y)


