/*
 * Copyright (C) 2017, 2022 Michael Pürrer, Jonathan Blackman.
 *
 *  This file is part of TPI.
 *
 *  TPI is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  TPI is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with TPI.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gsl/gsl_bspline.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>
#import <stdlib.h>

#define TPI_FAIL -1
#define TPI_SUCCESS 0
#define TPI_ERR_NODES_NOT_FINITE -2
#define TPI_ERR_NODES_NOT_INCREASING -3
#define TPI_ERR_SINGULAR -4

#define CHECK_RANGES

/******************************* 1D functions *********************************/

int Interpolation_Setup_1D(
    double *xvec,                       // Input: knots: FIXME: knots are calculate internally, so shouldn't need to do that here
    int nx,                             // Input length of knots array xvec
    gsl_bspline_workspace **bw          // Output: Initialized B-spline workspace
);

int Bspline_basis_1D(
    double *B_array,                    // Output: the evaluated cubic B-splines 
                                        // B_i(x) for the knots defined in bw
    int n,                              // Input: length of Bx4_array
    gsl_bspline_workspace *bw,          // Input: Initialized B-spline workspace
    double x                            // Input: evaluation point
);

int Bspline_basis_3rd_derivative_1D(
    double *D3_B_array,                 // Output: the evaluated 3rd derivative of cubic
                                        // B-splines B_i(x) for the knots defined in bw
    int n,                              // Input: length of Bx4_array
    gsl_bspline_workspace *bw,          // Input: Initialized B-spline workspace
    double x                            // Input: evaluation point
);

int AssembleSplineMatrix_C(
    gsl_vector *xi,                     // Input: nodes xi
    gsl_matrix **phi,                   // Output: the matrix of spline coefficients
    gsl_vector **knots,                 // Output: the vector of knots including the endpoints
                                        // with multiplicity three
    gsl_bspline_workspace **bw          // Output: Bspline workspace
);

int SetupSpline1D(
    double *x,                          // Input: nodes
    double *y,                          // Input: data
    int n,                              // Input: number of data points
    double **c,                         // Output: spline coefficients
    gsl_bspline_workspace **bw          // Output: Bspline workspace
);

double EvaluateSpline1D(
    double *c,                          // Input: spline coefficients output by SetupSpline1D()
    gsl_bspline_workspace *bw,          // Input: Bspline workspace
    double xx                           // Input: evaluation point for spline
);

/******************************* Generic code *********************************/

typedef struct array_t {
    double *vec;
    int n;
} array;

void TP_Interpolation_Setup_ND(
    array *nodes,                       // Input: array of arrys containing the nodes
                                        // for each parameter space dimension
    int n,                              // Input: Dimensionality of parameter space
    gsl_bspline_workspace ***bw_out     // Output: pointer to array of pointers to
                                        // B-spline workspaces
);

int TP_Interpolation_ND(
    double *v,                          // Input: flattened TP spline coefficient array
    int n,                              // Input: length of TP spline coefficient array v
    double* X,                          // Input: parameter space evaluation point of length m
    int m,                              // Input: dimensionality of parameter space
    gsl_bspline_workspace **bw,         // Input: array of pointers to B-spline workspaces
    double *y                           // Output: TP spline evaluated at X
);

int TP_Interpolation_ND_Vector(
    double *v,                          // Input: flattened TP spline coefficient array with
                                        // p contiguous components per grid coefficient
    int n,                              // Input: length of TP spline coefficient array v
    double* X,                          // Input: parameter space evaluation point of length m
    int m,                              // Input: dimensionality of parameter space
    int p,                              // Input: number of value components per grid point
    gsl_bspline_workspace **bw,         // Input: array of pointers to B-spline workspaces
    double *y                           // Output: TP spline evaluated at X, array of length p
);

int TP_Interpolation_ND_Batch(
    double *v,                          // Input: flattened TP spline coefficient array
    int n,                              // Input: length of TP spline coefficient array v
    double* X,                          // Input: M parameter space points, M x m row-major
    int M,                              // Input: number of evaluation points
    int m,                              // Input: dimensionality of parameter space
    gsl_bspline_workspace **bw,         // Input: array of pointers to B-spline workspaces
    double *y,                          // Output: TP spline evaluated at the M points
    int *fail_point,                    // Output: on TPI_FAIL, index of first out-of-range point
    int *fail_axis                      // Output: on TPI_FAIL, axis of the range violation
);

int Spline_1D_Batch_Sorted(
    double *x,                          // Input: breakpoints (nodes), length n
    int n,                              // Input: number of breakpoints
    double *c0,                         // Input: per-interval cubic coefficients, length n-1
    double *c1,                         //        (value, 1st, 2nd, 3rd order terms in t = xq - x[i])
    double *c2,
    double *c3,
    double *xq,                         // Input: M non-decreasing, in-range query points
    int M,                              // Input: number of query points
    double *y                           // Output: spline evaluated at the M points
);

int Spline_1D_NotAKnot_Factor(
    const double *x,                    // Input: strictly increasing nodes, length n (n >= 4)
    int n,                              // Input: number of nodes
    double *dx,                         // Output: node spacings, length n-1
    double *dl,                         // Output: LU subdiagonal multipliers, length n-1
    double *d,                          // Output: diagonal of U, length n
    double *du,                         // Output: first superdiagonal of U, length n-1
    double *du2,                        // Output: second superdiagonal fill-in, length n-2
    int *piv                            // Output: pivot flags, length n-1 (1 = rows swapped)
);

int Spline_1D_NotAKnot_Refit(
    const double *x,                    // Input: nodes (only the boundary widths are read)
    const double *dx,                   // Input: node spacings from Factor
    int n,                              // Input: number of nodes
    const double *dl,                   // Input: LU factors from Factor
    const double *d,
    const double *du,
    const double *du2,
    const int *piv,
    const double *f,                    // Input: data values at the nodes, length n
    double *work,                       // Work: length n (RHS, then node derivatives)
    double *c0,                         // Output: per-interval cubic coefficients, length n-1
    double *c1,                         //        (value, 1st, 2nd, 3rd order terms in t = xq - x[i])
    double *c2,
    double *c3
);

int Spline_1D_NotAKnot_Construct(       // Factor followed by Refit on the same buffers
    const double *x,
    const double *f,
    int n,
    double *dx,
    double *dl,
    double *d,
    double *du,
    double *du2,
    int *piv,
    double *work,
    double *c0,
    double *c1,
    double *c2,
    double *c3
);

int TP_Interpolation_ND_Vector_Batch(
    double *v,                          // Input: flattened TP spline coefficient array with
                                        // p contiguous components per grid coefficient
    int n,                              // Input: length of TP spline coefficient array v
    double* X,                          // Input: M parameter space points, M x m row-major
    int M,                              // Input: number of evaluation points
    int m,                              // Input: dimensionality of parameter space
    int p,                              // Input: number of value components per grid point
    gsl_bspline_workspace **bw,         // Input: array of pointers to B-spline workspaces
    double *y,                          // Output: TP spline evaluated at the M points, M x p row-major
    int *fail_point,                    // Output: on TPI_FAIL, index of first out-of-range point
    int *fail_axis                      // Output: on TPI_FAIL, axis of the range violation
);