/*
 * dykstra.c - plain Dykstra in dependency-free C, the baseline the accelerated
 * solver is timed against.
 *
 * Usage: dykstra.h documents the one entry point.
 * Build: Windows/MSVC   powershell -ExecutionPolicy Bypass -File build.ps1
 *        gcc/clang      make
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "dykstra.h"

DYK_API int dykstra_solve(int n_planes, int dim,
                          const double *A, const double *b, const double *z,
                          int max_cycles, double *x_out, double *duals_out,
                          dykstra_info *info)
{
    if (n_planes < 0 || dim < 1 || max_cycles < 1 || !z || !x_out
        || (n_planes > 0 && (!A || !b)))
        return DYK_ERR_ARGS;

    const size_t n = (size_t)n_planes, p = (size_t)dim;

    if (n_planes == 0) {
        memcpy(x_out, z, p * sizeof(double));
        if (info) { info->status = DYK_SETTLED; info->cycles = 0; }
        return DYK_SETTLED;
    }

    /* A size whose byte count could wrap size_t is refused before the count
     * is formed. */
    if (3.0 * (double)n * (double)p + 2.0 * ((double)n + (double)p)
        > (double)(SIZE_MAX / sizeof(double)) / 2.0)
        return DYK_ERR_ALLOC;

    const size_t nd = 3 * n * p + 2 * n + 2 * p;
    double *arena = (double *)malloc(nd * sizeof(double));
    if (!arena)
        return DYK_ERR_ALLOC;
    double *units = arena;
    double *e      = units + n * p;
    double *e_prev = e + n * p;
    double *bvec   = e_prev + n * p;
    double *rnorm  = bvec + n;
    double *x      = rnorm + n;
    double *x_prev = x + p;

    for (size_t m = 0; m < n; ++m) {
        const double *row = A + m * p;
        double norm = 0.0;
        for (size_t k = 0; k < p; ++k) norm += row[k] * row[k];
        norm = sqrt(norm);
        if (norm == 0.0) { free(arena); return DYK_ERR_ZERO_NORMAL; }
        rnorm[m] = norm;
        double *um = units + m * p;
        for (size_t k = 0; k < p; ++k) um[k] = row[k] / norm;
        bvec[m] = b[m] / norm;
    }

    memcpy(x, z, p * sizeof(double));
    memset(e, 0, n * p * sizeof(double));

    int status = DYK_MAXCYCLES;
    int cycles = 0;
    for (int i = 0; i < max_cycles; ++i) {
        memcpy(x_prev, x, p * sizeof(double));
        memcpy(e_prev, e, n * p * sizeof(double));

        for (size_t m = 0; m < n; ++m) {
            const double *a = units + m * p;
            double *em = e + m * p;
            double dp = 0.0;
            for (size_t k = 0; k < p; ++k) {
                x[k] += em[k];
                dp += x[k] * a[k];
            }
            const double viol = dp - bvec[m];
            if (viol <= 0.0) {
                for (size_t k = 0; k < p; ++k) em[k] = 0.0;
            } else {
                for (size_t k = 0; k < p; ++k) {
                    x[k] -= viol * a[k];
                    em[k] = viol * a[k];
                }
            }
        }
        cycles = i + 1;

        /* A cycle that leaves the point and every correction unchanged bit for
         * bit will be repeated exactly by every later cycle, so the iterate at
         * any larger budget is this one. */
        if (memcmp(x, x_prev, p * sizeof(double)) == 0
            && memcmp(e, e_prev, n * p * sizeof(double)) == 0) {
            status = DYK_SETTLED;
            break;
        }
    }

    memcpy(x_out, x, p * sizeof(double));
    if (duals_out) {
        for (size_t m = 0; m < n; ++m) {
            double d = 0.0;
            const double *a = units + m * p, *em = e + m * p;
            for (size_t k = 0; k < p; ++k) d += em[k] * a[k];
            duals_out[m] = d / rnorm[m];
        }
    }
    if (info) { info->status = status; info->cycles = cycles; }
    free(arena);
    return status;
}
