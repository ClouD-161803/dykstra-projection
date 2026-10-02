/*
 * accelerated.c - the accelerated Dykstra solver in dependency-free C.
 *
 * It is the deflated_modal preset of lti_solver.LTISolver (LTIVer5Solver) and
 * follows lti_numerics.py piece for piece: between changes of the active set
 * Dykstra's iteration is linear, so an episode of constant activity is
 * evaluated in closed form, jumped where an envelope certifies it switch-free,
 * fast-forwarded when frozen, and settled only on a point that passes the KKT
 * test.  Where it departs from the Python the comment at that place says why;
 * the departures are a reduced-coordinate scan in place of the
 * eigen-decomposition, an estimated condition number, and an incrementally
 * factored Lawson-Hanson.  No history is recorded.
 *
 * Usage: accelerated.h documents the one entry point.
 * Build: Windows/MSVC   powershell -ExecutionPolicy Bypass -File build.ps1
 *        gcc/clang      make
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>

#include "accelerated.h"

#define ROUNDING_UNITS     64.0
#define COND_LIMIT         1e8
#define RHO_CAP            (1.0 - 1e-12)
#define HORIZON_MAX        1e7
#define SKIP_MARGIN        1
#define SPAN_TOL           1e-12
#define LS_RANK_MARGIN     1e3
#define NN_DEPENDENT_QUAD  1e-5
#define NN_DEPENDENT_LS    1e-11
#define KKT_CANDIDATE_TOL  1e-7
#define KKT_EXHAUSTIVE     6
#define KKT_ACCURACY       1e-9

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

/* Rounding allowed, in units of eps times the magnitudes that cancel, before a
 * quantity that is zero in exact arithmetic counts as nonzero.  Every decision
 * below is taken against such a floor and never against an absolute constant,
 * so neither the problem's scale nor a coordinate no constraint touches can
 * change it. */
static double rounding_floor(double magnitude)
{
    return ROUNDING_UNITS * DBL_EPSILON * magnitude;
}

static double dot(const double *a, const double *b, int p)
{
    double s = 0.0;
    for (int k = 0; k < p; ++k) s += a[k] * b[k];
    return s;
}

static double absdot(const double *a, const double *b, int p)
{
    double s = 0.0;
    for (int k = 0; k < p; ++k) s += fabs(a[k]) * fabs(b[k]);
    return s;
}

static double nrm2(const double *a, int p)
{
    return sqrt(dot(a, a, p));
}

static void matvec(const double *A, const double *x, double *y, int r, int c)
{
    for (int i = 0; i < r; ++i)
        y[i] = dot(A + (size_t)i * c, x, c);
}

static void matmul(const double *A, const double *B, double *C, int p)
{
    for (int i = 0; i < p; ++i) {
        double *Ci = C + (size_t)i * p;
        for (int k = 0; k < p; ++k) Ci[k] = 0.0;
        for (int j = 0; j < p; ++j) {
            const double aij = A[(size_t)i * p + j];
            const double *Bj = B + (size_t)j * p;
            for (int k = 0; k < p; ++k) Ci[k] += aij * Bj[k];
        }
    }
}

static int lu_factor(double *A, int p, int *ipiv)
{
    for (int k = 0; k < p; ++k) {
        int piv = k;
        double amax = fabs(A[(size_t)k * p + k]);
        for (int i = k + 1; i < p; ++i) {
            double v = fabs(A[(size_t)i * p + k]);
            if (v > amax) { amax = v; piv = i; }
        }
        ipiv[k] = piv;
        if (amax == 0.0) return 1;
        if (piv != k)
            for (int j = 0; j < p; ++j) {
                double t = A[(size_t)k * p + j];
                A[(size_t)k * p + j] = A[(size_t)piv * p + j];
                A[(size_t)piv * p + j] = t;
            }
        const double inv_pivot = 1.0 / A[(size_t)k * p + k];
        for (int i = k + 1; i < p; ++i) {
            double m = A[(size_t)i * p + k] * inv_pivot;
            A[(size_t)i * p + k] = m;
            for (int j = k + 1; j < p; ++j)
                A[(size_t)i * p + j] -= m * A[(size_t)k * p + j];
        }
    }
    return 0;
}

static void lu_solve(const double *LU, const int *ipiv, int p, double *b)
{
    for (int k = 0; k < p; ++k)
        if (ipiv[k] != k) { double t = b[k]; b[k] = b[ipiv[k]]; b[ipiv[k]] = t; }
    for (int i = 1; i < p; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= LU[(size_t)i * p + j] * b[j];
        b[i] = s;
    }
    for (int i = p - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < p; ++j) s -= LU[(size_t)i * p + j] * b[j];
        b[i] = s / LU[(size_t)i * p + i];
    }
}

static void lu_solve_t(const double *LU, const int *ipiv, int p, double *b)
{
    for (int i = 0; i < p; ++i) {
        double s = b[i];
        for (int j = 0; j < i; ++j) s -= LU[(size_t)j * p + i] * b[j];
        b[i] = s / LU[(size_t)i * p + i];
    }
    for (int i = p - 2; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < p; ++j) s -= LU[(size_t)j * p + i] * b[j];
        b[i] = s;
    }
    for (int k = p - 1; k >= 0; --k)
        if (ipiv[k] != k) { double t = b[k]; b[k] = b[ipiv[k]]; b[ipiv[k]] = t; }
}

static void seed_vector(double *v, int p, unsigned int salt)
{
    unsigned int h = 2463534242u + salt * 2654435761u;
    for (int k = 0; k < p; ++k) {
        h ^= h << 13; h ^= h >> 17; h ^= h << 5;
        v[k] = 0.5 + (double)(h & 0xffffffu) / 33554432.0;
    }
}

static double frobenius_norm(const double *A, int p)
{
    double s = 0.0;
    for (size_t k = 0; k < (size_t)p * p; ++k) s += A[k] * A[k];
    return sqrt(s);
}

/* Largest singular value by power iteration.  It can stop short of the true
 * value when singular values cluster, so a caller that needs an upper bound
 * takes the Frobenius norm whenever *converged is left unset. */
static double spectral_norm(const double *A, int p, double *v, double *u,
                            int *converged)
{
    int reseeds = 0;
    *converged = 0;
    seed_vector(v, p, 0);
    double nv = nrm2(v, p);
    for (int k = 0; k < p; ++k) v[k] /= nv;

    double sig = 0.0;
    for (int it = 0; it < 500; ++it) {
        matvec(A, v, u, p, p);
        double nu = nrm2(u, p);
        if (!isfinite(nu)) return INFINITY;
        double new_sig;
        if (nu == 0.0) {
            new_sig = 0.0;
        } else {
            for (int i = 0; i < p; ++i)
                v[i] = 0.0;
            for (int j = 0; j < p; ++j) {
                const double uj = u[j];
                const double *Aj = A + (size_t)j * p;
                for (int i = 0; i < p; ++i) v[i] += Aj[i] * uj;
            }
            nv = nrm2(v, p);
            new_sig = nv / nu;
        }
        if (nu == 0.0 || nv == 0.0) {

            if (++reseeds > 2) return (sig > nu ? sig : nu);
            seed_vector(v, p, (unsigned int)reseeds);
            nv = nrm2(v, p);
            for (int k = 0; k < p; ++k) v[k] /= nv;
            continue;
        }
        for (int i = 0; i < p; ++i) v[i] /= nv;
        if (fabs(new_sig - sig) <= 1e-15 * (new_sig > 1.0 ? new_sig : 1.0) && it > 2) {
            *converged = 1;
            return new_sig;
        }
        sig = new_sig;
    }
    return sig;
}

/* 2-norm condition number of a factored matrix: inverse power iteration through
 * the LU gives 1 / sigma_min.  An estimate where the Python takes the exact
 * value; it only chooses between the closed form and the deflated or stepped
 * episode, all of which trace the same path. */
static double cond2_estimate(const double *LU, const int *ipiv, int p,
                             double smax, double *v, double *u)
{
    seed_vector(v, p, 7u);
    double nv = nrm2(v, p);
    for (int k = 0; k < p; ++k) v[k] /= nv;

    double inv_smin = 0.0;
    for (int it = 0; it < 200; ++it) {
        memcpy(u, v, (size_t)p * sizeof(double));
        lu_solve_t(LU, ipiv, p, u);
        double nu = nrm2(u, p);
        if (!isfinite(nu)) return INFINITY;
        lu_solve(LU, ipiv, p, u);
        nv = nrm2(u, p);
        if (!isfinite(nv)) return INFINITY;
        if (nv == 0.0) return 0.0;
        double est = sqrt(nv);
        for (int k = 0; k < p; ++k) v[k] = u[k] / nv;
        if (fabs(est - inv_smin) <= 1e-10 * est && it > 2) { inv_smin = est; break; }
        inv_smin = est;
    }
    return smax * inv_smin;
}

/* One-sided Jacobi: rotates the columns of M until they are orthogonal, leaving
 * M = U diag(sig) and V accumulated.  Slow, but accurate for the small, nearly
 * rank-deficient systems it is kept for. */
static void jacobi_svd(double *M, int r, int c, double *V, double *sig)
{
    for (int i = 0; i < c; ++i)
        for (int j = 0; j < c; ++j) V[(size_t)i * c + j] = (i == j) ? 1.0 : 0.0;

    for (int sweep = 0; sweep < 60; ++sweep) {
        int rotated = 0;
        for (int i = 0; i < c - 1; ++i)
            for (int j = i + 1; j < c; ++j) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int k = 0; k < r; ++k) {
                    const double mi = M[(size_t)k * c + i], mj = M[(size_t)k * c + j];
                    alpha += mi * mi; beta += mj * mj; gamma += mi * mj;
                }
                if (gamma == 0.0 || fabs(gamma) <= DBL_EPSILON * sqrt(alpha * beta))
                    continue;
                rotated = 1;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = copysign(1.0, zeta) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                const double cs = 1.0 / sqrt(1.0 + t * t), sn = cs * t;
                for (int k = 0; k < r; ++k) {
                    const double mi = M[(size_t)k * c + i], mj = M[(size_t)k * c + j];
                    M[(size_t)k * c + i] = cs * mi - sn * mj;
                    M[(size_t)k * c + j] = sn * mi + cs * mj;
                }
                for (int k = 0; k < c; ++k) {
                    const double vi = V[(size_t)k * c + i], vj = V[(size_t)k * c + j];
                    V[(size_t)k * c + i] = cs * vi - sn * vj;
                    V[(size_t)k * c + j] = sn * vi + cs * vj;
                }
            }
        if (!rotated) break;
    }
    for (int j = 0; j < c; ++j) {
        double s = 0.0;
        for (int k = 0; k < r; ++k) s += M[(size_t)k * c + j] * M[(size_t)k * c + j];
        sig[j] = sqrt(s);
    }
}

/* Everything a solve needs.  d holds one scalar auxiliary per half-space, since
 * Dykstra's correction for a half-space is always a multiple of its unit
 * normal.  The s-prefixed pointers select the system an episode is scanned in:
 * the full space for a regular episode, the span of the active normals for a
 * deflated one. */
typedef struct {
    int n, p, max_iter;

    double *units, *bvec, *rnorm;
    const double *z;

    double *x, *d;

    double *Am, *Bm, *R, *Rs, *svec, *ss;

    double *Q;
    int kb, span_ready;

    double *IA, *LU, *inv, *xinf, *RIA, *RIAs;
    double *G, *Gf, *beta, *betaf, *rowRIA, *rowR;
    double rho;

    int q, deflated;
    const double *sT, *sWy, *sWg, *sWys, *sWgs;
    double *T, *Wy, *Wg, *Wys, *Wgs;

    double *u0, *u1, *us, *usm1, *dtmp, *dend, *delta, *dfloor;
    double *pw1, *pw2, *pw3, *M1, *M2, *M3;

    double *lsA, *lsM, *lsV, *lsrhs, *lstau, *lsy, *lssig, *lsz;
    int *lsperm;

    double *fQ, *fR, *fcoef, *fy;
    int *fcol, *fpos, fn;

    double *gram, *qc, *nnw, *nnstep, *nnsol, *nnv, *nnmag, *nnprev;
    int *nnsup;
    unsigned char *nnpass, *nnpprev;

    double *kx, *kmag, *kstep, *kcorr, *klam, *kapex, *kdual, *xcert, *mult;
    int *krows, *ksup, *ktight, *ksub;

    int *ipiv;
    unsigned char *active;
    accelerated_info info;
} accelerated_ws;

enum { EP_SWITCH, EP_SETTLED, EP_BUDGET };

static double make_reflector(double *M, int rr, int cc, int j, double norm)
{
    const double alpha = M[(size_t)j * cc + j];
    const double beta = -copysign(norm, alpha);
    const double scale = 1.0 / (alpha - beta);
    for (int i = j + 1; i < rr; ++i) M[(size_t)i * cc + j] *= scale;
    M[(size_t)j * cc + j] = beta;
    return (beta - alpha) / beta;
}

static void reflect_columns(double *M, int rr, int cc, int j, double tau)
{
    for (int k = j + 1; k < cc; ++k) {
        double s = M[(size_t)j * cc + k];
        for (int i = j + 1; i < rr; ++i)
            s += M[(size_t)i * cc + j] * M[(size_t)i * cc + k];
        s *= tau;
        M[(size_t)j * cc + k] -= s;
        for (int i = j + 1; i < rr; ++i)
            M[(size_t)i * cc + k] -= s * M[(size_t)i * cc + j];
    }
}

static void reflect(const double *M, int rr, int cc, int j, double tau, double *y)
{
    double s = y[j];
    for (int i = j + 1; i < rr; ++i) s += M[(size_t)i * cc + j] * y[i];
    s *= tau;
    y[j] -= s;
    for (int i = j + 1; i < rr; ++i) y[i] -= s * M[(size_t)i * cc + j];
}

/* Minimum-norm least squares, as numpy.linalg.lstsq returns it.  Pivoted
 * Householder QR settles the full-rank case.  When the trailing pivots fall
 * clearly below numpy's cut, eps * max(r, c), the rank is read off the QR and
 * the minimum-norm solution comes from a second QR of the leading rows.  Only a
 * pivot between the cut and LS_RANK_MARGIN times it leaves the rank to the
 * singular values.  A wide system is solved through its transpose. */
static void lstsq(accelerated_ws *w, const double *A, int r, int c,
                  const double *b, double *x)
{
    const int tall = (r >= c);
    const int rr = tall ? r : c, cc = tall ? c : r;
    double *M = w->lsM, *tau = w->lstau, *y = w->lsy;
    int *perm = w->lsperm;

    for (int k = 0; k < c; ++k) x[k] = 0.0;
    if (cc == 0) return;

    for (int i = 0; i < rr; ++i)
        for (int j = 0; j < cc; ++j)
            M[(size_t)i * cc + j] = tall ? A[(size_t)i * c + j] : A[(size_t)j * c + i];
    for (int j = 0; j < cc; ++j) perm[j] = j;

    const double cut = DBL_EPSILON * (double)(r > c ? r : c);
    int rank = cc, clean = 1;
    double lead = 0.0;
    for (int j = 0; j < cc; ++j) {
        int best = j;
        double bestn = -1.0;
        for (int k = j; k < cc; ++k) {
            double s = 0.0;
            for (int i = j; i < rr; ++i) s += M[(size_t)i * cc + k] * M[(size_t)i * cc + k];
            if (s > bestn) { bestn = s; best = k; }
        }
        if (best != j) {
            for (int i = 0; i < rr; ++i) {
                double t = M[(size_t)i * cc + j];
                M[(size_t)i * cc + j] = M[(size_t)i * cc + best];
                M[(size_t)i * cc + best] = t;
            }
            int t = perm[j]; perm[j] = perm[best]; perm[best] = t;
        }
        const double norm = sqrt(bestn);
        if (j == 0) lead = norm;
        if (!(norm > LS_RANK_MARGIN * cut * lead)) {
            rank = j;
            clean = (norm <= cut * lead);
            break;
        }
        tau[j] = make_reflector(M, rr, cc, j, norm);
        reflect_columns(M, rr, cc, j, tau[j]);
    }

    if (rank == cc && tall) {
        memcpy(y, b, (size_t)rr * sizeof(double));
        for (int j = 0; j < cc; ++j) reflect(M, rr, cc, j, tau[j], y);
        for (int j = cc - 1; j >= 0; --j) {
            double s = y[j];
            for (int k = j + 1; k < cc; ++k) s -= M[(size_t)j * cc + k] * y[k];
            y[j] = s / M[(size_t)j * cc + j];
        }
        for (int j = 0; j < cc; ++j) x[perm[j]] = y[j];
        return;
    }
    if (rank == cc) {
        for (int j = 0; j < cc; ++j) {
            double s = b[perm[j]];
            for (int k = 0; k < j; ++k) s -= M[(size_t)k * cc + j] * y[k];
            y[j] = s / M[(size_t)j * cc + j];
        }
        for (int i = cc; i < rr; ++i) y[i] = 0.0;
        for (int j = cc - 1; j >= 0; --j) reflect(M, rr, cc, j, tau[j], y);
        memcpy(x, y, (size_t)rr * sizeof(double));
        return;
    }
    if (clean && rank == 0) return;

    if (clean) {
        double *T = w->lsV, *tau2 = w->lssig, *v = w->lsz;
        int sound = 1;
        for (int i = 0; i < cc; ++i)
            for (int j = 0; j < rank; ++j)
                T[(size_t)i * rank + j] = (i >= j) ? M[(size_t)j * cc + i] : 0.0;
        for (int j = 0; j < rank; ++j) {
            double s = 0.0;
            for (int i = j; i < cc; ++i) s += T[(size_t)i * rank + j] * T[(size_t)i * rank + j];
            if (!(s > 0.0)) { sound = 0; break; }
            tau2[j] = make_reflector(T, cc, rank, j, sqrt(s));
            reflect_columns(T, cc, rank, j, tau2[j]);
        }
        if (sound && tall) {
            memcpy(y, b, (size_t)rr * sizeof(double));
            for (int j = 0; j < rank; ++j) reflect(M, rr, cc, j, tau[j], y);
            for (int j = 0; j < rank; ++j) {
                double s = y[j];
                for (int k = 0; k < j; ++k) s -= T[(size_t)k * rank + j] * v[k];
                v[j] = s / T[(size_t)j * rank + j];
            }
            for (int i = rank; i < cc; ++i) v[i] = 0.0;
            for (int j = rank - 1; j >= 0; --j) reflect(T, cc, rank, j, tau2[j], v);
            for (int j = 0; j < cc; ++j) x[perm[j]] = v[j];
            return;
        }
        if (sound) {
            for (int j = 0; j < cc; ++j) v[j] = b[perm[j]];
            for (int j = 0; j < rank; ++j) reflect(T, cc, rank, j, tau2[j], v);
            for (int j = rank - 1; j >= 0; --j) {
                double s = v[j];
                for (int k = j + 1; k < rank; ++k) s -= T[(size_t)j * rank + k] * v[k];
                v[j] = s / T[(size_t)j * rank + j];
            }
            for (int i = 0; i < rr; ++i) y[i] = (i < rank) ? v[i] : 0.0;
            for (int j = rank - 1; j >= 0; --j) reflect(M, rr, cc, j, tau[j], y);
            memcpy(x, y, (size_t)rr * sizeof(double));
            return;
        }
    }

    double *V = w->lsV, *sig = w->lssig;
    for (int i = 0; i < rr; ++i)
        for (int j = 0; j < cc; ++j)
            M[(size_t)i * cc + j] = tall ? A[(size_t)i * c + j] : A[(size_t)j * c + i];
    jacobi_svd(M, rr, cc, V, sig);
    double smax = 0.0;
    for (int j = 0; j < cc; ++j) if (sig[j] > smax) smax = sig[j];
    for (int j = 0; j < cc; ++j) {
        if (!(sig[j] > cut * smax)) continue;
        double coef = 0.0;
        if (tall) {
            for (int i = 0; i < rr; ++i) coef += M[(size_t)i * cc + j] * b[i];
            coef /= sig[j] * sig[j];
            for (int k = 0; k < cc; ++k) x[k] += coef * V[(size_t)k * cc + j];
        } else {
            for (int k = 0; k < cc; ++k) coef += V[(size_t)k * cc + j] * b[k];
            coef /= sig[j] * sig[j];
            for (int i = 0; i < rr; ++i) x[i] += coef * M[(size_t)i * cc + j];
        }
    }
}

/* Minus the gradient of the objective, less the rounding of the terms summed
 * into each entry: a gradient that is zero in exact arithmetic must not bring a
 * row into the support, and a row of zeros, a coordinate no normal touches,
 * must not raise another entry's threshold.  Rows already in the support are
 * skipped, since only the others can enter. */
static void nn_gradient(accelerated_ws *w, int quad, int k, const int *idx,
                        const double *d, const double *lam, double *g)
{
    const int p = w->p;
    const unsigned char *passive = w->nnpass;
    if (quad) {
        int *used = w->nnsup, nu = 0;
        for (int j = 0; j < k; ++j) if (lam[j] != 0.0) used[nu++] = j;
        for (int i = 0; i < k; ++i) {
            if (passive[i]) continue;
            const double *Hi = w->gram + (size_t)i * k;
            double s = w->qc[i], mag = fabs(w->qc[i]);
            for (int u = 0; u < nu; ++u) {
                s -= Hi[used[u]] * lam[used[u]];
                mag += fabs(Hi[used[u]]) * lam[used[u]];
            }
            g[i] = s - rounding_floor(mag);
        }
        return;
    }
    double *v = w->nnv, *mag = w->nnmag;
    for (int j = 0; j < p; ++j) { v[j] = d[j]; mag[j] = fabs(d[j]); }
    for (int i = 0; i < k; ++i) {
        if (lam[i] == 0.0) continue;
        const double *a = w->units + (size_t)idx[i] * p;
        for (int j = 0; j < p; ++j) {
            v[j] -= lam[i] * a[j];
            mag[j] += lam[i] * fabs(a[j]);
        }
    }
    for (int i = 0; i < k; ++i) {
        if (passive[i]) continue;
        const double *a = w->units + (size_t)idx[i] * p;
        g[i] = dot(a, v, p) - rounding_floor(absdot(a, mag, p));
    }
}

static void nn_support_solve(accelerated_ws *w, int quad, int k, const int *idx,
                             const double *d, int ns, double *sol)
{
    const int p = w->p;
    const int *sup = w->nnsup;
    if (quad) {
        for (int i = 0; i < ns; ++i) {
            for (int j = 0; j < ns; ++j)
                w->lsA[(size_t)i * ns + j] = w->gram[(size_t)sup[i] * k + sup[j]];
            w->lsrhs[i] = w->qc[sup[i]];
        }
        lstsq(w, w->lsA, ns, ns, w->lsrhs, sol);
        return;
    }
    for (int j = 0; j < ns; ++j) {
        const double *a = w->units + (size_t)idx[sup[j]] * p;
        for (int i = 0; i < p; ++i) w->lsA[(size_t)i * ns + j] = a[i];
    }
    lstsq(w, w->lsA, p, ns, d, sol);
}

/* QR factorisation of the support's normals, updated as rows enter and leave
 * instead of being recomputed: the from-scratch solves inside Lawson-Hanson
 * were most of the solver's time at the larger sizes.  A normal within tol of
 * the span of the others is refused; the caller then falls back on lstsq, which
 * treats rank deficiency as the Python does. */
static int factor_append(accelerated_ws *w, int i, const int *idx, double tol)
{
    const int p = w->p, f = w->fn;
    if (f >= p) return 0;
    double *v = w->fQ + (size_t)f * p, *coef = w->fcoef;
    memcpy(v, w->units + (size_t)idx[i] * p, (size_t)p * sizeof(double));
    for (int j = 0; j < f; ++j) coef[j] = 0.0;
    for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < f; ++j) {
            const double *qj = w->fQ + (size_t)j * p;
            const double h = dot(qj, v, p);
            for (int k = 0; k < p; ++k) v[k] -= h * qj[k];
            coef[j] += h;
        }
    const double rnorm = nrm2(v, p);
    if (!(rnorm > tol)) return 0;
    for (int k = 0; k < p; ++k) v[k] /= rnorm;
    for (int j = 0; j < f; ++j) w->fR[(size_t)j * p + f] = coef[j];
    w->fR[(size_t)f * p + f] = rnorm;
    w->fcol[f] = i;
    w->fpos[i] = f;
    w->fn = f + 1;
    return 1;
}

/* Removing a column leaves R upper Hessenberg; Givens rotations restore the
 * triangle and turn Q with it. */
static void factor_delete(accelerated_ws *w, int j)
{
    const int p = w->p, f = w->fn;
    double *R = w->fR;
    w->fpos[w->fcol[j]] = -1;
    for (int c = j + 1; c < f; ++c) {
        for (int i = 0; i <= c; ++i) R[(size_t)i * p + c - 1] = R[(size_t)i * p + c];
        w->fcol[c - 1] = w->fcol[c];
        w->fpos[w->fcol[c - 1]] = c - 1;
    }
    for (int c = j; c < f - 1; ++c) {
        const double a = R[(size_t)c * p + c], b = R[(size_t)(c + 1) * p + c];
        const double r = hypot(a, b);
        if (r == 0.0) continue;
        const double cs = a / r, sn = b / r;
        for (int col = c; col < f - 1; ++col) {
            const double t1 = R[(size_t)c * p + col], t2 = R[(size_t)(c + 1) * p + col];
            R[(size_t)c * p + col] = cs * t1 + sn * t2;
            R[(size_t)(c + 1) * p + col] = -sn * t1 + cs * t2;
        }
        double *q1 = w->fQ + (size_t)c * p, *q2 = w->fQ + (size_t)(c + 1) * p;
        for (int k = 0; k < p; ++k) {
            const double u1 = q1[k], u2 = q2[k];
            q1[k] = cs * u1 + sn * u2;
            q2[k] = -sn * u1 + cs * u2;
        }
    }
    w->fn = f - 1;
}

static int factor_sync(accelerated_ws *w, int k, const int *idx, double tol)
{
    for (int j = w->fn - 1; j >= 0; --j)
        if (!w->nnpass[w->fcol[j]]) factor_delete(w, j);
    int complete = 1;
    for (int i = 0; i < k; ++i)
        if (w->nnpass[i] && w->fpos[i] < 0 && !factor_append(w, i, idx, tol))
            complete = 0;
    return complete;
}

static void factor_solve(accelerated_ws *w, int quad, const double *d)
{
    const int p = w->p, f = w->fn;
    const double *R = w->fR;
    double *y = w->fy;
    for (int j = 0; j < f; ++j) {
        if (quad) {
            double s = w->qc[w->fcol[j]];
            for (int k = 0; k < j; ++k) s -= R[(size_t)k * p + j] * y[k];
            y[j] = s / R[(size_t)j * p + j];
        } else {
            y[j] = dot(w->fQ + (size_t)j * p, d, p);
        }
    }
    for (int j = f - 1; j >= 0; --j) {
        double s = y[j];
        for (int k = j + 1; k < f; ++k) s -= R[(size_t)j * p + k] * y[k];
        y[j] = s / R[(size_t)j * p + j];
    }
}

/* Lawson and Hanson's active-set method over lam >= 0, for the least-squares
 * problem (quad = 0) and for the quadratic whose matrix is the Gram matrix of
 * the candidate normals (quad = 1).  Both iteration caps only guard against
 * rounding cycles.  The test that closes the outer loop stops as soon as an
 * iteration leaves the multipliers and the support as it found them: the next
 * would pick the same row and repeat it exactly, which the Python does until
 * its cap, to the same result. */
static void nn_solve(accelerated_ws *w, int quad, int k, const int *idx,
                     const double *d, double *lam)
{
    const double tol = quad ? NN_DEPENDENT_QUAD : NN_DEPENDENT_LS;
    double *g = w->nnw, *step = w->nnstep, *sol = w->nnsol, *before = w->nnprev;
    unsigned char *passive = w->nnpass, *passive_before = w->nnpprev;
    int *sup = w->nnsup;
    const int cap = 3 * k + 10;

    for (int i = 0; i < k; ++i) { lam[i] = 0.0; passive[i] = 0; w->fpos[i] = -1; }
    w->fn = 0;
    for (int it = 0; it < cap; ++it) {
        nn_gradient(w, quad, k, idx, d, lam, g);
        int enter = -1;
        for (int i = 0; i < k; ++i)
            if (!passive[i] && (enter < 0 || g[i] > g[enter])) enter = i;
        if (enter < 0 || !(g[enter] > 0.0)) break;
        memcpy(before, lam, (size_t)k * sizeof(double));
        memcpy(passive_before, passive, (size_t)k);
        passive[enter] = 1;

        for (int it2 = 0; it2 < cap; ++it2) {
            int ns = 0;
            for (int i = 0; i < k; ++i) if (passive[i]) sup[ns++] = i;
            if (factor_sync(w, k, idx, tol)) {
                factor_solve(w, quad, d);
                for (int i = 0; i < ns; ++i) sol[i] = w->fy[w->fpos[sup[i]]];
            } else {
                nn_support_solve(w, quad, k, idx, d, ns, sol);
            }
            for (int i = 0; i < k; ++i) step[i] = 0.0;
            int positive = 1;
            for (int i = 0; i < ns; ++i) {
                step[sup[i]] = sol[i];
                if (!(sol[i] > 0.0)) positive = 0;
            }
            if (positive) {
                memcpy(lam, step, (size_t)k * sizeof(double));
                break;
            }
            int first = -1;
            double ratio = INFINITY;
            for (int i = 0; i < ns; ++i) {
                const int m = sup[i];
                if (sol[i] > 0.0) continue;
                const double denom = lam[m] - step[m];
                const double r = (denom != 0.0) ? lam[m] / denom : 0.0;
                if (r < ratio) { ratio = r; first = m; }
            }
            if (first < 0) break;
            for (int i = 0; i < k; ++i) lam[i] += ratio * (step[i] - lam[i]);
            lam[first] = 0.0;
            for (int i = 0; i < k; ++i) passive[i] = (unsigned char)(passive[i] && lam[i] > 0.0);
        }
        if (memcmp(passive, passive_before, (size_t)k) == 0
            && memcmp(lam, before, (size_t)k * sizeof(double)) == 0)
            break;
    }
}

/* A point reached from z rounds with |z| and the step, not with |x|, which is
 * tiny at an apex through the origin. */
static void slack_magnitudes(const accelerated_ws *w, const double *x, double *mag)
{
    for (int k = 0; k < w->p; ++k)
        mag[k] = fabs(w->z[k]) + fabs(w->z[k] - x[k]);
}

/* KKT test at the projection of z onto the boundaries of a support.  The
 * projection is by least squares, which is backward stable however nearly
 * parallel the rows are.  A coordinate no support row touches keeps z's value
 * exactly: least squares leaves rounding there on the scale of the whole
 * correction, while the allowance there is built from that coordinate alone.
 * The point must be feasible, and the multipliers nonnegative on the tight rows
 * only, both at rounding level.  It is then the projection of z minus the
 * residual, so the residual bounds its error; the multiplier terms enter the
 * allowance only at rounding, because they grow as the tight normals become
 * nearly dependent and letting them loosen the accuracy certifies wrong
 * supports. */
static int kkt_verify(accelerated_ws *w, const int *sup, int ns)
{
    const int n = w->n, p = w->p;
    const double *z = w->z;
    double *x = w->kx, *step = w->kstep;

    memcpy(x, z, (size_t)p * sizeof(double));
    if (ns > 0) {
        for (int i = 0; i < ns; ++i) {
            const double *a = w->units + (size_t)sup[i] * p;
            memcpy(w->lsA + (size_t)i * p, a, (size_t)p * sizeof(double));
            w->lsrhs[i] = dot(a, z, p) - w->bvec[sup[i]];
        }
        lstsq(w, w->lsA, ns, p, w->lsrhs, w->kcorr);
        for (int k = 0; k < p; ++k) {
            int touched = 0;
            for (int i = 0; i < ns && !touched; ++i)
                touched = (w->units[(size_t)sup[i] * p + k] != 0.0);
            if (touched) x[k] -= w->kcorr[k];
        }
    }

    slack_magnitudes(w, x, w->kmag);
    int nt = 0;
    for (int m = 0; m < n; ++m) {
        const double *a = w->units + (size_t)m * p;
        const double slack = dot(a, x, p) - w->bvec[m];
        const double floor_m = rounding_floor(absdot(a, w->kmag, p) + fabs(w->bvec[m]));
        if (slack > floor_m) return 0;
        if (slack >= -floor_m) w->ktight[nt++] = m;
    }

    for (int k = 0; k < p; ++k) step[k] = z[k] - x[k];
    memset(w->mult, 0, (size_t)n * sizeof(double));
    if (nt > 0) {
        nn_solve(w, 0, nt, w->ktight, step, w->klam);
        for (int i = 0; i < nt; ++i) w->mult[w->ktight[i]] = w->klam[i];
    }

    for (int k = 0; k < p; ++k) {
        double residual = step[k], weight = 0.0;
        for (int i = 0; i < nt; ++i) {
            const double a = w->units[(size_t)w->ktight[i] * p + k];
            residual -= a * w->klam[i];
            weight += fabs(a) * w->klam[i];
        }
        const double allowed = KKT_ACCURACY * fabs(step[k])
                             + rounding_floor(fabs(z[k]) + fabs(x[k]) + weight);
        if (!(fabs(residual) <= allowed)) return 0;
    }
    memcpy(w->xcert, x, (size_t)p * sizeof(double));
    return 1;
}

static int same_support(const int *a, int na, const int *b, int nb)
{
    return na == nb && (na == 0 || memcmp(a, b, (size_t)na * sizeof(int)) == 0);
}

/* Candidate rows are the active set and every row near the hint.  Candidate
 * supports, in the order tried: the rows that bind at the projection onto the
 * candidate rows alone, by its dual; by the cone at a common point of their
 * boundaries; the active set; all candidate rows; and, when there are at most
 * KKT_EXHAUSTIVE of them, every subset, since nearly dependent rows leave the
 * choice among them to rounding. */
static int kkt_certificate(accelerated_ws *w, const double *hint)
{
    const int n = w->n, p = w->p;
    const double *z = w->z;
    int *rows = w->krows, *sup = w->ksup;
    int nr = 0, count = 0, len[4];

    slack_magnitudes(w, hint, w->kmag);
    for (int m = 0; m < n; ++m) {
        const double *a = w->units + (size_t)m * p;
        const double slack = dot(a, hint, p) - w->bvec[m];
        const double scale = absdot(a, w->kmag, p) + fabs(w->bvec[m]);
        if (w->active[m] || slack >= -KKT_CANDIDATE_TOL * scale) rows[nr++] = m;
    }

    if (nr > 0) {
        for (int i = 0; i < nr; ++i) {
            const double *ai = w->units + (size_t)rows[i] * p;
            for (int j = 0; j <= i; ++j) {
                const double g = dot(ai, w->units + (size_t)rows[j] * p, p);
                w->gram[(size_t)i * nr + j] = g;
                w->gram[(size_t)j * nr + i] = g;
            }
            w->qc[i] = dot(ai, z, p) - w->bvec[rows[i]];
        }
        nn_solve(w, 1, nr, rows, NULL, w->kdual);
        len[count] = 0;
        for (int i = 0; i < nr; ++i)
            if (w->kdual[i] > 0.0) sup[(size_t)count * n + len[count]++] = rows[i];
        ++count;

        for (int i = 0; i < nr; ++i) {
            memcpy(w->lsA + (size_t)i * p, w->units + (size_t)rows[i] * p,
                   (size_t)p * sizeof(double));
            w->lsrhs[i] = w->bvec[rows[i]];
        }
        lstsq(w, w->lsA, nr, p, w->lsrhs, w->kapex);
        for (int k = 0; k < p; ++k) w->kapex[k] = z[k] - w->kapex[k];
        nn_solve(w, 0, nr, rows, w->kapex, w->kdual);
        len[count] = 0;
        for (int i = 0; i < nr; ++i)
            if (w->kdual[i] > 0.0) sup[(size_t)count * n + len[count]++] = rows[i];
        ++count;
    }
    len[count] = 0;
    for (int m = 0; m < n; ++m)
        if (w->active[m]) sup[(size_t)count * n + len[count]++] = m;
    ++count;
    memcpy(sup + (size_t)count * n, rows, (size_t)nr * sizeof(int));
    len[count++] = nr;

    for (int s = 0; s < count; ++s) {
        int seen = 0;
        for (int e = 0; e < s && !seen; ++e)
            seen = same_support(sup + (size_t)s * n, len[s], sup + (size_t)e * n, len[e]);
        if (!seen && kkt_verify(w, sup + (size_t)s * n, len[s])) return 1;
    }

    if (nr <= KKT_EXHAUSTIVE) {
        int pick[KKT_EXHAUSTIVE], *sub = w->ksub;
        for (int size = 1; size <= nr; ++size) {
            for (int i = 0; i < size; ++i) pick[i] = i;
            for (;;) {
                for (int i = 0; i < size; ++i) sub[i] = rows[pick[i]];
                int seen = 0;
                for (int e = 0; e < count && !seen; ++e)
                    seen = same_support(sub, size, sup + (size_t)e * n, len[e]);
                if (!seen && kkt_verify(w, sub, size)) return 1;
                int i = size - 1;
                while (i >= 0 && pick[i] == nr - size + i) --i;
                if (i < 0) break;
                ++pick[i];
                for (int j = i + 1; j < size; ++j) pick[j] = pick[j - 1] + 1;
            }
        }
    }
    return 0;
}

/* The multipliers are the auxiliaries of the limit, x + sum d_m a_m = z, and
 * their support its active set. */
static void settle_certified(accelerated_ws *w)
{
    memcpy(w->x, w->xcert, (size_t)w->p * sizeof(double));
    memcpy(w->d, w->mult, (size_t)w->n * sizeof(double));
    for (int m = 0; m < w->n; ++m) w->active[m] = (w->d[m] > 0.0);
}

/* An active auxiliary kept within rounding of zero stays nonnegative, as in
 * Dykstra; an inactive one is zero. */
static void commit_duals(accelerated_ws *w, const double *active_dual)
{
    for (int m = 0; m < w->n; ++m)
        w->d[m] = (w->active[m] && active_dual[m] > 0.0) ? active_dual[m] : 0.0;
}

static void dykstra_cycle(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    for (int m = 0; m < n; ++m) {
        const double *a = w->units + (size_t)m * p;
        const double dm = w->d[m];

        double dp = 0.0;
        for (int k = 0; k < p; ++k) {
            w->x[k] += dm * a[k];
            dp += w->x[k] * a[k];
        }
        const double viol = dp - w->bvec[m];
        if (viol > 0.0) {
            for (int k = 0; k < p; ++k) w->x[k] -= viol * a[k];
            w->d[m] = viol;
        } else {
            w->d[m] = 0.0;
        }
        w->active[m] = (w->d[m] > 0.0);
    }
}

/* Am and Bm carry a cycle's start to its end; row m of R and svec gives the
 * increment of auxiliary m over a cycle.  Rs and ss hold the magnitudes R and
 * svec are summed from: for a half-space that repeats the last active one
 * before it R_m cancels to rounding, and the floors must not shrink with it. */
static void build_cycle_map(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    double *P = w->Am, *q = w->Bm;
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) P[(size_t)i * p + j] = (i == j) ? 1.0 : 0.0;
    for (int k = 0; k < p; ++k) q[k] = 0.0;

    double *tmp = w->pw1;
    for (int m = 0; m < n; ++m) {
        const double *a = w->units + (size_t)m * p;
        const double b = w->bvec[m];

        double *Rm = w->R + (size_t)m * p, *Rsm = w->Rs + (size_t)m * p;
        for (int k = 0; k < p; ++k) { Rm[k] = 0.0; Rsm[k] = 0.0; }
        for (int j = 0; j < p; ++j) {
            const double aj = a[j], abs_aj = fabs(a[j]);
            const double *Pj = P + (size_t)j * p;
            for (int k = 0; k < p; ++k) {
                Rm[k] += Pj[k] * aj;
                Rsm[k] += fabs(Pj[k]) * abs_aj;
            }
        }
        w->svec[m] = dot(a, q, p) - b;
        w->ss[m] = absdot(a, q, p) + fabs(b);
        if (w->active[m]) {

            for (int k = 0; k < p; ++k) tmp[k] = 0.0;
            for (int j = 0; j < p; ++j) {
                const double aj = a[j];
                const double *Pj = P + (size_t)j * p;
                for (int k = 0; k < p; ++k) tmp[k] += aj * Pj[k];
            }
            for (int i = 0; i < p; ++i) {
                const double ai = a[i];
                double *Pi = P + (size_t)i * p;
                for (int k = 0; k < p; ++k) Pi[k] -= ai * tmp[k];
            }

            const double aq = dot(a, q, p);
            for (int k = 0; k < p; ++k) q[k] += (b - aq) * a[k];
        }
    }
    w->span_ready = 0;
}

/* Orthonormal basis of the active normals, the only directions a cycle moves,
 * from the singular vectors so that nearly parallel normals are told apart
 * from dependent ones.  Built only when an episode needs it. */
static void ensure_span(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    if (w->span_ready) return;
    w->span_ready = 1;
    w->kb = 0;

    int k = 0;
    for (int m = 0; m < n; ++m) if (w->active[m]) w->nnsup[k++] = m;
    if (k == 0) return;

    double *M = w->lsM, *V = w->lsV, *sig = w->lssig;
    const int wide = (k > p);
    const int rr = wide ? k : p, cc = wide ? p : k;
    for (int i = 0; i < k; ++i) {
        const double *a = w->units + (size_t)w->nnsup[i] * p;
        for (int j = 0; j < p; ++j) {
            if (wide) M[(size_t)i * cc + j] = a[j];
            else      M[(size_t)j * cc + i] = a[j];
        }
    }
    jacobi_svd(M, rr, cc, V, sig);
    double smax = 0.0;
    for (int j = 0; j < cc; ++j) if (sig[j] > smax) smax = sig[j];
    for (int j = 0; j < cc; ++j) {
        if (!(sig[j] > SPAN_TOL * smax)) continue;
        double *qr = w->Q + (size_t)w->kb * p;
        if (wide) for (int i = 0; i < p; ++i) qr[i] = V[(size_t)i * cc + j];
        else      for (int i = 0; i < p; ++i) qr[i] = M[(size_t)i * cc + j] / sig[j];
        ++w->kb;
    }
}

static double factor_resolvent(accelerated_ws *w)
{
    const int p = w->p;
    int conv = 0;
    double smax = spectral_norm(w->IA, p, w->pw1, w->pw2, &conv);
    if (!conv) smax = frobenius_norm(w->IA, p);
    memcpy(w->LU, w->IA, (size_t)p * p * sizeof(double));
    if (lu_factor(w->LU, p, w->ipiv)) return INFINITY;
    const double cond = cond2_estimate(w->LU, w->ipiv, p, smax, w->pw1, w->pw2);
    return isfinite(cond) ? cond : INFINITY;
}

/* Constants of an episode's closed form around its fixed point xinf: the levels
 * G and drifts beta of the auxiliaries and the rounding level of each.  RIA =
 * R (I - A_m)^-1 comes from solves, not an explicit inverse, which loses cond *
 * eps.  A row of R that cancelled to the rounding of the magnitudes it was
 * formed from is noise, and so is its row of RIA: only such a row takes its
 * rounding from those magnitudes through the inverse, since charging every row
 * with the resolvent's condition would hide real drains. */
static void finish_closed_form(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    int have_inv = 0;

    for (int m = 0; m < n; ++m) {
        const double *Rm = w->R + (size_t)m * p, *Rsm = w->Rs + (size_t)m * p;
        double *RIAm = w->RIA + (size_t)m * p, *RIAsm = w->RIAs + (size_t)m * p;
        memcpy(RIAm, Rm, (size_t)p * sizeof(double));
        lu_solve_t(w->LU, w->ipiv, p, RIAm);

        const double r_norm = nrm2(Rm, p);
        const double r_rounding = rounding_floor(nrm2(Rsm, p));
        if (r_norm <= r_rounding) {
            if (!have_inv) {
                double *col = w->pw1;
                for (int j = 0; j < p; ++j) {
                    for (int i = 0; i < p; ++i) col[i] = (i == j) ? 1.0 : 0.0;
                    lu_solve(w->LU, w->ipiv, p, col);
                    for (int i = 0; i < p; ++i) w->inv[(size_t)i * p + j] = col[i];
                }
                have_inv = 1;
            }
            for (int k = 0; k < p; ++k) {
                double sum = 0.0;
                for (int j = 0; j < p; ++j) sum += Rsm[j] * fabs(w->inv[(size_t)j * p + k]);
                RIAsm[k] = sum;
            }
        } else {
            for (int k = 0; k < p; ++k) RIAsm[k] = fabs(RIAm[k]);
        }
        w->rowRIA[m] = nrm2(RIAm, p);
        w->rowR[m] = (r_norm > r_rounding) ? r_norm - r_rounding : 0.0;
    }

    double *z0 = w->pw3;
    for (int k = 0; k < p; ++k) z0[k] = w->x[k] - w->xinf[k];
    for (int m = 0; m < n; ++m) {
        const double *Rm = w->R + (size_t)m * p, *Rsm = w->Rs + (size_t)m * p;
        const double *RIAm = w->RIA + (size_t)m * p, *RIAsm = w->RIAs + (size_t)m * p;
        w->G[m] = w->d[m] + dot(RIAm, z0, p);
        w->Gf[m] = rounding_floor(fabs(w->d[m]) + absdot(RIAsm, z0, p));
        w->beta[m] = dot(Rm, w->xinf, p) + w->svec[m];
        w->betaf[m] = rounding_floor(absdot(Rsm, w->xinf, p) + w->ss[m]);
    }
}

/* The envelope needs an upper bound on the contraction rate, so a converged
 * estimate is inflated slightly and an unconverged one replaced by the
 * Frobenius norm. */
static void setup_regular(accelerated_ws *w)
{
    const int p = w->p;
    memcpy(w->xinf, w->Bm, (size_t)p * sizeof(double));
    lu_solve(w->LU, w->ipiv, p, w->xinf);
    finish_closed_form(w);

    w->deflated = 0;
    w->q = p;
    w->sT = w->Am; w->sWy = w->RIA; w->sWg = w->R;
    w->sWys = w->RIAs; w->sWgs = w->Rs;
    memcpy(w->u0, w->pw3, (size_t)p * sizeof(double));

    int conv = 0;
    double smax = spectral_norm(w->Am, p, w->pw1, w->pw2, &conv);
    smax = conv ? smax * (1.0 + 1e-9) : frobenius_norm(w->Am, p);
    w->rho = (smax < RHO_CAP) ? smax : RHO_CAP;
}

static void span_project(const accelerated_ws *w, const double *v, double *out)
{
    const int p = w->p;
    double *coef = w->lsy;
    for (int r = 0; r < w->kb; ++r) coef[r] = dot(w->Q + (size_t)r * p, v, p);
    for (int k = 0; k < p; ++k) out[k] = 0.0;
    for (int r = 0; r < w->kb; ++r) {
        const double *qr = w->Q + (size_t)r * p;
        for (int k = 0; k < p; ++k) out[k] += coef[r] * qr[k];
    }
}

/* I - A_m is singular exactly on the complement of the span of the active
 * normals, where the state never moves.  Adding the projector onto that
 * complement makes it invertible, and keeping only the span component of the
 * solve stops rounding from moving the kernel component.  The episode is then
 * scanned in the coordinates u = Q^T (x - xinf) of the span, with T = Q^T A_m Q
 * in place of A_m.  The Python diagonalises T and scans its modes in blocks;
 * iterating T gives the same sequence without an eigen-solver, and powers of T,
 * unlike powers of A_m, carry no eigenvalue of one to drift. */
static int setup_deflated(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    ensure_span(w);
    const int kb = w->kb;
    if (kb == 0) return 0;

    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) {
            double s = ((i == j) ? 2.0 : 0.0) - w->Am[(size_t)i * p + j];
            for (int r = 0; r < kb; ++r)
                s -= w->Q[(size_t)r * p + i] * w->Q[(size_t)r * p + j];
            w->IA[(size_t)i * p + j] = s;
        }
    if (!(factor_resolvent(w) < COND_LIMIT)) return 0;

    span_project(w, w->Bm, w->pw1);
    lu_solve(w->LU, w->ipiv, p, w->pw1);
    span_project(w, w->pw1, w->pw2);
    span_project(w, w->x, w->pw1);
    for (int k = 0; k < p; ++k) w->xinf[k] = w->x[k] - w->pw1[k] + w->pw2[k];
    finish_closed_form(w);

    for (int r = 0; r < kb; ++r) {
        const double *qr = w->Q + (size_t)r * p;
        matvec(w->Am, qr, w->pw1, p, p);
        for (int i = 0; i < kb; ++i)
            w->T[(size_t)i * kb + r] = dot(w->Q + (size_t)i * p, w->pw1, p);
        for (int m = 0; m < n; ++m) {
            w->Wy[(size_t)m * p + r] = dot(w->RIA + (size_t)m * p, qr, p);
            w->Wg[(size_t)m * p + r] = dot(w->R + (size_t)m * p, qr, p);
            w->Wys[(size_t)m * p + r] = absdot(w->RIAs + (size_t)m * p, qr, p);
            w->Wgs[(size_t)m * p + r] = absdot(w->Rs + (size_t)m * p, qr, p);
        }
        w->u0[r] = dot(qr, w->pw3, p);
    }

    w->deflated = 1;
    w->q = kb;
    w->sT = w->T; w->sWy = w->Wy; w->sWg = w->Wg;
    w->sWys = w->Wys; w->sWgs = w->Wgs;

    int conv = 0;
    double smax = spectral_norm(w->T, kb, w->pw1, w->pw2, &conv);
    smax = conv ? smax * (1.0 + 1e-9) : frobenius_norm(w->T, kb);
    w->rho = (smax < RHO_CAP) ? smax : RHO_CAP;
    return 1;
}

static void state_from_modes(accelerated_ws *w, const double *u, const double *y)
{
    const int p = w->p;
    memcpy(w->x, w->xinf, (size_t)p * sizeof(double));
    if (w->deflated) {
        for (int r = 0; r < w->q; ++r) {
            const double *qr = w->Q + (size_t)r * p;
            for (int k = 0; k < p; ++k) w->x[k] += u[r] * qr[k];
        }
    } else {
        for (int k = 0; k < p; ++k) w->x[k] += u[k];
    }
    commit_duals(w, y);
}

/* Rounding of an auxiliary extrapolated to cycle t.  A drift within rounding may
 * be rounding of either sign, which extrapolation multiplies by t; a drift
 * beyond it is real and decides on its own. */
static double auxiliary_floor(const accelerated_ws *w, int m, double t)
{
    return w->Gf[m] + (fabs(w->beta[m]) <= w->betaf[m] ? t * w->betaf[m] : 0.0);
}

static void closed_form_auxiliaries(const accelerated_ws *w, double t,
                                    const double *u, double *y)
{
    for (int m = 0; m < w->n; ++m)
        y[m] = w->G[m] + t * w->beta[m] - dot(w->sWy + (size_t)m * w->p, u, w->q);
}

/* Activity changes only on a value beyond rounding: an inactive half-space
 * reactivates on a slack above it, and an active one deactivates on an
 * auxiliary below minus the rounding its extrapolation carries, which is all
 * the idle member of an equality written as two half-spaces ever shows. */
static int activity_holds(const accelerated_ws *w, double t, const double *y,
                          const double *u, const double *u_prev)
{
    const int p = w->p, q = w->q;
    for (int m = 0; m < w->n; ++m) {
        int on;
        if (w->active[m]) {
            const double y_floor = auxiliary_floor(w, m, t)
                + rounding_floor(absdot(w->sWys + (size_t)m * p, u, q));
            on = (y[m] > -y_floor);
        } else {
            const double g = w->beta[m] + dot(w->sWg + (size_t)m * p, u_prev, q);
            const double g_floor = w->betaf[m]
                + rounding_floor(absdot(w->sWgs + (size_t)m * p, u_prev, q));
            on = (g > g_floor);
        }
        if (on != (int)w->active[m]) return 0;
    }
    return 1;
}

/* Every later transient is bounded by the current one.  A drift is zero only up
 * to rounding, since any real drain reaches zero eventually; an inactive slack
 * must stay within the rounding its prediction ignores. */
static int active_set_is_final(const accelerated_ws *w, double t, double unorm)
{
    for (int m = 0; m < w->n; ++m) {
        if (w->active[m]) {
            if (!(w->beta[m] >= -w->betaf[m])) return 0;
            if (!(w->G[m] + t * w->beta[m] - w->rowRIA[m] * unorm
                  > -auxiliary_floor(w, m, t))) return 0;
        } else {
            if (!(w->beta[m] + w->rowR[m] * unorm <= w->betaf[m])) return 0;
        }
    }
    return 1;
}

static double envelope_L(double gamma, double beta, double alpha,
                         double rho, double k)
{
    return gamma + beta * k - alpha * pow(rho, k);
}

/* Horizon of the envelope lower bound L(k) = gamma + beta k - alpha rho^k, by
 * bracketing its descending root and bisecting (L is concave).  Without a
 * negative drift the bound never descends.  Past HORIZON_MAX the last bracket
 * point still has L > 0, so the whole run up to it is certified. */
static double rigorous_horizon(double gamma, double beta, double alpha,
                               double rho)
{
    if (envelope_L(gamma, beta, alpha, rho, 1.0) <= 0.0) return 0.0;
    if (beta >= 0.0) return INFINITY;
    double hi = 2.0;
    while (envelope_L(gamma, beta, alpha, rho, hi) > 0.0) {
        hi *= 2.0;
        if (hi > HORIZON_MAX) return hi / 2.0;
    }
    double lo = hi / 2.0;
    for (int it = 0; it < 200; ++it) {
        if (hi - lo <= 1e-12) break;
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        if (envelope_L(gamma, beta, alpha, rho, mid) > 0.0) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* No jump while an inactive slack could rise above the rounding its prediction
 * ignores.  Otherwise the jump is the smallest horizon over the active rows,
 * each read as closed_form activity reads it: a drift within rounding is no
 * drain, whatever its sign, and the level keeps the drift itself so that the
 * floor, which grows at least as fast, covers it. */
static int skip_estimate(const accelerated_ws *w, int t, double unorm)
{
    for (int j = 0; j < w->n; ++j)
        if (!w->active[j] && w->beta[j] + w->rowR[j] * unorm > w->betaf[j])
            return 0;

    double s = INFINITY;
    for (int m = 0; m < w->n; ++m) {
        if (!w->active[m]) continue;
        const double beta = (fabs(w->beta[m]) <= w->betaf[m]) ? 0.0 : w->beta[m];
        const double gamma = w->G[m] + auxiliary_floor(w, m, (double)t)
                           + (double)t * w->beta[m];
        const double h = rigorous_horizon(gamma, beta, w->rowRIA[m] * unorm, w->rho);
        if (h < s) s = h;
    }
    if (!isfinite(s)) return 0;

    const double est = floor(s) - (double)SKIP_MARGIN;
    return (est > 0.0) ? (int)est : 0;
}

static void mat_power_vec(accelerated_ws *w, int e, const double *v, double *out)
{
    const int q = w->q;
    if (e == 0) { memcpy(out, v, (size_t)q * sizeof(double)); return; }
    if (e == 1) { matvec(w->sT, v, out, q, q); return; }

    double *res = w->M1, *base = w->M2, *scratch = w->M3;
    memcpy(base, w->sT, (size_t)q * q * sizeof(double));
    int have_res = 0;
    while (e > 0) {
        if (e & 1) {
            if (!have_res) { memcpy(res, base, (size_t)q * q * sizeof(double)); have_res = 1; }
            else { matmul(res, base, scratch, q); double *t = res; res = scratch; scratch = t; }
        }
        e >>= 1;
        if (e > 0) { matmul(base, base, scratch, q); double *t = base; base = scratch; scratch = t; }
    }
    matvec(res, v, out, q, q);
}

/* The envelope proves every cycle up to the landing switch-free, reading drifts
 * and slacks within rounding as zero.  The landing state is computed with
 * rounding, so its signs are checked, and a shorter jump stays inside the
 * proven run. */
static int fast_forward_jump(accelerated_ws *w, int t, double *u, int cyc)
{
    const int q = w->q;
    const int budget = w->max_iter - cyc;
    if (budget < 1) return 0;

    int s = skip_estimate(w, t, nrm2(u, q));
    if (s > budget) s = budget;

    while (s >= 1) {
        mat_power_vec(w, s - 1, u, w->usm1);
        matvec(w->sT, w->usm1, w->us, q, q);
        closed_form_auxiliaries(w, (double)t + (double)s, w->us, w->dend);
        if (activity_holds(w, (double)t + (double)s, w->dend, w->us, w->usm1)) {
            state_from_modes(w, w->us, w->dend);
            memcpy(u, w->us, (size_t)q * sizeof(double));
            w->info.skips += 1;
            w->info.cycles_skipped += s;
            return s;
        }
        s /= 2;
    }
    return 0;
}

/* Fixed to rounding, coordinate by coordinate: holding a state that still
 * moves, however slowly, would leave Dykstra's path. */
static int is_frozen(const accelerated_ws *w)
{
    const int p = w->p;
    for (int k = 0; k < p; ++k) {
        const double *Ak = w->Am + (size_t)k * p;
        const double motion = fabs(dot(Ak, w->x, p) + w->Bm[k] - w->x[k]);
        const double scale = absdot(Ak, w->x, p) + fabs(w->Bm[k]) + fabs(w->x[k]);
        if (!(motion <= rounding_floor(scale))) return 0;
    }
    return 1;
}

/* Cycles until the first activity change of a frozen state.  A draining active
 * auxiliary reaches zero after ceil(d / -delta) cycles, one already at zero on
 * the next; a positive increment on an inactive half-space reactivates it at
 * once.  A crossing past the limit is never taken, and refining it one cycle at
 * a time would not finish at ratios like 1e30. */
static double stall_crossing(const accelerated_ws *w, double limit)
{
    double crossing = INFINITY;
    for (int m = 0; m < w->n; ++m) {
        const double y = w->active[m] ? w->d[m] : 0.0;
        const double delta = w->delta[m];
        if (!w->active[m]) {
            if (delta > w->dfloor[m]) crossing = 1.0;
            continue;
        }
        if (!(delta < -w->dfloor[m])) continue;
        if (y <= 0.0) { crossing = 1.0; continue; }
        const double ratio = y / (-delta);
        if (!(ratio <= limit)) {
            if (ratio < crossing) crossing = ratio;
            continue;
        }
        double k = floor(ratio);
        while (y + k * delta > 0.0) k += 1.0;
        while (k > 1.0 && y + (k - 1.0) * delta <= 0.0) k -= 1.0;
        if (k < 1.0) k = 1.0;
        if (k < crossing) crossing = k;
    }
    return crossing;
}

/* Jump to the cycle before the crossing, or through the rest of the budget when
 * the crossing lies beyond it, applying the increments at once while the state
 * stays frozen. */
static int stall_episode(accelerated_ws *w, int cycle, int *switch_cycle)
{
    const int n = w->n, p = w->p;
    for (int m = 0; m < n; ++m) {
        w->delta[m] = dot(w->R + (size_t)m * p, w->x, p) + w->svec[m];
        w->dfloor[m] = rounding_floor(absdot(w->Rs + (size_t)m * p, w->x, p) + w->ss[m]);
    }

    const double remaining = (double)w->max_iter - (double)cycle + 1.0;
    const double crossing = stall_crossing(w, (double)w->max_iter + 1.0);
    const double k = (crossing <= remaining) ? floor(crossing) - 1.0 : remaining;
    if (k >= 1.0) {
        for (int m = 0; m < n; ++m)
            w->dtmp[m] = (w->active[m] ? w->d[m] : 0.0) + k * w->delta[m];
        commit_duals(w, w->dtmp);
        w->info.skips += 1;
        w->info.cycles_skipped += (long long)k;
        w->info.cycles = cycle + (int)k - 1;
    }
    if ((double)cycle + k <= (double)w->max_iter) {
        *switch_cycle = cycle + (int)k;
        return EP_SWITCH;
    }
    return EP_BUDGET;
}

/* A final active set settles only if the fixed point passes the KKT test, since
 * that fixed point can still miss the projection.  Otherwise no switch can
 * follow, so the budget's iterate is one closed-form jump away. */
static int settle_final(accelerated_ws *w, int cyc, int t, const double *u)
{
    if (kkt_certificate(w, w->xinf)) {
        settle_certified(w);
        return EP_SETTLED;
    }
    const int k = w->max_iter - cyc;
    if (k >= 1) {
        mat_power_vec(w, k, u, w->us);
        closed_form_auxiliaries(w, (double)t + (double)k, w->us, w->dend);
        state_from_modes(w, w->us, w->dend);
        w->info.skips += 1;
        w->info.cycles_skipped += k;
        w->info.cycles = w->max_iter;
    }
    return EP_BUDGET;
}

/* Scan an episode in closed form.  The cycle that changes the active set is run
 * exactly by the caller, so on a switch the state stays as the previous cycle
 * committed it: rebuilding it from the closed form would move it by rounding
 * times the conditioning of the episode.  A state can freeze part-way through
 * a singular episode; probing at offsets 1, 2, 3, 5, 9, ... keeps the probes
 * logarithmic in its length. */
static int scan_episode(accelerated_ws *w, int cycle, int *switch_cycle)
{
    const int q = w->q;
    double *u_prev = w->u0, *u = w->u1;
    w->info.episodes += 1;

    int t = 1;
    long long next_probe = 1;
    /* Written so that nothing exceeds max_iter + 1: after the last pass t is
     * two past the cycles left, and cycle + t would overflow at the largest
     * budget. */
    while (t - 1 <= w->max_iter - cycle) {
        const int cyc = cycle + t - 1;

        if (w->deflated && t >= next_probe) {
            next_probe = (long long)t + (t > 2 ? t - 1 : 1);
            if (is_frozen(w)) return stall_episode(w, cyc, switch_cycle);
        }

        matvec(w->sT, u_prev, u, q, q);
        closed_form_auxiliaries(w, (double)t, u, w->dtmp);
        if (!activity_holds(w, (double)t, w->dtmp, u, u_prev)) {
            *switch_cycle = cyc;
            return EP_SWITCH;
        }

        state_from_modes(w, u, w->dtmp);
        w->info.cycles = cyc;

        if (active_set_is_final(w, (double)t, nrm2(u, q)))
            return settle_final(w, cyc, t, u);

        const int s = fast_forward_jump(w, t, u, cyc);
        if (s > 0) w->info.cycles = cyc + s;
        t += s + 1;
        double *sw = u_prev; u_prev = u; u = sw;
    }
    return EP_BUDGET;
}

/* Step an episode through its cycle map when neither closed form applies.
 * Confining the rounded step to the span stops the kernel component, which no
 * cycle moves, from drifting by a rounding error per cycle. */
static int step_episode(accelerated_ws *w, int cycle, int *switch_cycle)
{
    const int n = w->n, p = w->p;
    w->info.step_episodes += 1;
    ensure_span(w);

    long long next_probe = cycle;
    for (int cyc = cycle; cyc <= w->max_iter; ++cyc) {
        if (cyc == next_probe) {
            next_probe += (cyc - cycle > 1) ? cyc - cycle : 1;
            if (is_frozen(w)) return stall_episode(w, cyc, switch_cycle);
        }

        for (int m = 0; m < n; ++m) {
            const double delta = dot(w->R + (size_t)m * p, w->x, p) + w->svec[m];
            const double y_next = w->d[m] + delta;
            const double floor_m = rounding_floor(
                fabs(w->d[m]) + absdot(w->Rs + (size_t)m * p, w->x, p) + w->ss[m]);
            const int on = w->active[m] ? (y_next > -floor_m) : (delta > floor_m);
            if (on != (int)w->active[m]) {
                *switch_cycle = cyc;
                return EP_SWITCH;
            }
            w->dtmp[m] = y_next;
        }
        commit_duals(w, w->dtmp);

        for (int k = 0; k < p; ++k)
            w->pw1[k] = dot(w->Am + (size_t)k * p, w->x, p) + w->Bm[k] - w->x[k];
        span_project(w, w->pw1, w->pw2);
        for (int k = 0; k < p; ++k) w->x[k] += w->pw2[k];
        w->info.cycles = cyc;
    }
    return EP_BUDGET;
}

/* Episodes separated by exact switching cycles.  A point that passes the KKT
 * test is the projection whatever the episode would do next, and finding it
 * needs no resolvent, so it is tried first at every entry. */
static int accelerate(accelerated_ws *w)
{
    const int n = w->n, p = w->p;
    int cycle = 2;
    while (cycle <= w->max_iter) {

        if (kkt_certificate(w, w->x)) {
            settle_certified(w);
            return ACCELERATED_SETTLED;
        }

        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j)
                w->IA[(size_t)i * p + j] =
                    ((i == j) ? 1.0 : 0.0) - w->Am[(size_t)i * p + j];

        int ep, switch_cycle = 0;
        if (factor_resolvent(w) < COND_LIMIT) {
            setup_regular(w);
            ep = scan_episode(w, cycle, &switch_cycle);
        } else {
            int any = 0;
            for (int m = 0; m < n && !any; ++m) any = w->active[m];
            if (!any || !setup_deflated(w)) {
                ep = step_episode(w, cycle, &switch_cycle);
            } else if (kkt_certificate(w, w->xinf)) {
                settle_certified(w);
                return ACCELERATED_SETTLED;
            } else {
                ep = scan_episode(w, cycle, &switch_cycle);
            }
        }

        if (ep == EP_SETTLED) return ACCELERATED_SETTLED;
        if (ep == EP_BUDGET) return ACCELERATED_MAXCYCLES;

        dykstra_cycle(w);
        w->info.cycles = switch_cycle;
        w->info.switches += 1;
        build_cycle_map(w);
        cycle = switch_cycle + 1;
    }
    return ACCELERATED_MAXCYCLES;
}

/* Carves every double array out of one block; called first with base NULL to
 * size it.  solve_impl carves the int and flag arrays out of two more.  Each
 * scratch array belongs to one stage and is never shared with a stage that
 * can run while its contents are still needed. */
static size_t layout(accelerated_ws *w, double *base, size_t n, size_t p)
{
    const size_t mx = (n > p) ? n : p;
    size_t off = 0;
#define TAKE(field, count) do { if (base) w->field = base + off; off += (count); } while (0)
    TAKE(units, n * p);  TAKE(R, n * p);     TAKE(Rs, n * p);
    TAKE(RIA, n * p);    TAKE(RIAs, n * p);
    TAKE(Wy, n * p);     TAKE(Wg, n * p);    TAKE(Wys, n * p);  TAKE(Wgs, n * p);
    TAKE(Am, p * p);     TAKE(IA, p * p);    TAKE(LU, p * p);   TAKE(inv, p * p);
    TAKE(Q, p * p);      TAKE(T, p * p);
    TAKE(M1, p * p);     TAKE(M2, p * p);    TAKE(M3, p * p);
    TAKE(bvec, n);       TAKE(rnorm, n);     TAKE(d, n);
    TAKE(svec, n);       TAKE(ss, n);
    TAKE(G, n);          TAKE(Gf, n);        TAKE(beta, n);     TAKE(betaf, n);
    TAKE(rowRIA, n);     TAKE(rowR, n);
    TAKE(dtmp, n);       TAKE(dend, n);      TAKE(delta, n);    TAKE(dfloor, n);
    TAKE(x, p);          TAKE(Bm, p);        TAKE(xinf, p);
    TAKE(u0, p);         TAKE(u1, p);        TAKE(us, p);       TAKE(usm1, p);
    TAKE(pw1, p);        TAKE(pw2, p);       TAKE(pw3, p);
    TAKE(lsA, n * mx);   TAKE(lsM, n * mx);  TAKE(lsV, n * n);
    TAKE(lsrhs, mx);     TAKE(lstau, mx);    TAKE(lsy, mx);     TAKE(lssig, mx);
    TAKE(lsz, mx);       TAKE(fQ, p * p);    TAKE(fR, p * p);
    TAKE(fcoef, p);      TAKE(fy, p);
    TAKE(gram, n * n);   TAKE(qc, n);
    TAKE(nnw, n);        TAKE(nnstep, n);    TAKE(nnsol, n);    TAKE(nnprev, n);
    TAKE(nnv, p);        TAKE(nnmag, p);
    TAKE(kx, p);         TAKE(kmag, p);      TAKE(kstep, p);    TAKE(kcorr, p);
    TAKE(kapex, p);      TAKE(xcert, p);
    TAKE(klam, n);       TAKE(kdual, n);     TAKE(mult, n);
#undef TAKE
    return off;
}

static int solve_impl(int n_planes, int dim,
                      const double *A, const double *b, const double *z,
                      int max_cycles, double *x_out, double *duals_out,
                      accelerated_info *info)
{
    /* A budget of INT_MAX is refused: the cycle counters step one past the
     * budget before they stop. */
    if (n_planes < 0 || dim < 1 || max_cycles < 1 || max_cycles == INT_MAX
        || !z || !x_out || (n_planes > 0 && (!A || !b)))
        return ACCELERATED_ERR_ARGS;

    if (n_planes == 0) {
        memcpy(x_out, z, (size_t)dim * sizeof(double));
        if (info) {
            memset(info, 0, sizeof(*info));
            info->status = ACCELERATED_SETTLED;
        }
        return ACCELERATED_SETTLED;
    }

    const size_t n = (size_t)n_planes, p = (size_t)dim;
    const size_t mx = (n > p) ? n : p;

    /* A size whose byte count could wrap size_t is refused before the count
     * is formed, whatever the width of size_t: the arrays hold fewer than
     * 24 (n + p)^2 + 64 (n + p) doubles. */
    {
        const double s = (double)n + (double)p;
        if (24.0 * s * s + 64.0 * s > (double)(SIZE_MAX / sizeof(double)) / 2.0)
            return ACCELERATED_ERR_ALLOC;
    }

    accelerated_ws w;
    memset(&w, 0, sizeof(w));
    const size_t nd = layout(&w, NULL, n, p);
    const size_t ni = 3 * mx + 8 * n + p + KKT_EXHAUSTIVE;
    double *arena = (double *)malloc(nd * sizeof(double));
    int *iarena = (int *)malloc(ni * sizeof(int));
    unsigned char *barena = (unsigned char *)malloc(3 * n);
    if (!arena || !iarena || !barena) {
        free(arena); free(iarena); free(barena);
        return ACCELERATED_ERR_ALLOC;
    }

    layout(&w, arena, n, p);
    w.n = n_planes; w.p = dim; w.max_iter = max_cycles;
    int *ia = iarena;
    w.ipiv   = ia; ia += mx;
    w.lsperm = ia; ia += mx;
    w.nnsup  = ia; ia += mx;
    w.krows  = ia; ia += n;
    w.ksup   = ia; ia += 4 * n;
    w.ktight = ia; ia += n;
    w.ksub   = ia; ia += n + KKT_EXHAUSTIVE;
    w.fcol   = ia; ia += p;
    w.fpos   = ia; ia += n;
    w.active = barena;
    w.nnpass = barena + n;
    w.nnpprev = barena + 2 * n;
    w.z = z;

    int status = -1000;
    for (int m = 0; m < n_planes; ++m) {
        const double *row = A + (size_t)m * dim;
        const double norm = nrm2(row, dim);
        if (norm == 0.0) { status = ACCELERATED_ERR_ZERO_NORMAL; goto done; }
        w.rnorm[m] = norm;
        double *um = w.units + (size_t)m * dim;
        for (int k = 0; k < dim; ++k) um[k] = row[k] / norm;
        w.bvec[m] = b[m] / norm;
    }

    memcpy(w.x, z, p * sizeof(double));
    memset(w.d, 0, n * sizeof(double));

    dykstra_cycle(&w);
    w.info.cycles = 1;

    build_cycle_map(&w);
    status = accelerate(&w);

done:
    if (status >= 0) {
        memcpy(x_out, w.x, p * sizeof(double));
        if (duals_out)
            for (int m = 0; m < n_planes; ++m)
                duals_out[m] = w.d[m] / w.rnorm[m];
        w.info.status = status;
        if (info) *info = w.info;
    }
    free(arena); free(iarena); free(barena);
    return status;
}

ACCELERATED_API int accelerated_dykstra_solve(int n_planes, int dim,
                                              const double *A, const double *b,
                                              const double *z, int max_cycles,
                                              double *x_out, double *duals_out,
                                              accelerated_info *info)
{
    return solve_impl(n_planes, dim, A, b, z, max_cycles,
                      x_out, duals_out, info);
}
