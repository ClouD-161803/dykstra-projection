/*
 * dykstra.h - plain Dykstra in dependency-free C.
 * Projects a point z onto the polyhedron {x : A x <= b} with the standard
 * cyclic algorithm; the baseline the accelerated solver is timed against.
 *
 * Usage: call dykstra_solve (documented below); build the library with
 * build.ps1 (Windows/MSVC) or make (gcc/clang).
 * Nothing beyond libm is needed.  No globals, so calls on different threads
 * do not interfere.  No per-cycle history is recorded.
 */
#ifndef DYKSTRA_H
#define DYKSTRA_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(DYKSTRA_BUILD_DLL)
#  define DYK_API __declspec(dllexport)
#elif !defined(_WIN32) && defined(__GNUC__)
#  define DYK_API __attribute__((visibility("default")))
#else
#  define DYK_API
#endif

enum {
    DYK_SETTLED   = 0,  /* a full cycle left the state unchanged bit for bit,
                           so x_out is the iterate at every larger budget too */
    DYK_MAXCYCLES = 1   /* budget exhausted: x_out is the iterate after
                           max_cycles cycles                                  */
};

enum {
    DYK_ERR_ARGS        = -1,  /* bad dimensions / NULL pointer              */
    DYK_ERR_ZERO_NORMAL = -2,  /* a constraint row has zero norm             */
    DYK_ERR_ALLOC       = -3   /* out of memory                              */
};

typedef struct {
    int status;   /* DYK_SETTLED or DYK_MAXCYCLES                            */
    int cycles;   /* full cycles executed                                    */
} dykstra_info;

/*
 * Project z onto {x : A x <= b} with plain Dykstra.
 *
 *   n_planes    number of half-spaces (>= 0)
 *   dim         dimension of the point (>= 1)
 *   A           n_planes x dim row-major matrix of half-space normals
 *               (rows need not be unit length; a zero row is an error)
 *   b           n_planes offsets
 *   z           point to project (dim entries)
 *   max_cycles  cycle budget (>= 1)
 *   x_out       result (dim entries)
 *   duals_out   optional (may be NULL): n_planes multipliers lambda >= 0 with
 *               respect to the caller's rows, Dykstra's corrections being
 *               e_m = lambda_m A_m, so that z - x_out = A^T lambda
 *   info        optional counters; pass NULL if not wanted
 *
 * Returns the status code (>= 0) on success, or a negative error code.
 * A finite number of cycles generally gives an approximation, not the
 * projection.  The inputs are not checked for NaN or infinity.
 */
DYK_API int dykstra_solve(int n_planes, int dim,
                          const double *A, const double *b, const double *z,
                          int max_cycles, double *x_out, double *duals_out,
                          dykstra_info *info);

#ifdef __cplusplus
}
#endif

#endif /* DYKSTRA_H */
