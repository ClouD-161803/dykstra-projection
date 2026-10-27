/*
 * accelerated.h - the accelerated Dykstra solver in dependency-free C.
 * Projects a point z onto the polyhedron {x : A x <= b}.
 *
 * It is the deflated_modal preset of lti_solver.LTISolver (LTIVer5Solver):
 * Dykstra's own iterate after the cycle budget unless it settles first, and
 * the projection once it settles.
 *
 * Usage: call accelerated_dykstra_solve (documented below); build the library
 * with build.ps1 (Windows/MSVC) or make (gcc/clang).
 * Nothing beyond libm is needed.  No globals, so calls on different threads
 * do not interfere.  No per-cycle history is recorded: only the final
 * iterate, a status code and the counters below are returned.
 */
#ifndef ACCELERATED_H
#define ACCELERATED_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(ACCELERATED_BUILD_DLL)
#  define ACCELERATED_API __declspec(dllexport)
#elif !defined(_WIN32) && defined(__GNUC__)
#  define ACCELERATED_API __attribute__((visibility("default")))
#else
#  define ACCELERATED_API
#endif

/* status codes (accelerated_info.status and non-negative return values) */
enum {
    ACCELERATED_SETTLED   = 0,  /* x_out passed the KKT test: it is the
                                   projection, to about 1e-9 of the distance
                                   moved from z beyond the rounding of the
                                   data                                       */
    ACCELERATED_MAXCYCLES = 1   /* cycle budget exhausted: x_out is the
                                   iterate Dykstra holds after max_cycles
                                   cycles                                     */
};

/* schedule codes: where the KKT test is tried, as lti_numerics.KKTSchedule
 * describes and in the order of lti_numerics.SCHEDULES */
enum {
    ACCELERATED_SCHEDULE_WORK     = 0,  /* at the first phase entry, then once the
                                           work since the last failed entry test
                                           exceeds that test's work              */
    ACCELERATED_SCHEDULE_EVERY    = 1,  /* at every phase entry                  */
    ACCELERATED_SCHEDULE_DOUBLING = 2,  /* at phase entries 1, 2, 4, 8, ...      */
    ACCELERATED_SCHEDULE_NEVER    = 3   /* nowhere, not at a singular limit nor
                                           at a final active set either, so the
                                           solver never settles                  */
};

/* error return values (negative) */
enum {
    ACCELERATED_ERR_ARGS        = -1,  /* bad dimensions, a NULL pointer, or
                                          an unknown schedule                */
    ACCELERATED_ERR_ZERO_NORMAL = -2,  /* a constraint row has zero norm     */
    ACCELERATED_ERR_ALLOC       = -3   /* out of memory                      */
};

typedef struct {
    int       status;          /* ACCELERATED_SETTLED or _MAXCYCLES           */
    int       cycles;          /* cycles accounted for when the solver
                                  stopped, skipped ones included              */
    int       episodes;        /* episodes scanned in closed form, regular
                                  or deflated                                 */
    int       step_episodes;   /* episodes stepped through the cycle map      */
    int       switches;        /* exact Dykstra cycles after the first        */
    int       skips;           /* jumps taken: envelope jumps, frozen-stall
                                  fast-forwards and the jump to the budget    */
    long long cycles_skipped;  /* cycles those jumps covered                  */
    int       kkt_attempts;    /* KKT tests run, at phase entries, at the
                                  limits of singular phases and at final
                                  active sets                                 */
    int       entry_tests;     /* those tried at phase entries                */
    long long failed_entry_work; /* work of the entry tests that failed, in
                                  the flops of lti_numerics.kkt_test_work     */
    long long work;            /* all other work, in the same unit            */
    int       settled_at;      /* on ACCELERATED_SETTLED, the cycle from which
                                  x_out is the projection, as LTISolver's
                                  settled_at: the entry of the phase whose test
                                  passed, or the cycle a final active set was
                                  proven on                                   */
} accelerated_info;

/*
 * Project z onto {x : A x <= b}.
 *
 *   n_planes    number of half-spaces (>= 0)
 *   dim         dimension of the point (>= 1)
 *   A           n_planes x dim row-major matrix of half-space normals
 *               (rows need not be unit length; a zero row is an error)
 *   b           n_planes offsets
 *   z           point to project (dim entries)
 *   max_cycles  cycle budget (1 <= max_cycles < INT_MAX), counted in Dykstra
 *               cycles, skipped ones included
 *   schedule    one of the ACCELERATED_SCHEDULE_ codes
 *   x_out       result (dim entries)
 *   duals_out   optional (may be NULL): n_planes multipliers lambda >= 0 with
 *               respect to the caller's rows, z - x_out = A^T lambda.  On
 *               ACCELERATED_SETTLED they are the KKT multipliers of the
 *               projection; on ACCELERATED_MAXCYCLES they are Dykstra's
 *               corrections at the budget, e_m = lambda_m A_m
 *   info        optional counters; pass NULL if not wanted
 *
 * Returns the status code (>= 0) on success, or a negative error code.
 * The inputs are not checked for NaN or infinity.
 */
ACCELERATED_API int accelerated_dykstra_solve(int n_planes, int dim,
                                              const double *A, const double *b,
                                              const double *z, int max_cycles,
                                              int schedule,
                                              double *x_out, double *duals_out,
                                              accelerated_info *info);

#ifdef __cplusplus
}
#endif

#endif /* ACCELERATED_H */
