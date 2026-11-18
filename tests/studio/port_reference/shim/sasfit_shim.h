/* Minimal stand-in for the SASfit headers: enough to compile selected plugin sources unchanged as a plain
 * shared library (reference for the Python ports, tests/studio/test_ports.py). Not the SASfit plugin ABI. */
#ifndef SASFIT_SHIM_H
#define SASFIT_SHIM_H
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

typedef double scalar;
#define MAXPAR 50
/* kernel selectors used by the imported plugins (values arbitrary, sasfit_function.h has the full list) */
typedef enum {
    SPHERE_RMA, SPHERE_RMA_RC, SPHERE_RMA_NAGG, SPHERE_2ND_RMA, SPHERE_2ND_RMA_RC, SPHERE_2ND_RMA_NAGG,
    SPHERE_RMA_RC_SMOOTH, SPHERE_RMA_NAGG_SMOOTH, SPHERE_RWBRUSH, SPHERE_RWBRUSH_RC, SPHERE_RWBRUSH_NAGG,
    SPHERE_SAW, SPHERE_SAW_RC, SPHERE_SAW_NAGG
} sasfit_kernel_type;
typedef struct {
    scalar p[MAXPAR];
    int errStatus;
    char errStr[256];
    int kernelSelector; /* sasfit_kernel_type */
} sasfit_param;

#define SASFIT_ASSERT_PTR(ptr) do { if (!(ptr)) return 0.0; } while (0)
#define SASFIT_CHECK_COND(cond, param, msg) \
    do { if (cond) { (param)->errStatus = 1; snprintf((param)->errStr, 256, "%s", msg); return 0.0; } } while (0)
#define SASFIT_CHECK_COND1(cond, param, fmt, a) \
    do { if (cond) { (param)->errStatus = 1; snprintf((param)->errStr, 256, fmt, a); return 0.0; } } while (0)
#define SASFIT_CHECK_COND2(cond, param, fmt, a, b) \
    do { if (cond) { (param)->errStatus = 1; snprintf((param)->errStr, 256, fmt, a, b); return 0.0; } } while (0)

scalar sasfit_param_override_get_psi(scalar default_val);
void sasfit_init_param(sasfit_param *param);
void sasfit_param_set_err(sasfit_param *param, const char *format, ...);
scalar sasfit_integrate(scalar a, scalar b, scalar (*f)(scalar, sasfit_param *), sasfit_param *param);

void sasfit_out(const char *format, ...);
void sasfit_err(const char *format, ...);
scalar sasfit_eps_get_nriq(void);
scalar sasfit_eps_get_aniso(void);
typedef double (*sasfit_cubature_kernel)(const double *x, size_t ndim, void *param);
int sasfit_cubature(size_t ndim, const double *xmin, const double *xmax, sasfit_cubature_kernel f, void *param,
                    double eps, double *sum, double *err);

/* sasfit_common utilities used by sheared_objects */
#ifndef FALSE
#define FALSE 0
#endif
#define SASFIT_CHECK_SUB_ERR(param, sub) do { (void)(sub); } while (0)
scalar find_LogNorm_int_range(scalar dim, scalar x0, scalar sigma, scalar *Xstart, scalar *Xend, sasfit_param *param);
scalar sasfit_sd_LogNorm(scalar x, sasfit_param *param);
double re_cerfi_z(double x, double y);
double im_cerfi_z(double x, double y);

/* sasfit_common */
#define EMPTY ((scalar *)0)
/* sasfit_common.h: what the _v function of a distribution returns */
typedef enum {
    DISTRIBUTION_TYPE, DISTRIBUTION_QUANTILE, DISTRIBUTION_QUANTILE_DENS, DISTRIBUTION_PROBABILITY,
    DISTRIBUTION_CUMULATIVE, DISTRIBUTION_MEDIAN, DISTRIBUTION_MEAN, DISTRIBUTION_MODE, DISTRIBUTION_VARIANCE,
    DISTRIBUTION_SKEWNESS, DISTRIBUTION_EXCESS_KURTOSIS
} sasfit_distribution_kind;
#define SASFIT_RETURNVAL_ON_ERROR 0.0
#define DBGINFO(s) s
#define SASFIT_EQUAL(a, b) ((a) == (b)) /* only used for exact-zero tests here */
void sasfit_get_param(sasfit_param *param, int num, ...);
#define SASFIT_ERR_UNKNOWN_KERNEL(param, name) SASFIT_CHECK_COND(1, param, "unknown kernel")
scalar sasfit_jinc(scalar x);
scalar sasfit_eps_get_robertus_p(void);
scalar sasfit_2f1(scalar a, scalar b, scalar c, scalar x, sasfit_param *param);
#ifndef GSL_POSINF
#define GSL_POSINF (1.0/0.0)
#endif
scalar sasfit_rwbrush_w(scalar Q, scalar R);
scalar sasfit_gauss_fc(scalar Q, scalar R);

/* functions of other plugins used through the plugin import table in SASfit */
scalar sasfit_sq_p__q___worm_ps3_(scalar q, sasfit_param *param); /* not in the reference library: NaN */
scalar sasfit_sphere_fc(scalar Q, scalar R);
scalar sasfit_ff_ringpolymerBMB(scalar q, sasfit_param *param);
scalar sasfit_ff_ringpolymerBZ(scalar q, sasfit_param *param);
#endif
