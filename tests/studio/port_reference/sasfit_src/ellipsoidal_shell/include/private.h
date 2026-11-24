/* shim: replaces the SASfit private.h of plugin ellipsoidal_shell (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_ELLIPSOIDAL_SHELL_PRIVATE_H
#define SHIM_ELLIPSOIDAL_SHELL_PRIVATE_H
#include <sasfit_shim.h>
#define f_sph ellsh_f_sph   /* reference library: f_sph is also defined in sheared_objects */

scalar sasfit_ff_ellip_shell_0_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_0_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_0_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_0t_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_0t_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_0t_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0t(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_0t_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_2_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_2_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_2_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_2t_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_2t_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_2t_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2t(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_2t_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_re_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_re_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_re_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_re_t_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_re_t_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_re_t_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re_t(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re_t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_re_t_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_rp_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_rp_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_rp_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellip_shell_rp_t_core(sasfit_param * param);
scalar sasfit_ff_ellip_shell_rp_t_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellip_shell_rp_t_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp_t(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp_t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellipsoidal_shell_rp_t_v(scalar q, sasfit_param * param, int dist);
scalar f_sph(scalar x);
scalar F_EllSh(sasfit_param * param);
scalar ellip_shell_cubature(const double *x, size_t ndim, void *pam);
scalar ellip_shell_mint(const double *x, size_t ndim, void *pam);
scalar sasfit_ff_spheroid_R_core(sasfit_param * param);
scalar sasfit_ff_spheroid_R_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_spheroid_R(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_R_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_R_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_spheroid_V_core(sasfit_param * param);
scalar sasfit_ff_spheroid_V_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_spheroid_V_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_spheroid_V(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_V_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_V_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_spheroid_nu_core(sasfit_param * param);
scalar sasfit_ff_spheroid_nu_core_MU(scalar x, sasfit_param * param);
scalar sasfit_ff_spheroid_nu(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_nu_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spheroid_nu_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
typedef scalar sasfit_cubature_f(sasfit_param *);

typedef struct
{
	sasfit_param *param;
	sasfit_cubature_f *func; //!< Selects the kernel function to use, when used with gsl functions.
	scalar *cubxmin;
	scalar *cubxmax;
	int ndim;
} cubature_param;

#define SIGMA	    fabs(param->p[4])
#define ETA_CORE	param->p[5]
#define ETA_SHELL	param->p[6]
#define ETA_SOLV	param->p[7]
#define X_CORE      param->p[8]
#define X_SHELL     param->p[9]
#define Q	param->p[MAXPAR-1]
#define P	param->p[MAXPAR-2]
#define NU	param->p[MAXPAR-3]
#define MU	param->p[MAXPAR-4]
#define A	param->p[MAXPAR-5]
#define B	param->p[MAXPAR-6]
#define TNU	param->p[MAXPAR-7]
#define LNDISTR	param->p[MAXPAR-8]
#define NUMIN	param->p[MAXPAR-9]
#define NUMAX	param->p[MAXPAR-10]

scalar F_EllSh(sasfit_param *param);
scalar ellip_shell_cubature(const double *x, size_t ndim, void *pam);
scalar ellip_shell_mint(const double *x, size_t ndim, void *pam);
#endif
