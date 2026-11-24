/* shim: replaces the SASfit private.h of plugin triax_ellip_shell (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_TRIAX_ELLIP_SHELL_PRIVATE_H
#define SHIM_TRIAX_ELLIP_SHELL_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_triax_ellip_shell_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_cub_core(const double *x, size_t dim, void * pam);
scalar sasfit_ff_triax_ellip_shell_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_core_y(scalar y, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_core_x(scalar x, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_1_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_1t_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1t(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_1t_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_2_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_2t_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t_core_NU(scalar x, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t_core_y(scalar y, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t_core_x(scalar x, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_2t_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_3_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_3t_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3t(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_3t_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_syl_core(scalar y, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_syl_core_x(scalar x, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_syl(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_syl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_syl_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_triax_ellip_shell_t_core(sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_t(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_t_f(scalar q, sasfit_param * param);
scalar sasfit_ff_triax_ellip_shell_t_v(scalar x, sasfit_param * param, int dist);
scalar triax_ellip_shell_cubature(const scalar *x, size_t ndim, void *pam);
scalar triax_ellip_shell_mean_volume(scalar x, sasfit_param *param, int dist, int sa, int sb, int sc, int st);
scalar triax_ellip_shell_phi(scalar u);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);

typedef scalar sasfit_cubature_f(sasfit_param *);
typedef scalar sasfit_cubature_p(scalar , sasfit_param *);

typedef struct
{
	sasfit_param *param;
	sasfit_cubature_f *func; //!< Selects the kernel function to use, when used with gsl functions.
	sasfit_cubature_p *p1; //!< Selects the kernel function to use, when used with gsl functions.
	sasfit_cubature_p *p2; //!< Selects the kernel function to use, when used with gsl functions.
	sasfit_cubature_p *p3; //!< Selects the kernel function to use, when used with gsl functions.
	scalar *cubxmin;
	scalar *cubxmax;
	int ndim;
} cubature_param;

//
// add local defines here:
// #define P0 param->p[0]
//
#define	Q	param->p[MAXPAR-1]
#define	P	param->p[MAXPAR-2] // pow
#define	X	param->p[MAXPAR-3] // param from outer integration
#define	Y	param->p[MAXPAR-4] // param from middle integration
#define NU	param->p[MAXPAR-5] // param from innermost integration
#define LNDISTR	param->p[MAXPAR-6]
#define NUMIN	param->p[MAXPAR-7]
#define NUMAX	param->p[MAXPAR-8]

#define SIGMA	param->p[4]
#define ETA_C	param->p[5]
#define ETA_SH	param->p[6]
#define ETA_SOL	param->p[7]

scalar triax_ellip_shell_cubature(const double *x, size_t ndim, void *pam);
scalar triax_ellip_shell_phi(scalar u);
scalar triax_ellip_shell_mean_volume(scalar x, sasfit_param *param, int dist, int sa, int sb, int sc, int st);
#endif
