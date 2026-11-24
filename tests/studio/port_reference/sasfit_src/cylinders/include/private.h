/* shim: replaces the SASfit private.h of plugin cylinders (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_CYLINDERS_PRIVATE_H
#define SHIM_CYLINDERS_PRIVATE_H
#include <sasfit_shim.h>

scalar ell_rad(scalar radius, scalar epsilo, scalar t, scalar theta);
scalar AcylSHell(sasfit_param *param);
scalar A_ellcyl(const double *x, size_t ndim, void *pam);
scalar sasfit_ff_alignedcylshell(scalar q, sasfit_param * param);
scalar sasfit_ff_alignedcylshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_alignedcylshell_v(scalar q, sasfit_param * param, int dist);
scalar cylinder_kernel_f(scalar x, sasfit_param *param);
scalar cylinder_kernel(scalar x, sasfit_param * param);
scalar sasfit_ff_cylinder(scalar q, sasfit_param * param);
scalar sasfit_ff_cylinder_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cylinder_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_cylshell1(scalar q, sasfit_param * param);
scalar sasfit_ff_cylshell1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cylshell1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_cylshell2(scalar q, sasfit_param * param);
scalar sasfit_ff_cylshell2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cylshell2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_disc(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_v(scalar q, sasfit_param * param, int dist);
scalar ell_cylshell1_avg(scalar theta, scalar phi, sasfit_param *param);
scalar sasfit_ff_ellcylshell1(scalar q, sasfit_param * param);
scalar sasfit_ff_ellcylshell1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellcylshell1_v(scalar x, sasfit_param * param, int dist);
scalar ell_cylshell2_avg(scalar theta, scalar phi, sasfit_param *param);
scalar sasfit_ff_ellcylshell2(scalar q, sasfit_param * param);
scalar sasfit_ff_ellcylshell2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellcylshell2_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_flatcylinder(scalar q, sasfit_param * param);
scalar sasfit_ff_flatcylinder_f(scalar q, sasfit_param * param);
scalar sasfit_ff_flatcylinder_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_longcylinder(scalar q, sasfit_param * param);
scalar sasfit_ff_longcylinder_f(scalar q, sasfit_param * param);
scalar sasfit_ff_longcylinder_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_longcylshell(scalar q, sasfit_param * param);
scalar sasfit_ff_longcylshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_longcylshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_porodcylinder(scalar q, sasfit_param * param);
scalar sasfit_ff_porodcylinder_f(scalar q, sasfit_param * param);
scalar sasfit_ff_porodcylinder_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_porodcylinder_v2(scalar q, sasfit_param * param);
scalar sasfit_ff_porodcylinder_v2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_porodcylinder_v2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_rod(scalar q, sasfit_param * param);   /* in sasfit_src/rod (identical copy) */
scalar Torus_kernel_int_r(scalar r, sasfit_param *param);
scalar Torus_kernel_int_theta(scalar theta, sasfit_param *param);
scalar sasfit_ff_torus(scalar q, sasfit_param * param);
scalar sasfit_ff_torus_f(scalar q, sasfit_param * param);
scalar sasfit_ff_torus_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_pcs_ellcylsh(scalar, sasfit_param *);
scalar sasfit_sq_p__q___rod(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
scalar AcylSHell(sasfit_param *);
scalar A_theta(scalar, void *);
scalar A_alpha(scalar, void *);
scalar A_alpha_sasfit(scalar, sasfit_param *);
int A_cub2(unsigned, const double *, void *, unsigned, double *);
double A_ellcyl(const double *, size_t, void *);
#endif
