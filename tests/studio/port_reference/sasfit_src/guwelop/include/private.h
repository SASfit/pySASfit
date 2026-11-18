/* shim: replaces the SASfit private.h of plugin guwelop (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_GUWELOP_PRIVATE_H
#define SHIM_GUWELOP_PRIVATE_H
#include <sasfit_shim.h>

scalar guwelop_cyl_Pcs(scalar x, sasfit_param *param);
scalar sasfit_ff_guwelop_cyl_Pcs(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_cyl_Pcs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_cyl_Pcs_v(scalar q, sasfit_param * param, int dist);
scalar guwelop_planar_Pcs(scalar x, sasfit_param *param);
scalar sasfit_ff_guwelop_planar_Pcs(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_planar_Pcs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_planar_Pcs_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_guwelop_profile(scalar x, sasfit_param * param);
scalar sasfit_ff_guwelop_profile_f(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_profile_v(scalar q, sasfit_param * param, int dist);
scalar guwelop_sphere(scalar x, sasfit_param *param);
scalar sasfit_ff_guwelop_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_guwelop_sphere_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_p_guwelop(scalar x, sasfit_param * param);
scalar sasfit_sd_p_guwelop_f(scalar x, sasfit_param * param);
scalar sasfit_sd_p_guwelop_v(scalar u, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#define N	param->p[0]

#define BL	GSL_MIN(param->p[1],param->p[2])
#define BU	GSL_MAX(param->p[1],param->p[2])
#define DIM	param->p[3]

#define THETA	param->p[4]
#define ALPHA	fabs(param->p[5])
#define BETA	fabs(param->p[6])
#define GAMMA	fabs(param->p[7])
#define LAMBDA 	fabs(param->p[8])

#define Q       param->p[MAXPAR-1]
#endif
