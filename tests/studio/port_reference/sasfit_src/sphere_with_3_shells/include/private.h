/* shim: replaces the SASfit private.h of plugin sphere_with_3_shells (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_SPHERE_WITH_3_SHELLS_PRIVATE_H
#define SHIM_SPHERE_WITH_3_SHELLS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_sphere_with_3_shells(scalar q, sasfit_param * param);
scalar SF(scalar q,scalar r);
scalar sasfit_ff_sphere_with_3_shells_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_with_3_shells_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
