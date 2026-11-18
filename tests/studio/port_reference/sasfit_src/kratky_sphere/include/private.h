/* shim: replaces the SASfit private.h of plugin kratky_sphere (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_KRATKY_SPHERE_PRIVATE_H
#define SHIM_KRATKY_SPHERE_PRIVATE_H
#include <sasfit_shim.h>

scalar FFPhiKratky(scalar qperp, sasfit_param * param);
scalar sasfit_ff_kratkysphere(scalar q, sasfit_param * param);
scalar sasfit_ff_kratkysphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_kratkysphere_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
