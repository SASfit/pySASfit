/* shim: replaces the SASfit private.h of plugin linear_pearls (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_LINEAR_PEARLS_PRIVATE_H
#define SHIM_LINEAR_PEARLS_PRIVATE_H
#include <sasfit_shim.h>

scalar linear_pearls_sphere_f(scalar x);
scalar linear_pearls_sinc(scalar x);
scalar sasfit_ff_linear_pearls_kernel(scalar q, scalar NN, sasfit_param * param);
scalar sasfit_ff_linear_pearls(scalar q, sasfit_param * param);
scalar sasfit_ff_linear_pearls_f(scalar q, sasfit_param * param);
scalar sasfit_ff_linear_pearls_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
