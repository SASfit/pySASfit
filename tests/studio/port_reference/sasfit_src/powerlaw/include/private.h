/* shim: replaces the SASfit private.h of plugin powerlaw (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_POWERLAW_PRIVATE_H
#define SHIM_POWERLAW_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_power_law(scalar q, sasfit_param * param);
scalar sasfit_ff_power_law_f(scalar q, sasfit_param * param);
scalar sasfit_ff_power_law_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile__power_law(scalar r, sasfit_param * param);
scalar sasfit_ff_profile__power_law_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile__power_law_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
