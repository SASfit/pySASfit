/* shim: replaces the SASfit private.h of plugin hammouda_branched (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_HAMMOUDA_BRANCHED_PRIVATE_H
#define SHIM_HAMMOUDA_BRANCHED_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_fract_branched_polym(scalar q, sasfit_param * param);
scalar sasfit_ff_fract_branched_polym_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fract_branched_polym_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
