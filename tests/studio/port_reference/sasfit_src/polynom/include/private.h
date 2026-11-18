/* shim: replaces the SASfit private.h of plugin polynom (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_POLYNOM_PRIVATE_H
#define SHIM_POLYNOM_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_cv2(scalar x, sasfit_param * param);
scalar sasfit_ff_cv2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cv2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_polynom(scalar q, sasfit_param * param);
scalar sasfit_ff_polynom_f(scalar q, sasfit_param * param);
scalar sasfit_ff_polynom_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
