/* shim: replaces the SASfit private.h of plugin cv (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_CV_PRIVATE_H
#define SHIM_CV_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_cv(scalar q, sasfit_param * param);
scalar sasfit_ff_cv_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cv_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
