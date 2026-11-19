/* shim: replaces the SASfit private.h of plugin ribbon (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_RIBBON_PRIVATE_H
#define SHIM_RIBBON_PRIVATE_H
#include <sasfit_shim.h>

scalar ribbon_f1f2(scalar x);
scalar sasfit_ff_ribbon_mono(scalar q, sasfit_param * param);
scalar sasfit_ff_ribbon_mono_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ribbon_mono_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_peaks_ribbon_w_sd(scalar q, sasfit_param * param);
scalar sasfit_peaks_ribbon_w_sd_f(scalar q, sasfit_param * param);
scalar sasfit_peaks_ribbon_w_sd_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_rod(scalar, sasfit_param *);

//
// 1F2(1/2; 3/2, 2; -x^2) by the Neumann series of odd Bessel functions (sasfit_ff_ribbon_mono.c)
scalar ribbon_f1f2(scalar x);
// add local defines here:
// #define P0 param->p[0]
//
#endif
