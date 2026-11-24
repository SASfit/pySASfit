/* shim: replaces the SASfit private.h of plugin peak_trapezoidal (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_PEAK_TRAPEZOIDAL_PRIVATE_H
#define SHIM_PEAK_TRAPEZOIDAL_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_general__trapez__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_general__trapez__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_general__trapez__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_general__trapez__area(scalar x, sasfit_param * param);
scalar sasfit_ff_general__trapez__area_f(scalar q, sasfit_param * param);
scalar sasfit_ff_general__trapez__area_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_smoothed_symm_trapez_ampl(scalar x, sasfit_param * param);
scalar sasfit_ff_smoothed_symm_trapez_ampl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_smoothed_symm_trapez_ampl_v(scalar q, sasfit_param * param, int dist);
scalar smoothed_box(scalar u, sasfit_param *param);
scalar smoothed_trapez(scalar u, sasfit_param *param);
scalar sasfit_ff_smoothed_symm_trapez_area(scalar x, sasfit_param * param);
scalar sasfit_ff_smoothed_symm_trapez_area_f(scalar q, sasfit_param * param);
scalar sasfit_ff_smoothed_symm_trapez_area_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_symm__general__trapez__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_symm__general__trapez__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_symm__general__trapez__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_symm__general__trapez__area(scalar x, sasfit_param * param);
scalar sasfit_ff_symm__general__trapez__area_f(scalar q, sasfit_param * param);
scalar sasfit_ff_symm__general__trapez__area_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_symm__trapez__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_symm__trapez__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_symm__trapez__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_symm__trapez__area(scalar x, sasfit_param * param);
scalar sasfit_ff_symm__trapez__area_f(scalar q, sasfit_param * param);
scalar sasfit_ff_symm__trapez__area_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_trapez__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_trapez__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_trapez__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_trapez__area(scalar x, sasfit_param * param);
scalar sasfit_ff_trapez__area_f(scalar q, sasfit_param * param);
scalar sasfit_ff_trapez__area_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
