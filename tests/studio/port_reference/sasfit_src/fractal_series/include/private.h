/* shim: replaces the SASfit private.h of plugin fractal_series (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_FRACTAL_SERIES_PRIVATE_H
#define SHIM_FRACTAL_SERIES_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_fractal_series1(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_fractal_series2(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_fractal_series3(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fractal_series3_v(scalar q, sasfit_param * param, int dist);
scalar fractalSD_single(scalar x, scalar a,scalar fd, scalar xmin, scalar xmax,sasfit_param * param);
scalar sc_ExpIntegralE(double n, double x);
scalar fractalFF_single(scalar Q, scalar a,scalar fd, scalar xmin, scalar xmax,sasfit_param * param);
scalar sasfit_sd_fractal_series1(scalar x, sasfit_param * param);
scalar sasfit_sd_fractal_series1_f(scalar q, sasfit_param * param);
scalar sasfit_sd_fractal_series1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_fractal_series2(scalar x, sasfit_param * param);
scalar sasfit_sd_fractal_series2_f(scalar q, sasfit_param * param);
scalar sasfit_sd_fractal_series2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_fractal_series3(scalar x, sasfit_param * param);
scalar sasfit_sd_fractal_series3_f(scalar q, sasfit_param * param);
scalar sasfit_sd_fractal_series3_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
