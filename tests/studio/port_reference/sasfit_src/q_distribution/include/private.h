/* shim: replaces the SASfit private.h of plugin q_distribution (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_Q_DISTRIBUTION_PRIVATE_H
#define SHIM_Q_DISTRIBUTION_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_q_exponential__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_exponential__ampl__f(scalar x, sasfit_param * param);
scalar sasfit_ff_q_exponential__ampl__v(scalar p, sasfit_param * param, int dist);
scalar sasfit_ff_q_exponential__area_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_exponential__area__f(scalar x, sasfit_param * param);
scalar sasfit_ff_q_exponential__area__v(scalar p, sasfit_param * param, int dist);
scalar sasfit_ff_q_gaussian__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_gaussian__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_q_gaussian__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_q_gaussian__area_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_gaussian__area__f(scalar q, sasfit_param * param);
scalar sasfit_ff_q_gaussian__area__v(scalar q, sasfit_param * param, int dist);
scalar eq(scalar q, scalar x);
scalar Cq(scalar q);
scalar lnq(scalar q, scalar x);
scalar sasfit_ff_q_weibull__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_weibull__ampl__f(scalar x, sasfit_param * param);
scalar sasfit_ff_q_weibull__ampl__v(scalar p, sasfit_param * param, int dist);
scalar sasfit_ff_q_weibull__area_(scalar x, sasfit_param * param);
scalar sasfit_ff_q_weibull__area__f(scalar q, sasfit_param * param);
scalar sasfit_ff_q_weibull__area__v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
scalar eq(scalar, scalar);
scalar Cq(scalar);
scalar lnq(scalar, scalar);
#endif
