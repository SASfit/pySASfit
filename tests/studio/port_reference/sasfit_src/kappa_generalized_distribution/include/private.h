/* shim: replaces the SASfit private.h of plugin kappa_generalized_distribution (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_KAPPA_GENERALIZED_DISTRIBUTION_PRIVATE_H
#define SHIM_KAPPA_GENERALIZED_DISTRIBUTION_PRIVATE_H
#include <sasfit_shim.h>

scalar exp_kappa(scalar x, sasfit_param * param);
scalar ln_kappa(scalar x, sasfit_param * param);
scalar erf_kernel(scalar x, sasfit_param * param);
double erf1(double x);
scalar erf_kappa(scalar x, sasfit_param * param);
scalar mode_kappa(sasfit_param * param);
scalar Z_kappa(sasfit_param * param);
scalar sasfit_ff_kappa_generalized_gamma_distribution__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_kappa_generalized_gamma_distribution__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_kappa_generalized_gamma_distribution__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_kappa_generalized_gamma_distribution__area_(scalar x, sasfit_param * param);
scalar sasfit_ff_kappa_generalized_gamma_distribution__area__f(scalar q, sasfit_param * param);
scalar sasfit_ff_kappa_generalized_gamma_distribution__area__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_kappa_logistic_distribution__ampl_(scalar x, sasfit_param * param);
scalar sasfit_ff_kappa_logistic_distribution__ampl__f(scalar q, sasfit_param * param);
scalar sasfit_ff_kappa_logistic_distribution__ampl__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_kappa_logistic_distribution__area_(scalar x, sasfit_param * param);
scalar sasfit_ff_kappa_logistic_distribution__area__f(scalar q, sasfit_param * param);
scalar sasfit_ff_kappa_logistic_distribution__area__v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
#define A	param->p[0]
#define SIGMA param->p[1]
#define XI  param->p[2]
#define KAPPA	param->p[3]
#define ALPHA	param->p[4]
#define BETA	param->p[5]
#define NU	param->p[6]
#define LAMBDA	param->p[6]
//
scalar exp_kappa(scalar, sasfit_param *);
scalar mode_kappa(sasfit_param *);
scalar erf_kappa(scalar, sasfit_param *);
#endif
