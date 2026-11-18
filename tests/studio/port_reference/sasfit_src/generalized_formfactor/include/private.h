/* shim: replaces the SASfit private.h of plugin generalized_formfactor (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_GENERALIZED_FORMFACTOR_PRIVATE_H
#define SHIM_GENERALIZED_FORMFACTOR_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_generalized_form_factor(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_form_factor_f(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_form_factor_v(scalar q, sasfit_param * param, int dist);
scalar lesser_gFF(scalar x, sasfit_param *param);
scalar lesser_gFF_core(scalar x, sasfit_param *param);
scalar sasfit_ff_lesser_generalized_form_factor(scalar q, sasfit_param * param);
scalar sasfit_ff_lesser_generalized_form_factor_f(scalar q, sasfit_param * param);
scalar sasfit_ff_lesser_generalized_form_factor_v(scalar q, sasfit_param * param, int dist);
scalar lesser_gFF2(scalar x, sasfit_param *param);
scalar lesser_gFF2_core(scalar x, sasfit_param *param);
scalar sasfit_ff_lesser_generalized_form_factor2(scalar q, sasfit_param * param);
scalar sasfit_ff_lesser_generalized_form_factor2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_lesser_generalized_form_factor2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
