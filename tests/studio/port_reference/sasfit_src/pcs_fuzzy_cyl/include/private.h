/* shim: replaces the SASfit private.h of plugin pcs_fuzzy_cyl (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_PCS_FUZZY_CYL_PRIVATE_H
#define SHIM_PCS_FUZZY_CYL_PRIVATE_H
#include <sasfit_shim.h>

scalar boucher2_cyl(scalar x, sasfit_param *param);
scalar boucher2_cyl_core(scalar x, sasfit_param *param);
scalar sasfit_ff_boucher2_cyl(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher2_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher2_cyl_v(scalar q, sasfit_param * param, int dist);
scalar boucher_cyl(scalar x, sasfit_param *param);
scalar boucher_cyl_core(scalar x, sasfit_param *param);
scalar sasfit_ff_boucher_cyl(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_cyl_v(scalar q, sasfit_param * param, int dist);
scalar fuzzy_cyl(scalar x, sasfit_param *param);
scalar fuzzy_cyl_core(scalar x, sasfit_param *param);
scalar sasfit_ff_fuzzy_cyl(scalar q, sasfit_param * param);
scalar sasfit_ff_fuzzy_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fuzzy_cyl_v(scalar q, sasfit_param * param, int dist);
scalar linsh_cyl(scalar x, sasfit_param *param);
scalar linsh_cyl_core(scalar x, sasfit_param *param);
scalar sasfit_ff_linsh_cyl(scalar q, sasfit_param * param);
scalar sasfit_ff_linsh_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_linsh_cyl_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile_boucher2_cyl(scalar r, sasfit_param * param);
scalar sasfit_ff_profile_boucher2_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile_boucher2_cyl_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile_boucher_cyl(scalar r, sasfit_param * param);
scalar sasfit_ff_profile_boucher_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile_boucher_cyl_v(scalar q, sasfit_param * param, int dist);
scalar profile_fuzzy_cyl_kernel(scalar s, sasfit_param *param);
scalar sasfit_ff_profile_fuzzy_cyl(scalar r, sasfit_param * param);
scalar sasfit_ff_profile_fuzzy_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile_fuzzy_cyl_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile_linsh_cyl(scalar r, sasfit_param * param);
scalar sasfit_ff_profile_linsh_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile_linsh_cyl_v(scalar q, sasfit_param * param, int dist);
scalar rc_exp_cyl(scalar x,sasfit_param * param);
scalar sasfit_ff_rc_exp_sh_cyl(scalar q, sasfit_param * param);
scalar sasfit_ff_rc_exp_sh_cyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rc_exp_sh_cyl_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_rc_exp_sh_cyl_profile(scalar r, sasfit_param * param);
scalar sasfit_ff_rc_exp_sh_cyl_profile_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rc_exp_sh_cyl_profile_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
