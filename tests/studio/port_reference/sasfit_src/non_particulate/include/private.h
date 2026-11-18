/* shim: replaces the SASfit private.h of plugin non_particulate (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_NON_PARTICULATE_PRIVATE_H
#define SHIM_NON_PARTICULATE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_beaucage_exppowlaw(scalar q, sasfit_param * param);
scalar sasfit_ff_beaucage_exppowlaw_f(scalar q, sasfit_param * param);
scalar sasfit_ff_beaucage_exppowlaw_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_beaucage_exppowlaw_2(scalar q, sasfit_param * param);
scalar sasfit_ff_beaucage_exppowlaw_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_beaucage_exppowlaw_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_broad_peak(scalar q, sasfit_param * param);
scalar sasfit_ff_broad_peak_f(scalar q, sasfit_param * param);
scalar sasfit_ff_broad_peak_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_debye_anderson_brumberger(scalar q, sasfit_param * param);
scalar sasfit_ff_debye_anderson_brumberger_f(scalar q, sasfit_param * param);
scalar sasfit_ff_debye_anderson_brumberger_v(scalar q, sasfit_param * param, int dist);
scalar def_g_dab(scalar theta, sasfit_param * param);
scalar sasfit_ff_def_g_dab(scalar q, sasfit_param * param);
scalar sasfit_ff_def_g_dab_f(scalar q, sasfit_param * param);
scalar sasfit_ff_def_g_dab_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_extended_guinier_law(scalar q, sasfit_param * param);
scalar sasfit_ff_extended_guinier_law_f(scalar q, sasfit_param * param);
scalar sasfit_ff_extended_guinier_law_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_g_dab(scalar q, sasfit_param * param);
scalar sasfit_ff_g_dab_f(scalar q, sasfit_param * param);
scalar sasfit_ff_g_dab_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_generalized_guinier_porod_law(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_guinier_porod_law_f(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_guinier_porod_law_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ornstein_zernike(scalar q, sasfit_param * param);
scalar sasfit_ff_ornstein_zernike_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ornstein_zernike_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_spinodal(scalar q, sasfit_param * param);
scalar sasfit_ff_spinodal_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spinodal_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_spinodal2(scalar q, sasfit_param * param);
scalar sasfit_ff_spinodal2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spinodal2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_teubner_strey(scalar q, sasfit_param * param);
scalar sasfit_ff_teubner_strey_f(scalar q, sasfit_param * param);
scalar sasfit_ff_teubner_strey_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
