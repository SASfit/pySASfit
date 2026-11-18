/* shim: replaces the SASfit private.h of plugin polymer_stars (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_POLYMER_STARS_PRIVATE_H
#define SHIM_POLYMER_STARS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_benoitstar(scalar q, sasfit_param * param);
scalar sasfit_ff_benoitstar_f(scalar q, sasfit_param * param);
scalar sasfit_ff_benoitstar_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_broken_rods_star(scalar q, sasfit_param * param);
scalar sasfit_ff_broken_rods_star_f(scalar q, sasfit_param * param);
scalar sasfit_ff_broken_rods_star_v(scalar q, sasfit_param * param, int dist);
scalar xg(scalar x, sasfit_param * param);
scalar xf(scalar x, sasfit_param * param);
scalar phi(scalar x, sasfit_param *param);
scalar koyama_worm_Pkernel(scalar x, sasfit_param *param);
scalar koyama_worm_Fkernel(scalar x, sasfit_param *param);
scalar sasfit_ff_broken_worms_star(scalar q, sasfit_param * param);
scalar sasfit_ff_broken_worms_star_f(scalar q, sasfit_param * param);
scalar sasfit_ff_broken_worms_star_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_dozierstar(scalar q, sasfit_param * param);
scalar sasfit_ff_dozierstar_f(scalar q, sasfit_param * param);
scalar sasfit_ff_dozierstar_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_dozierstar2(scalar q, sasfit_param * param);
scalar sasfit_ff_dozierstar2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_dozierstar2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_gauss_exvol_star(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss_exvol_star_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss_exvol_star_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_polydispersestar(scalar q, sasfit_param * param);
scalar sasfit_ff_polydispersestar_f(scalar q, sasfit_param * param);
scalar sasfit_ff_polydispersestar_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_generalized_gaussian_coil(scalar, sasfit_param *);
scalar sasfit_ff_generalized_gaussian_coil_f(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
