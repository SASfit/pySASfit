/* shim: replaces the SASfit private.h of plugin polymer_semi_flexible (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_POLYMER_SEMI_FLEXIBLE_PRIVATE_H
#define SHIM_POLYMER_SEMI_FLEXIBLE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_freely_joined_chain_of_rods(scalar q, sasfit_param * param);
scalar sasfit_ff_freely_joined_chain_of_rods_f(scalar q, sasfit_param * param);
scalar sasfit_ff_freely_joined_chain_of_rods_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_kholodenko_worm(scalar q, sasfit_param * param);
scalar sasfit_ff_kholodenko_worm_f(scalar q, sasfit_param * param);
scalar sasfit_ff_kholodenko_worm_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worms_ps1_(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps1__f(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps1__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worms_ps2_(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps2__f(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps2__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worms_ps3_(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps3__f(scalar q, sasfit_param * param);
scalar sasfit_ff_worms_ps3__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_koyama_worm(scalar q, sasfit_param * param);
scalar sasfit_ff_koyama_worm_f(scalar q, sasfit_param * param);
scalar sasfit_ff_koyama_worm_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_koyama_worms(scalar, sasfit_param *);
scalar sasfit_sq_freely_joined_chain_of_rods(scalar, sasfit_param *);
scalar sasfit_sq_p__q___kholodenko_worm(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps1_(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps2_(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps3_(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
