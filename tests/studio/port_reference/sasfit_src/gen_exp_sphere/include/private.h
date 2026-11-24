/* shim: replaces the SASfit private.h of plugin gen_exp_sphere (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_GEN_EXP_SPHERE_PRIVATE_H
#define SHIM_GEN_EXP_SPHERE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_gen__exp__sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_gen__exp__sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gen__exp__sphere_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_gen__gauss__sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_gen__gauss__sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gen__gauss__sphere_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile__gen__exp__sphere(scalar r, sasfit_param * param);
scalar sasfit_ff_profile__gen__exp__sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile__gen__exp__sphere_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_profile__gen__gauss__sphere(scalar r, sasfit_param * param);
scalar sasfit_ff_profile__gen__gauss__sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_profile__gen__gauss__sphere_v(scalar q, sasfit_param * param, int dist);

#endif
