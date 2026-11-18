/* shim: replaces the SASfit private.h of plugin sphericalshells (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_SPHERICALSHELLS_PRIVATE_H
#define SHIM_SPHERICALSHELLS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_bilayered_vesicle(scalar q, sasfit_param * param);
scalar sasfit_ff_bilayered_vesicle_f(scalar q, sasfit_param * param);
scalar sasfit_ff_bilayered_vesicle_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_hollow_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_hollow_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_hollow_sphere_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_mlv_frielinghaus(scalar q, sasfit_param * param);
scalar sasfit_ff_mlv_frielinghaus_f(scalar q, sasfit_param * param);
scalar sasfit_ff_mlv_frielinghaus_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_multilamellar_vesicle(scalar q, sasfit_param * param);
scalar sasfit_ff_multilamellar_vesicle_f(scalar q, sasfit_param * param);
scalar sasfit_ff_multilamellar_vesicle_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_rnd_multilamellar_vesicle(scalar q, sasfit_param * param);
scalar sasfit_ff_rnd_multilamellar_vesicle_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rnd_multilamellar_vesicle_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_rnd_multilamellar_vesicle_2(scalar q, sasfit_param * param);
scalar sasfit_ff_rnd_multilamellar_vesicle_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rnd_multilamellar_vesicle_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_spherical_shell_i(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_i_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_i_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_spherical_shell_ii(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_ii_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_ii_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_spherical_shell_iii(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_iii_f(scalar q, sasfit_param * param);
scalar sasfit_ff_spherical_shell_iii_v(scalar x, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
