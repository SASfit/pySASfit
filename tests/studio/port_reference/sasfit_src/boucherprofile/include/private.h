/* shim: replaces the SASfit private.h of plugin boucherprofile (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_BOUCHERPROFILE_PRIVATE_H
#define SHIM_BOUCHERPROFILE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_boucher_profiles(scalar r, sasfit_param * param);
scalar sasfit_ff_boucher_profiles_f(scalar r, sasfit_param * param);
scalar sasfit_ff_boucher_profiles_v(scalar r, sasfit_param * param, int dist);
scalar sasfit_ff_boucher_profiles2(scalar r, sasfit_param * param);
scalar sasfit_ff_boucher_profiles2_f(scalar r, sasfit_param * param);
scalar sasfit_ff_boucher_profiles2_v(scalar r, sasfit_param * param, int dist);
scalar sasfit_ff_boucher_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_sphere_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_boucher_sphere2(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_sphere2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boucher_sphere2_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
