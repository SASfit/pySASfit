/* shim: replaces the SASfit private.h of plugin pearl_necklace (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_PEARL_NECKLACE_PRIVATE_H
#define SHIM_PEARL_NECKLACE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_pearl_necklace_ext(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_ext_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_ext_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_pearl_necklace_mod(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_mod_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_mod_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_pearl_necklace(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pearl_necklace_v(scalar q, sasfit_param * param, int dist);
scalar Lambda(scalar qb, sasfit_param *param);
scalar LambdaExt(scalar qR, scalar qL, sasfit_param *param);
scalar Psi(scalar x);
scalar pn_sum_u(int m, scalar g);
scalar pn_sum_geo(int m, scalar g);
scalar pn_geo_real(scalar x, int j, scalar s);

//
// add local defines here:
// #define P0 param->p[0]
//

scalar Lambda(scalar, sasfit_param *);
scalar LambdaExt(scalar, scalar, sasfit_param *);
scalar Psi(scalar);
scalar pn_sum_u(int, scalar);
scalar pn_sum_geo(int, scalar);
scalar pn_geo_real(scalar, int, scalar);
#endif
