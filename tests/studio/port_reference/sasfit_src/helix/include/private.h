/* shim: replaces the SASfit private.h of plugin helix (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_HELIX_PRIVATE_H
#define SHIM_HELIX_PRIVATE_H
#include <sasfit_shim.h>

scalar hphi(scalar q, sasfit_param * param);
scalar hPsi2(scalar q, int j, sasfit_param * param);
scalar sasfit_ff_beads_helix(scalar q, sasfit_param * param);
scalar sasfit_ff_beads_helix_f(scalar q, sasfit_param * param);
scalar sasfit_ff_beads_helix_v(scalar q, sasfit_param * param, int dist);
scalar bessfct(scalar r, sasfit_param * param);
scalar gn2(scalar q, int n, sasfit_param * param);
scalar prefct(int n, sasfit_param * param);
scalar sasfit_ff_fan_helix(scalar q, sasfit_param * param);
scalar sasfit_ff_fan_helix_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fan_helix_v(scalar x, sasfit_param * param, int dist);
scalar twoJ1x_x(scalar x);
scalar sasfit_ff_round_helix(scalar q, sasfit_param * param);
scalar sasfit_ff_round_helix_f(scalar q, sasfit_param * param);
scalar sasfit_ff_round_helix_v(scalar x, sasfit_param * param, int dist);
void pitchangle(sasfit_param * param);
scalar r2(scalar g1, scalar g2, void * pam);
scalar sqrt_f(scalar gamma, sasfit_param *param);
scalar f2D_cubature(const double *x, size_t ndim, void *pam);
scalar sasfit_ff_superhelices_coiled(scalar q, sasfit_param * param);
scalar sasfit_ff_superhelices_coiled_f(scalar q, sasfit_param * param);
scalar sasfit_ff_superhelices_coiled_v(scalar q, sasfit_param * param, int dist);
scalar Kernel_w(scalar w, sasfit_param *param);
scalar sasfit_ff_superhelices_straight(scalar q, sasfit_param * param);
scalar sasfit_ff_superhelices_straight_f(scalar q, sasfit_param * param);
scalar sasfit_ff_superhelices_straight_v(scalar q, sasfit_param * param, int dist);
scalar sinc(scalar x);
scalar thinrod_helix(scalar q, scalar L);

//
// add local defines here:
// #define P0 param->p[0]
//
#define NMAX 200

scalar thinrod_helix(scalar, scalar);
scalar sinc(scalar);
#endif
