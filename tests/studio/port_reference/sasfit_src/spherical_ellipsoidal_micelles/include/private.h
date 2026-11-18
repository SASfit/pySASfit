/* shim: replaces the SASfit private.h of plugin spherical_ellipsoidal_micelles (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_SPHERICAL_ELLIPSOIDAL_MICELLES_PRIVATE_H
#define SHIM_SPHERICAL_ELLIPSOIDAL_MICELLES_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_ell_chains_rw_(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__f(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ell_chains_rw__nagg(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__nagg_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__nagg_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ell_chains_rw__rc(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__rc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ell_chains_rw__rc_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_2nd_r__a_rc(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_2nd_r__a_rc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_2nd_r__a_rc_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_rw_(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_rw__nagg(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__nagg_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__nagg_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_rw__rc(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__rc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_rw__rc_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_saw_(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_saw__nagg(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__nagg_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__nagg_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_chains_saw__rc(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__rc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_chains_saw__rc_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_r__a(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_r__a_nagg(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_nagg_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_nagg_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_r__a_rc(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_rc_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_r__a_rc_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_sphere_smooth_interface_r__a_nagg(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_smooth_interface_r__a_nagg_f(scalar q, sasfit_param * param);
scalar sasfit_ff_sphere_smooth_interface_r__a_nagg_v(scalar q, sasfit_param * param, int dist);
scalar XI(scalar Q, scalar R, scalar H, scalar alpha);
scalar sasfit_ff_P18(scalar Q, scalar Rg);
scalar f1(scalar q, scalar r);
scalar r_ell(scalar r, scalar epsilon, scalar alpha);
scalar sasfit_ff_ellip_rwbrush_ssc_core(scalar alpha, sasfit_param *param);
scalar sasfit_ff_ellip_rwbrush_scc_core(scalar alpha, sasfit_param *param);
scalar sasfit_ff_ellip_rwbrush_fs_core(scalar alpha, sasfit_param *param);
scalar sasfit_ff_ellip_rwbrush(scalar q, sasfit_param * param);
scalar sasfit_ff_ellip_rwbrush_rc(scalar q, sasfit_param * param);
scalar sasfit_ff_ellip_rwbrush_nagg(scalar q, sasfit_param * param);
scalar Fone(scalar q, scalar R);
scalar V(scalar R);
scalar Sg(scalar q, scalar Rc, scalar s);
scalar Ac(scalar q, scalar Rc, scalar R, scalar s);
scalar Scc(scalar q, scalar Rc, scalar R, scalar s);
scalar Ssc(scalar q, scalar Rc, scalar R, scalar s);
scalar Ptwentyone(scalar x);
scalar sasfit_Pthirtynine(scalar q, sasfit_param * param);
scalar sasfit_ff_SphereWithGaussChains(scalar q, sasfit_param * param);
scalar sasfit_ff_Sphere_RWbrush(scalar q, sasfit_param * param);
scalar sasfit_ff_Sphere_R_ma_Profile_ave_core(scalar x, sasfit_param * param);
scalar sasfit_ff_Sphere_R_ma_Profile(scalar q, sasfit_param * param);
scalar sasfit_ff_Sphere_R_ma_Profile_v(scalar q, sasfit_param * param, int distr);
scalar fRg2(scalar cl,scalar lb);
scalar falpha2(scalar cl, scalar lb);
scalar sasfit_sphere_SAWbrush(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps3_(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
scalar sasfit_ff_Sphere_RWbrush(scalar, sasfit_param *);
scalar sasfit_ff_Sphere_R_ma_Profile(scalar, sasfit_param *);
scalar sasfit_ff_ellip_rwbrush(scalar, sasfit_param *);
scalar sasfit_ff_ellip_rwbrush_rc(scalar, sasfit_param *);
scalar sasfit_ff_ellip_rwbrush_nagg(scalar, sasfit_param *);
scalar sasfit_sphere_SAWbrush(scalar, sasfit_param *);
#endif
