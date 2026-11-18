/* shim: replaces the SASfit private.h of plugin fractal_obj (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_FRACTAL_OBJ_PRIVATE_H
#define SHIM_FRACTAL_OBJ_PRIVATE_H
#include <sasfit_shim.h>

scalar MassFractExp_mpow_x_a_core(scalar x, sasfit_param *param);
scalar sasfit_ff_aggregate__exp__x_a__cut_off_(scalar q, sasfit_param * param);
scalar sasfit_ff_aggregate__exp__x_a__cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_ff_aggregate__exp__x_a__cut_off__v(scalar q, sasfit_param * param, int dist);
scalar MassFractOverlappingSpheres_core(scalar x, sasfit_param * param);
scalar sasfit_ff_aggregate__overlapsph_cut_off_(scalar q, sasfit_param * param);
scalar sasfit_ff_aggregate__overlapsph_cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_ff_aggregate__overlapsph_cut_off__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_dlcaggregation(scalar q, sasfit_param * param);
scalar sasfit_ff_dlcaggregation_f(scalar q, sasfit_param * param);
scalar sasfit_ff_dlcaggregation_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_doubleshellchain(scalar q, sasfit_param * param);
scalar sasfit_ff_doubleshellchain_f(scalar q, sasfit_param * param);
scalar sasfit_ff_doubleshellchain_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_dumbbellshell(scalar q, sasfit_param * param);
scalar sasfit_ff_dumbbellshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_dumbbellshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_fisher_burford(scalar q, sasfit_param * param);
scalar sasfit_ff_fisher_burford_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fisher_burford_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_massfractexp(scalar q, sasfit_param * param);
scalar sasfit_ff_massfractexp_f(scalar q, sasfit_param * param);
scalar sasfit_ff_massfractexp_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_massfractgauss(scalar q, sasfit_param * param);
scalar sasfit_ff_massfractgauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_massfractgauss_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_rlcaggregation(scalar q, sasfit_param * param);
scalar sasfit_ff_rlcaggregation_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rlcaggregation_v(scalar q, sasfit_param * param, int dist);
scalar StackDiscs_core(scalar theta, sasfit_param * param);
scalar sasfit_ff_stackdiscs(scalar q, sasfit_param * param);
scalar sasfit_ff_stackdiscs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_stackdiscs_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_tetrahedrondoubleshell(scalar q, sasfit_param * param);
scalar sasfit_ff_tetrahedrondoubleshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_tetrahedrondoubleshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_two_attached_spheres(scalar q, sasfit_param * param);
scalar sasfit_ff_two_attached_spheres_f(scalar q, sasfit_param * param);
scalar sasfit_ff_two_attached_spheres_v(scalar q, sasfit_param * param, int dist);
scalar two_attached_spheres_v(const double *x, size_t ndim, void *pam);
scalar two_attached_spheres_f(const double *x, size_t ndim, void *pam);
scalar two_attached_spheres(const double *x, size_t ndim, void *pam);
scalar sasfit_ff_two_attached_spheres_w_distr(scalar q, sasfit_param * param);
scalar sasfit_ff_two_attached_spheres_w_distr_f(scalar q, sasfit_param * param);
scalar sasfit_ff_two_attached_spheres_w_distr_v(scalar q, sasfit_param * param, int dist);
scalar sphere_f(scalar q, scalar R, scalar Deta);
scalar sq_MassFractExp_mpow_x_a_core(scalar x, sasfit_param * param);
scalar sasfit_sq_MassFractExp_mpow_x_a(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__exp__x_a__cut_off_(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__exp__x_a__cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__exp__x_a__cut_off__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_mass_fractal__exp_cut_off_(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__exp_cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__exp_cut_off__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_mass_fractal__gaussian_cut_off_(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__gaussian_cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__gaussian_cut_off__v(scalar q, sasfit_param * param, int dist);
scalar sq_MassFractOverlapSph_core(scalar x, sasfit_param * param);
scalar sasfit_sq_mass_fractal__overapsph_cut_off_(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__overapsph_cut_off__f(scalar q, sasfit_param * param);
scalar sasfit_sq_mass_fractal__overapsph_cut_off__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
