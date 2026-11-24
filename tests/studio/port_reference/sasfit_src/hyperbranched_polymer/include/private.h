/* shim: replaces the SASfit private.h of plugin hyperbranched_polymer (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_HYPERBRANCHED_POLYMER_PRIVATE_H
#define SHIM_HYPERBRANCHED_POLYMER_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_comb_hetero(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_hetero_f(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_hetero_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_comb_random(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_random_f(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_random_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_comb_regular(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_regular_f(scalar q, sasfit_param * param);
scalar sasfit_ff_comb_regular_v(scalar q, sasfit_param * param, int dist);
scalar Paf_Gauss(scalar x, sasfit_param * param);
scalar Paf_exact(scalar x, sasfit_param * param);
scalar sasfit_ff_f_funct_homo_polym(scalar q, sasfit_param * param);
scalar sasfit_ff_f_funct_homo_polym_f(scalar q, sasfit_param * param);
scalar sasfit_ff_f_funct_homo_polym_v(scalar q, sasfit_param * param, int dist);
scalar P_linfract(scalar u, sasfit_param * param);
scalar P_DBfract(scalar u, sasfit_param * param);
scalar sasfit_ff_fract_hyperbranched_polym(scalar q, sasfit_param * param);
scalar sasfit_ff_fract_hyperbranched_polym_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fract_hyperbranched_polym_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ideal_dendrimer(scalar q, sasfit_param * param);
scalar sasfit_ff_ideal_dendrimer_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ideal_dendrimer_v(scalar q, sasfit_param * param, int dist);

/*
 * Positive auxiliary functions of the comb and dendrimer form factors (2026-11-24): with them the closed forms
 * need no subtraction of O(x^2) terms to get O(x^4) results (the direct forms lost digits for small q R).
 *   hbp_g(z)  = (exp(-z) - 1 + z)/z^2     (1/2 at 0)
 *   hbp_h(z)  = (1 - exp(-z))/z           (1 at 0)
 *   hbp_gplus(c, &r1, &r2, &inv): r1 = G+(c)/H+(c), r2 = G+(c)/H+(c)^2, inv = 1/H+(c) with
 *   G+(c) = (exp(c) - 1 - c)/c^2, H+(c) = (exp(c) - 1)/c, without overflow for large c
 */
static scalar hbp_g(scalar z)
{
	scalar s, t;
	int n;
	if (fabs(z) < 0.1) {
		s = 0.0; t = 0.5;
		for (n = 1; n < 14; n++) { s += t; t = -t*z/(n+2); }
		return s;
	}
	return (expm1(-z) + z)/(z*z);
}
static scalar hbp_h(scalar z)
{
	if (z == 0.0) return 1.0;
	return -expm1(-z)/z;
}
static void hbp_gplus(scalar c, scalar *r1, scalar *r2, scalar *inv)
{
	scalar s, t, hp, em;
	int n;
	if (c < 0.1) {
		s = 0.0; t = 0.5;
		for (n = 1; n < 14; n++) { s += t; t = t*c/(n+2); }
		hp = (c == 0.0) ? 1.0 : expm1(c)/c;
		*r1 = s/hp; *r2 = s/(hp*hp); *inv = 1.0/hp;
		return;
	}
	em = exp(-c);
	*r1 = (1.0 - (1.0 + c)*em)/(c*(-expm1(-c)));
	*r2 = (em - (1.0 + c)*em*em)/gsl_pow_2(expm1(-c));
	*inv = c*em/(-expm1(-c));
}


//
// add local defines here:
// #define P0 param->p[0]
//
#endif
