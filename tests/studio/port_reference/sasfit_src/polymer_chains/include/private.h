/* shim: replaces the SASfit private.h of plugin polymer_chains (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_POLYMER_CHAINS_PRIVATE_H
#define SHIM_POLYMER_CHAINS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_gauspoly(scalar q, sasfit_param * param);
scalar sasfit_ff_gauspoly_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gauspoly_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_gauss(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_gauss2(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_gauss3(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_gauss3_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_generalized_gaussian_coil(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_f(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_generalized_gaussian_coil_2(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_2_v(scalar x, sasfit_param * param, int dist);
scalar sasfit_ff_generalized_gaussian_coil_3(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_generalized_gaussian_coil_3_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//

/* u^-a gamma(a,u) = sum_n (-u)^n/(n! (a+n)) (series for u < 2, else via the regularised incomplete gamma
 * function P(a,u)); no cancellation for small u (the forms Gamma(a)-Gamma(a,u) lose all digits there) */
static scalar lower_gamma_ratio(scalar a, scalar u) {
	scalar term, s;
	int n;
	if (u <= 0) return 1.0/a;
	if (u < 2.0) {
		term = 1.0;
		s = 1.0/a;
		for (n = 1; n < 60; n++) {
			term *= -u/n;
			s += term/(a+n);
			if (fabs(term/(a+n)) < 1e-17*fabs(s)) break;
		}
		return s;
	}
	return exp(gsl_sf_lngamma(a) - a*log(u))*gsl_sf_gamma_inc_P(a,u);
}

/* 2(u-1+exp(-u))/u^2 and (1-exp(-u))/u without cancellation for small u */
static scalar debye_stable(scalar u) {
	scalar term, s;
	int n;
	if (u < 0.05) {
		term = 1.0; s = 1.0;
		for (n = 1; n < 12; n++) { term *= -u/(n+2); s += term; }
		return s;
	}
	return 2.0*(u+expm1(-u))/(u*u);
}
static scalar debye_amplitude(scalar u) {
	if (u < 1e-8) return 1.0 - u/2.0;
	return -expm1(-u)/u;
}
#endif
