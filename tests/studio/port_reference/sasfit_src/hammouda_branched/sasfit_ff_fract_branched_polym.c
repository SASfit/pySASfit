/*
 * Author(s) of this file:
 *   <your name> (<email address>)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define VM	param->p[1]
#define C	param->p[2]
#define I0	param->p[3]


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

scalar sasfit_ff_fract_branched_polym(scalar q, sasfit_param * param)
{
	scalar ub, norm_inv;
	scalar (*gamma) (scalar);
	scalar (*gamma_inc) (scalar, scalar);
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((VM <= 0.0), param, "vm(%lg) < 0",VM); // modify condition to your needs
	SASFIT_CHECK_COND1((C <= 0.0), param, "c(%lg) < 0",C); // modify condition to your needs

	// insert your code here
	ub = q*q * RG*RG * (2.*VM+C) *(2.*VM+C+1.)/6.;
	norm_inv = I0*0.5*(C*C+C);
	gamma=gsl_sf_gamma;
	gamma_inc=gsl_sf_gamma_inc;
	// exact for all q (was exp(-q^2Rg^2/3) for q Rg < 0.01 and Gamma-Gamma_inc, which loses all digits for small ub)
	return (lower_gamma_ratio(0.5*C/VM,ub) - lower_gamma_ratio(0.5*(C+1.)/VM,ub))/VM * norm_inv;
}

scalar sasfit_ff_fract_branched_polym_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_fract_branched_polym_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

