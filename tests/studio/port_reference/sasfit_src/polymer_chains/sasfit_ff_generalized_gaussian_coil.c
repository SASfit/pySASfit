/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define NU	param->p[1]
#define I0	fabs(param->p[2])

scalar sasfit_ff_generalized_gaussian_coil(scalar q, sasfit_param * param)
{
    scalar u;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((NU <= 0.0), param, "nu(%lg) <= 0",NU); // modify condition to your needs

	// insert your code here
    u = (2.0*NU+1.0)*(2.0*NU+2.0)*q*q*RG*RG/6.0;
    // exact for all u: (1/nu)[u^-a gamma(a,u) - u^-2a gamma(2a,u)], a = 1/(2 nu) (was exp(-q^2Rg^2/3) for q Rg < 0.1)
    return I0*(lower_gamma_ratio(1.0/(2.*NU),u) - lower_gamma_ratio(1.0/NU,u))/NU;
}

scalar sasfit_ff_generalized_gaussian_coil_f(scalar q, sasfit_param * param)
{
    scalar u;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	 u = (2.0*NU+1.0)*(2.0*NU+2.0)*q*q*RG*RG/6.0;

	return sqrt(I0)*lower_gamma_ratio(1.0/(2.*NU),u)/(2*NU);
}

scalar sasfit_ff_generalized_gaussian_coil_v(scalar x, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return gsl_pow_3(x);
}

