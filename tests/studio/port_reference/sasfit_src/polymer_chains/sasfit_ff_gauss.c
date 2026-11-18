/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define DUMMY	param->p[1]
#define I0	param->p[2]

scalar sasfit_ff_gauss(scalar q, sasfit_param * param)
{
    scalar u;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);

	// insert your code here
	u = gsl_pow_2(q*RG);
	return I0*debye_stable(u); // series for small u (the closed form loses digits there)
}

scalar sasfit_ff_gauss_f(scalar q, sasfit_param * param)
{
    scalar u;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	u = gsl_pow_2(q*RG);
	return sqrt(I0)*debye_amplitude(u); // (1-exp(-u))/u (was (1-exp(+u))/u)
}

scalar sasfit_ff_gauss_v(scalar q, sasfit_param * param, int dist)
{
    scalar rho;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	rho=8/3/M_SQRTPI;
	return 4/3.*M_PI*gsl_pow_3(RG/rho);
}

