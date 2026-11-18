/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define MW	param->p[1]
#define MN	param->p[2]
#define I0	param->p[3]

scalar sasfit_ff_gauspoly(scalar q, sasfit_param * param)
{
    scalar u,w,k, na;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((MW < 0.0), param, "Mw(%lg) < 0",MW); // modify condition to your needs
	SASFIT_CHECK_COND1((MN < 0.0), param, "Mn(%lg) < 0",MN); // modify condition to your needs

	// insert your code here
	na = 6.0221415e23;
	u = gsl_pow_2(q*RG);
	w = MW/MN-1.0;
	k = u/(1.0 + 2.0*w);

	if (k*(1.0+w) < 0.25) { // Taylor series: the closed form loses digits by cancellation for small k
		scalar term = (1.0+w)/2.0, sum = term;
		int n;
		for (n = 3; n < 60; n++) {
			term *= -(1.0+(n-1)*w)*k/n;
			sum += term;
			if (fabs(term) < 1e-17*fabs(sum)) break;
		}
		return I0*2.0*sum/(1.0+w);
	}
	if (w == 0) return I0*2.0*(exp(-k)+k-1.0)/(k*k);
	return I0*2.0*(exp(-log1p(w*k)/w)+k-1.0)/((1+w)*k*k);
}

scalar sasfit_ff_gauspoly_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_gauspoly_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

