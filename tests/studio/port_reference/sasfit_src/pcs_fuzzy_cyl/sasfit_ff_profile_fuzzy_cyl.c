/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define R	fabs(param->p[0])
#define DUMMY1	param->p[1]
#define SIGMA	fabs(param->p[3])
#define DUMMY4	param->p[4]
#define DUMMY5	param->p[5]
#define DUMMY6	param->p[6]
#define ETA_CYL	param->p[7]
#define ETA_SOL	param->p[8]
#define DETA (ETA_CYL-ETA_SOL)

#define X   param->p[MAXPAR-1]

// disc of radius R convolved with a 2D Gaussian of width SIGMA, evaluated at distance X:
// integrand (s/sigma^2) exp(-(X-s)^2/(2 sigma^2)) I0_scaled(X s/sigma^2) over 0 <= s <= R
scalar profile_fuzzy_cyl_kernel(scalar s, sasfit_param *param) {
    return s/gsl_pow_2(SIGMA)*exp(-0.5*gsl_pow_2((X-s)/SIGMA))*gsl_sf_bessel_I0_scaled(X*s/gsl_pow_2(SIGMA));
}

scalar sasfit_ff_profile_fuzzy_cyl(scalar r, sasfit_param * param)
{
    scalar a, b;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((param->p[0] < 0.0), param, "R(%lg) < 0",param->p[0]); // modify condition to your needs

	// insert your code here
	X = fabs(r);
    if (SIGMA == 0) {
        if (X<=R) {
            return ETA_CYL;
        } else {
            return ETA_SOL;
        }
    }
    a = GSL_MAX(0.0, X-12*SIGMA);
    b = GSL_MIN(R, X+12*SIGMA);
    if (b <= a) return ETA_SOL;
    return sasfit_integrate(a, b, &profile_fuzzy_cyl_kernel, param)*DETA + ETA_SOL;
}

scalar sasfit_ff_profile_fuzzy_cyl_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0;
}

scalar sasfit_ff_profile_fuzzy_cyl_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0;
}

