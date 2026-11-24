/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define A	param->p[0]
#define Q	param->p[1]
#define BETA	param->p[2]
#define DUMMY	param->p[3]
#define SHIFT	param->p[4]
#define Y0	param->p[5]

scalar sasfit_ff_q_gaussian__ampl_(scalar x, sasfit_param * param)
{
    // https://en.wikipedia.org/wiki/Q-Gaussian_distribution
    /*
     *   THIS IS THE PROBABILITY DISTRIBUTION FUNCTION
     */
    scalar z;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((Q >= 3.0), param, "q(%lg) >= 3",Q); // modify condition to your needs
	SASFIT_CHECK_COND1((BETA == 0.0), param, "beta(%lg) == 0",BETA); // modify condition to your needs


	// insert your code here
	z=x-SHIFT;
	if (A==0) return Y0; // symmetric: no cut at z < 0 (before 2026-11-24 the left half was y0)
	return A*(sasfit_ff_q_gaussian__area_(x,param)-Y0)/(sasfit_ff_q_gaussian__area_(SHIFT,param)-Y0)+Y0; // maximum at x = shift
}

scalar sasfit_ff_q_gaussian__ampl__f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_q_gaussian__ampl__v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

