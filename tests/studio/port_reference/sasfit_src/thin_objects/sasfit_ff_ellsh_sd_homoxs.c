/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define T	param->p[0]
#define SIGMA_T	fabs(param->p[1])
#define R0	param->p[2]
#define SIGMA_R0	fabs(param->p[3])
#define EPSILON	param->p[4]
#define DUMMY	param->p[5]
#define ETA_L	param->p[6]
#define ETA_SOL	param->p[7]

scalar sasfit_ff_ellsh_sd_homoxs(scalar q, sasfit_param * param)
{
	scalar Pcs, Pprime;
	sasfit_param subParam;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((T < 0.0), param, "t(%lg) < 0",T); // modify condition to your needs
	SASFIT_CHECK_COND1((R0 < 0.0), param, "R0(%lg) < 0",R0); // modify condition to your needs
	SASFIT_CHECK_COND1((SIGMA_R0 < 0.0), param, "sigma_R0(%lg) < 0",SIGMA_R0); // modify condition to your needs
	SASFIT_CHECK_COND1((EPSILON <= 0.0), param, "epsilon(%lg) <= 0",EPSILON); // modify condition to your needs

	// insert your code here
	sasfit_init_param( &subParam );
	subParam.p[0] =	T;
	subParam.p[1] =	SIGMA_T;
	subParam.p[4] =	ETA_L;
	subParam.p[5] =	ETA_SOL;
	Q=q;
	P=2;

	Pcs = sasfit_ff_pcs_homogeneousplate(q,&subParam);

	sasfit_init_param( &subParam );
	subParam.p[0] = R0;
	subParam.p[1] = EPSILON;
	subParam.p[2] = SIGMA_R0;

	Pprime = sasfit_sq_p__q___thin_ellipsoidal_shell(q,&subParam);
	return Pcs*Pprime;
}

scalar sasfit_ff_ellsh_sd_homoxs_f(scalar q, sasfit_param * param)
{
	scalar Pcs, Pprime;
	sasfit_param subParam;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code
	sasfit_init_param( &subParam );
	subParam.p[0] =	T;
	subParam.p[1] =	SIGMA_T;
	subParam.p[4] =	ETA_L;
	subParam.p[5] =	ETA_SOL;
	Q=q;
	P=1;

	Pcs = sasfit_ff_pcs_homogeneousplate_f(q,&subParam);

	sasfit_init_param( &subParam );
	subParam.p[0] = R0;
	subParam.p[1] = EPSILON;
	subParam.p[2] = SIGMA_R0;

	Pprime = sasfit_sq_p__q___thin_ellipsoidal_shell_f(q,&subParam);
	return Pcs*Pprime;
}

scalar sasfit_ff_ellsh_sd_homoxs_v(scalar q, sasfit_param * param, int dist)
{
	scalar v[5], S, ae;
	int k;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid
	// 2026-11-24: mean shell volume <S(R)> t = S(R0, epsilon) exp(2 sigma_R0^2) |t| with p[dist] replaced by q
	// (before: 0)
	for (k = 0; k < 5; k++) v[k] = param->p[k];
	if (dist >= 0 && dist < 5) v[dist] = q;
	if (v[4] < 1.) {
		ae = acos(v[4]);
		S = 2.*M_PI*(1.+v[4]*gsl_atanh(sin(ae))/sin(ae));
	} else if (v[4] > 1.) {
		ae = acos(1./v[4]);
		S = 2.*M_PI*(1.+v[4]*ae/tan(ae));
	} else {
		S = 4.*M_PI;
	}
	return S*v[2]*v[2]*exp(2.0*v[3]*v[3])*fabs(v[0]);
}

