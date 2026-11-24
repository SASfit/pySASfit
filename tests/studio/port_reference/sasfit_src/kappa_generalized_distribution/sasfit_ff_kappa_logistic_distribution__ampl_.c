/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables

static scalar kl_neg(scalar s, sasfit_param *param) {
	return -sasfit_ff_kappa_logistic_distribution__area_(XI+SIGMA*exp(s),param);
}

static scalar kappa_logistic_mode(sasfit_param *param) {
	int i, k = 0;
	scalar s, v, best = GSL_POSINF, a, b, c, d, fc, fd, g = (sqrt(5.0)-1)/2;
	for (i = 0; i <= 300; i++) {
		s = -20.0 + 0.1*i;
		v = kl_neg(s,param);
		if (v < best) { best = v; k = i; }
	}
	if (k == 0) return XI;
	if (k == 300) return XI+SIGMA*exp(10.0);
	a = -20.0+0.1*(k-1); b = -20.0+0.1*(k+1);
	c = b-g*(b-a); d = a+g*(b-a); fc = kl_neg(c,param); fd = kl_neg(d,param);
	for (i = 0; i < 200 && b-a > 1e-13; i++) {
		if (fc < fd) { b = d; d = c; fd = fc; c = b-g*(b-a); fc = kl_neg(c,param); }
		else { a = c; c = d; fc = fd; d = a+g*(b-a); fd = kl_neg(d,param); }
	}
	return XI+SIGMA*exp(0.5*(a+b));
}

scalar sasfit_ff_kappa_logistic_distribution__ampl_(scalar x, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

    scalar z,mode,Norm;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((KAPPA < 0.0), param, "kappa(%lg) < 0",KAPPA); // modify condition to your needs
	SASFIT_CHECK_COND1((KAPPA >= 1.0), param, "kappa(%lg) >= 1",KAPPA); // modify condition to your needs
	SASFIT_CHECK_COND1((ALPHA <= 0.0), param, "alpha(%lg) <= 0",ALPHA); // modify condition to your needs
	SASFIT_CHECK_COND1((BETA <= 0.0), param, "beta(%lg) <= 0",BETA); // modify condition to your needs
	SASFIT_CHECK_COND1((LAMBDA <= 0.0), param, "lambda(%lg) <= 0",LAMBDA);


	// insert your code here

	z = (x-XI)/SIGMA;
	if (x<=XI) return 0.0;
	// maximum of the density (before 2026-11-24: mode_kappa, the mode of the kappa-generalised gamma distribution
	// with nu = lambda, i.e. a wrong point): coarse scan of log z, then golden section
	mode = kappa_logistic_mode(param);
	Norm = sasfit_ff_kappa_logistic_distribution__area_(mode,param);
//	sasfit_out("mode:%lf, Norm:%lf\n",mode,Norm);
	return A*sasfit_ff_kappa_logistic_distribution__area_(x,param)/Norm;
}

scalar sasfit_ff_kappa_logistic_distribution__ampl__f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_kappa_logistic_distribution__ampl__v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

