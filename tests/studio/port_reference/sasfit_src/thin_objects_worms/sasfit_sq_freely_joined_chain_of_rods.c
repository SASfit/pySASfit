/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_sq.h>

// define shortcuts for local parameters/variables

static scalar fjc_one_minus_sinc(scalar x)
// 1 - sin(x)/x, series below 0.5
{
	scalar x2, t, s;
	int k;
	if (fabs(x) >= 0.5) return 1.0 - sin(x)/x;
	x2 = x*x;
	t = x2/6.0;
	s = 0.0;
	for (k = 1; k < 12; k++) {
		s += t;
		t *= -x2/((2.0*k+2.0)*(2.0*k+3.0));
	}
	return s;
}

static scalar fjc_h(scalar delta, scalar gmma, scalar n)
// [n delta - 1 + gmma^n]/(n delta^2), delta = 1 - gmma: series in delta for small n delta,
// gmma^n = |gmma|^n cos(pi n) for gmma < 0 (equal to gmma^n for integer n)
{
	scalar c, s, pw, gn;
	int k;
	if (fabs(delta)*GSL_MAX(n,1.0) < 0.5 && fabs(delta) < 0.5) {
		c = n*(n-1.0)/2.0;
		s = 0.0;
		pw = 1.0;
		for (k = 2; k < 70; k++) {
			s += c*pw;
			c *= (n-k)/(k+1.0);
			pw *= -delta;
		}
		return s/n;
	}
	if (gmma > 0) {
		gn = exp(n*log(gmma));
	} else {
		gn = pow(fabs(gmma),n)*cos(M_PI*n);
	}
	return (n*delta - 1.0 + gn)/(n*delta*delta);
}

scalar sasfit_sq_freely_joined_chain_of_rods(scalar q, sasfit_param * param)
{
	scalar gmma, Gmma, cB, cC, qb, n, delta;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((LB <= 0.0), param, "lb(%lg) <= 0",LB); // modify condition to your needs
	SASFIT_CHECK_COND1((L <= 0.0), param, "L(%lg) <= 0",L); // modify condition to your needs

	// 2026-11-23: the closed form cancelled for small qb (100 % at qb = 1e-5, the series was used only below 1e-6
	// and gave S(0) = 1 - 2/(3n)); pow(gmma, L/LB) was NaN for gmma < 0 and non-integer L/LB.
	n = L/LB;
	qb = q*LB;
	if (qb == 0.0) return L*L;
	gmma = sin(qb)/qb;
	Gmma = gsl_sf_Si(qb)/qb;
	delta = fjc_one_minus_sinc(qb);
	cB = 2.0/n*Gmma*Gmma*fjc_h(delta, gmma, n);
	cC = 0.5*gsl_pow_2(gsl_sf_bessel_j0(qb/2));
	return L*L*(2.0/n*(Gmma-cC)+cB);
}

scalar sasfit_sq_freely_joined_chain_of_rods_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_sq_freely_joined_chain_of_rods_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

