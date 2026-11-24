/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

 // doi: 10.1021/ma049950l
 // Burchard, Walther
 // Angular Dependence of Scattered Light from Hyperbranched Structures in a Good Solvent. A Fractal Approach
 // 2004-04
 // Macromolecules , Vol. 37, No. 10
 // American Chemical Society (ACS)
 // p. 3841-3849

#include "include/private.h"
#include <sasfit_error_ff.h>
#include <math.h>


// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define F	param->p[1]
#define K   param->p[2]
#define I0	param->p[3]


scalar sasfit_ff_comb_hetero(scalar q, sasfit_param * param)
{
    scalar X,X2,X4,Comb,a,b;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG <= 0.0), param, "Rg(%lg) <= 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((F <= 1.0), param, "f(%lg) <= 1",F); // modify condition to your needs
	SASFIT_CHECK_COND1((K < 0.0), param, "k(%lg) < 0",K); // modify condition to your needs
	SASFIT_CHECK_COND1((K > 1.0), param, "k(%lg) > 1",K); // modify condition to your needs

	// insert your code here

    X= q*RG;
    X2=X*X;
    X4=X2*X2;
    if (X2==0) return I0;
    // 2026-11-24: same closed form with positive terms (the direct form cancelled O(X^2) terms and returned
    // 2 I0/X^4/(1+(1-K)^2/F) for its NaN, e.g. at K = 0)
    a = X2*(1-K)/F;
    b = X2*K;
    Comb = F*a*a*hbp_g(a) + hbp_g(b)*gsl_pow_2(b + F*a*hbp_h(a));
    return 2*I0*Comb/X4/(1+gsl_pow_2(1-K)/F);
}

scalar sasfit_ff_comb_hetero_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_comb_hetero_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

