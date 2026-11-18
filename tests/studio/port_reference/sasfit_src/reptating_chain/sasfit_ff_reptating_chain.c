/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define I0	param->p[0]
#define RG	param->p[1]
#define LAMBDA	param->p[2]
#define T_TAU	param->p[3]
#define THETA_0	param->p[4]*M_PI/180.
#define PSIDEG	param->p[5]
#define PSI	param->p[MAXPAR-1]
#define ALPHA	param->p[MAXPAR-2]
#define BETA	param->p[MAXPAR-3]

#define NMAX 100

scalar H(scalar x, scalar y, sasfit_param * param) {
	int n;
	scalar Hsum,Hsum_old;
	Hsum = 0.0;
	Hsum_old = 1.0;
	
	if (y == 0.0) return 0.0;
	for (n=1;n<NMAX;n=n+2) {
		Hsum_old= Hsum;
		if (T_TAU == 0.0)	{
			Hsum = Hsum+96./gsl_pow_2(M_PI*n)/(gsl_pow_2(n*M_PI)+gsl_pow_2(x*y));
		} else {
			Hsum = Hsum+96./gsl_pow_2(M_PI*n)/(gsl_pow_2(n*M_PI)+gsl_pow_2(x*y))*exp(-gsl_pow_2(n/y)*T_TAU);
		}
		if (n>3 && fabs(Hsum-Hsum_old) < 1e-10*fabs(Hsum)) break;
	}
	return Hsum;
}

scalar HKernel(scalar y, sasfit_param * param) {
	return y*y*y*(exp(BETA*(y-1))+exp(BETA*(y-1)-ALPHA*y))*H(ALPHA,y,param);
}
scalar D(scalar x) {
	// Debye function 2(x-1+exp(-x))/x^2; for small x the series sum_k 2(-x)^k/(k+2)! (the closed form loses
	// all digits by cancellation, relative error ~1e-16/x^2)
	scalar sum, fact, term;
	int k;
	if (fabs(x) < 0.2) {
		sum = 0.0;
		fact = 2.0;
		term = 1.0;
		for (k = 0; k < 14; k++) {
			if (k > 0) {
				fact *= k+2;
				term *= -x;
			}
			sum += 2.0*term/fact;
		}
		return sum;
	}
	return 2*(x+expm1(-x))/(x*x);
}
scalar sasfit_ff_reptating_chain(scalar q, sasfit_param * param)
{
	scalar Dpsi, q_ll, q_perp, E, L3;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((I0 < 0.0), param, "I0(%lg) < 0",I0); // modify condition to your needs
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((LAMBDA < 1.0), param, "lambda(%lg) < 1",LAMBDA); // modify condition to your needs
	SASFIT_CHECK_COND1((T_TAU < 0.0), param, "t/tau(%lg) < 0",T_TAU); // modify condition to your needs
	// theta_0 and psi may have any value (only psi - theta_0 matters); the former check of psi < 0 read the
	// scratch value PSI of the previous call before it was set below

	// insert your code here
	
    PSI = sasfit_param_override_get_psi(PSIDEG*M_PI/180.);
	Dpsi = PSI-THETA_0;
	q_ll = q*cos(Dpsi);
	q_perp = q*sin(Dpsi);
	ALPHA = gsl_pow_2(q*RG);
	if (LAMBDA == 1) return I0*D(ALPHA);  // before L3 = 0 is divided by; I0 was not applied before
	L3 = sqrt(gsl_pow_3(LAMBDA)-1);
	E= 1./2.*(LAMBDA+gsl_asinh(L3)/sqrt(LAMBDA)/L3); 
	BETA = (gsl_pow_2(LAMBDA*q_ll)+ gsl_pow_2(q_perp)/LAMBDA)*RG*RG/E;
	
	return I0*(D(ALPHA)+1./6.*(ALPHA-BETA)*(1.+exp(-ALPHA))*H(ALPHA,1,param)
				   +1./6.*gsl_pow_2(ALPHA-BETA)*sasfit_integrate(0.0,1.0,&HKernel,param));

	return D(BETA)-4/(M_PI*M_PI)*T_TAU*(ALPHA/BETA-1)*(1-exp(-BETA));
	return D(ALPHA)+16/(M_PI*M_PI) * (ALPHA-BETA)/(ALPHA*ALPHA+M_PI*M_PI)*(1+exp(-ALPHA))*exp(-T_TAU);
}

scalar sasfit_ff_reptating_chain_f(scalar q, sasfit_param * param)
{
	
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid
	// insert your code here
	return 0.0;
}

scalar sasfit_ff_reptating_chain_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

