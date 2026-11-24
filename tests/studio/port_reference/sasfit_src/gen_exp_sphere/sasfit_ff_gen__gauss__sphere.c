/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define ALPHA	param->p[1]
#define DETA	param->p[2]
#define BETA	(3+ALPHA)
// sum_n (a)_n (b)_n / n! z^(-n), stopped before the terms grow (asymptotic series)
static scalar ggs_asym(scalar a, scalar b, scalar z)
{
	scalar t = 1.0, s = 1.0, tn;
	int n;
	for (n = 0; n < 60; n++) {
		tn = t*(a+n)*(b+n)/((n+1)*z);
		if (fabs(tn) >= fabs(t) || tn == 0) break;
		t = tn;
		s += t;
		if (fabs(t) < 1e-17*fabs(s)) break;
	}
	return s;
}

scalar sasfit_ff_gen__gauss__sphere(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((ALPHA <= -3.0), param, "alpha(%lg) <= -3",ALPHA); // modify condition to your needs

    return gsl_pow_2(sasfit_ff_gen__gauss__sphere_f(q,param));
}

scalar sasfit_ff_gen__gauss__sphere_f(scalar q, sasfit_param * param)
{
    scalar V, F11, z;
	int status;
	gsl_sf_result gslres;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here

	V = 4*gsl_pow_3(RG)*M_PI;
//    if (q*RG<1e-6) return (V*DETA)*(1-gsl_pow_2(RG*q)/6.);
	if (q*RG==0) return DETA*V;
    gsl_set_error_handler_off();
	status = gsl_sf_hyperg_1F1_e(BETA/2.,1.5, -gsl_pow_2(q*RG)/(2*BETA),&gslres);
	if (status == 0 ) { //&& fabs(gslres.err)<fabs(1e-2*gslres.val)
		F11 = gslres.val;
	} else {
/*
	    F11 = M_PI/(16*gsl_pow_8(q))
                *( gsl_pow_2(M_PI)*gsl_pow_2((ALPHA-2)*(ALPHA-1)*ALPHA*(1+ALPHA)*BETA*BETA-4*ALPHA*(ALPHA+1)*BETA*gsl_pow_2(RG*q)+8*gsl_pow_4(RG*q))/gsl_pow_2(RG)
                           *exp(-gsl_pow_2(RG*q)/BETA-2*gsl_sf_lngamma(BETA/2.)+2*(ALPHA)*log(RG*q)-ALPHA*log(2*BETA))
                  + gsl_pow_2(sin(ALPHA*M_PI/2.))*q*q
                           *exp(2*gsl_sf_lngamma(1+ALPHA/2)-2*ALPHA*log(RG*q)+BETA*log(BETA)+(9+ALPHA)*log(2))
                  - 16*sqrt(2)*M_PI*q/RG*sin(ALPHA*M_PI)*pow(BETA,1.5)
                           *((ALPHA-2)*(ALPHA-1)*ALPHA*(1+ALPHA)*BETA*BETA-4*ALPHA*(ALPHA+1)*BETA*gsl_pow_2(RG*q)+8*gsl_pow_4(RG*q))
                           *exp(-0.5*gsl_pow_2(RG*q)/BETA-gsl_sf_lngamma(BETA/2.)+gsl_sf_lngamma(1+ALPHA/2.))
                );

        F11 = M_PI/gsl_pow_6(q)*( gsl_pow_2(M_PI)
                                    *gsl_pow_2(ALPHA*(1+ALPHA)*BETA-2*gsl_pow_2(RG*q))
                                    *exp(-gsl_pow_2(RG*q)/BETA-2*gsl_sf_lngamma(BETA/2.)+2*(1+ALPHA)*log(RG*q)-ALPHA*log(2*BETA))
                                 + (gsl_fcmp(lround(ALPHA),ALPHA,1e-10) && GSL_IS_EVEN(lround(ALPHA))?0:1)
                                    *gsl_pow_2(sin(ALPHA*M_PI/2.))
                                    *exp(2*gsl_sf_lngamma(1+ALPHA/2)-2*ALPHA*log(RG*q)+BETA*log(BETA)+(5+ALPHA)*log(2))
                                 + 4*sqrt(2)*M_PI*q*RG*sin(ALPHA*M_PI)*pow(BETA,1.5)
                                    *(ALPHA*(1+ALPHA)*BETA-2*gsl_pow_2(RG*q))
                                    *exp(-0.5*gsl_pow_2(RG*q)/BETA-gsl_sf_lngamma(BETA/2.)+gsl_sf_lngamma(1+ALPHA/2.))
                                );
*/
        // fallback (2026-11-23): asymptotic expansion of 1F1(a; 3/2; -z), a = b/2, z = (q Rg)^2/(2b),
        //   Gamma(3/2)/Gamma(3/2-a) z^(-a) sum_n (a)_n (a-1/2)_n/n! z^(-n)
        //   + cos(pi alpha/2) Gamma(3/2)/Gamma(a) e^(-z) z^(a-3/2) sum_n (3/2-a)_n (1-a)_n/n! (-z)^(-n)   (series stopped at the smallest term)
        // (was: sqrt of an uninitialised F11, then the square of the Gaussian term only)
        z = gsl_pow_2(q*RG)/(2*BETA);
        F11 = 0.5*sqrt(M_PI)*(gsl_sf_gammainv(1.5-BETA/2.)*exp(-0.5*BETA*log(z))*ggs_asym(BETA/2., BETA/2.-0.5, z)
                              + cos(M_PI*ALPHA/2.)*exp(-z+0.5*(BETA-3)*log(z)-gsl_sf_lngamma(BETA/2.))*ggs_asym(1.5-BETA/2., 1-BETA/2., -z));
	}
	return DETA*V* F11;
}

scalar sasfit_ff_gen__gauss__sphere_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 4*gsl_pow_3(RG)*M_PI;

}

