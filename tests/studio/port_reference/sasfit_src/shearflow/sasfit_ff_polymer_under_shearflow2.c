/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define RG	param->p[0]
#define FP	param->p[1]
#define ALPHA	param->p[2]
#define BETA	param->p[3]
#define GAMMA	param->p[4]
#define DELTA	param->p[5]
#define I0  param->p[6]
#define THETA0_DEG	param->p[7]
#define THETA0	(param->p[7]*M_PI/180)
#define PSI_DEG	param->p[8]
#define DELTA_THETA0DEG  param->p[9]
#define DELTA_THETA0    (param->p[9]*M_PI/180.)
#define Q   param->p[MAXPAR-1]
#define PSI	param->p[MAXPAR-2]
#define NN  param->p[MAXPAR-3]
#define MM  param->p[MAXPAR-4]

// Two-stage form factor P = 2 int_0^1 (1-x) exp(-g(x)) dx, g = b x (x < f), g = b f + c (x-f) (x > f), in closed form
// with E1(z) = (1-exp(-z))/z and E2(z) = int_0^1 t exp(-z t) dt (series for small z). The former expression
// 2(e^(-bf)/c + (1-e^(-bf))(1/b-1/b^2) + ...) lost digits for small q (rel. error ~1e-5 at q Rg = 0.005), 0/0 at q = 0.
static scalar two_stage_E1(scalar z) {
    scalar sum = 0.0, term = 1.0;
    int k;
    if (fabs(z) >= 0.1) return -expm1(-z)/z;
    for (k = 0; k < 14; k++) {  // sum_k (-z)^k/(k+1)!
        if (k > 0) term *= -z/(k+1);
        sum += term;
    }
    return sum;
}
static scalar two_stage_E2(scalar z) {
    scalar sum = 0.0, term = 1.0;
    int k;
    if (fabs(z) >= 0.1) return (-expm1(-z) - z*exp(-z))/(z*z);
    for (k = 0; k < 14; k++) {  // sum_k (-z)^k/(k! (k+2))
        if (k > 0) term *= -z/k;
        sum += term/(k+2);
    }
    return sum;
}
static scalar two_stage(scalar b, scalar c, scalar f) {
    scalar L = 1.0-f;
    return 2.0*f*(two_stage_E1(b*f) - f*two_stage_E2(b*f))
         + 2.0*exp(-b*f)*L*L*(two_stage_E1(c*L) - two_stage_E2(c*L));
}

scalar sasfit_ff_polymer_under_shearflow2_core(scalar theta, sasfit_param * param) {
    scalar b,c;
    b= gsl_pow_2(RG*ALPHA*Q*cos(theta+THETA0-PSI))+gsl_pow_2(RG*BETA *Q*sin(theta+THETA0-PSI));
    c= gsl_pow_2(RG*GAMMA*Q*cos(theta+THETA0-PSI))+gsl_pow_2(RG*DELTA*Q*sin(theta+THETA0-PSI));


    return two_stage(b,c,FP);
	return 2*(exp(-c - b*FP)*(gsl_pow_2(b)*exp(c*FP) +
       exp(c)*((-1 + b)*gsl_pow_2(c)*exp(b*FP) +
          (-b + c)*(b + c + b*c*(-1 + FP))))) / (gsl_pow_2(b*c));

	return 2/(b*b)*(b-1+exp(-FP*b)
	                   *(1+(gsl_pow_2(b/c)-1)*exp((FP-1)*c)+(1-FP)*b/c*(b-c)));
}

scalar sasfit_ff_polymer_under_shearflow2(scalar q, sasfit_param * param)
{

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0) , param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG < 0.0), param, "Rg(%lg) < 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((FP < 0.0), param, "fp(%lg) < 0",FP); // modify condition to your
	SASFIT_CHECK_COND1((FP > 1.0), param, "fp(%lg) > 1",FP); // modify condition to your needs

	// insert your code here

	Q = q;
	PSI = sasfit_param_override_get_psi(PSI_DEG * M_PI/180.);

	if (DELTA_THETA0 == 0) return I0*sasfit_ff_polymer_under_shearflow2_core(0,param);
	return I0*sasfit_integrate(-DELTA_THETA0/2.,DELTA_THETA0/2.,&sasfit_ff_polymer_under_shearflow2_core,param)/DELTA_THETA0;
}

scalar sasfit_ff_polymer_under_shearflow2_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_polymer_under_shearflow2_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

