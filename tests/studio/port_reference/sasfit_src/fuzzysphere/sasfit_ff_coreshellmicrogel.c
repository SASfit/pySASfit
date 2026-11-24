/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define W_CORE		param->p[0]
#define SIGMA_CORE	param->p[1]
#define W_SH		param->p[2]
#define SIGMA_SHIN	param->p[3]
#define D			param->p[4]
#define SIGMA_OUT	param->p[5]
#define ETA_CORE	param->p[6]
#define ETA_SHELL	param->p[7]
#define ETA_SOL		param->p[8]


#define R_CORE		W_CORE+SIGMA_CORE
#define R_OUT		R_CORE+D+SIGMA_SHIN+W_SH+SIGMA_OUT
#define R_SHIN		R_CORE+D

typedef struct {
	scalar rad, sigma, contrast;
} microgel_prof;

// parabolic transition: 1 inside rad-sigma, 0 outside rad+sigma (as radial_profile_of_coreshellmicrogel)
static double microgel_rho(double r, void *pp)
{
	microgel_prof *p = (microgel_prof *) pp;
	double rad = p->rad, s = p->sigma;
	r = fabs(r);
	if (r <= rad-s) return p->contrast;
	if (r <= rad) return p->contrast*(1.0-0.5*gsl_pow_2((r-rad+s)/s));
	if (r <= rad+s) return p->contrast*0.5*gsl_pow_2((rad-r+s)/s);
	return 0.0;
}

scalar ampl(scalar q, scalar rad, scalar sigma, scalar contrast)
{
	scalar u,v,w,t,s,Vol;

	u = q*fabs(rad);
	v=q*(fabs(rad)+fabs(sigma));
	w=q*(fabs(rad)-fabs(sigma));
	t=gsl_pow_5(q)*gsl_pow_2(sigma);
	s=gsl_pow_4(q)*gsl_pow_2(sigma);

	Vol=4.*M_PI*(gsl_pow_3(rad)/3.+rad*gsl_pow_2(sigma)/6.0);

	// insert your code here
	
	if (Vol==0) return 0;

	// small q sigma (2026-11-23): the closed forms divide by q^3 ... q^5 (and the series in sigma stopped at
	// sigma^4); for q sigma < 2 the inner sphere (stable phi) plus the two transition zones by Gauss-Legendre
	if (fabs(sigma)*q < 2.0) {
		microgel_prof pr;
		pr.rad = fabs(rad); pr.sigma = fabs(sigma); pr.contrast = contrast;
		if (pr.sigma == 0) return fz_sphere(q, pr.rad, contrast);
		return (pr.rad > pr.sigma ? fz_sphere(q, pr.rad-pr.sigma, contrast) : 0.0)
			+ fz_zone(q, pr.rad-pr.sigma, pr.rad, microgel_rho, &pr)
			+ fz_zone(q, pr.rad, pr.rad+pr.sigma, microgel_rho, &pr);
	}

	if (sigma*q<1e-3) {
		if (q==0) {
			return Vol * contrast;
		} else {
			return Vol
				* contrast / (gsl_pow_3(rad)/3.+rad*gsl_pow_2(sigma)/6.0) *
				(					 (sin(u)-u*cos(u))/gsl_pow_3(q)
				+ gsl_pow_2(sigma) * (sin(u)+u*cos(u))/(12.*q)
				- gsl_pow_4(sigma) * ((1./360.)*q*u*cos(u)+1./120.*q*sin(u))
				);
		}
	} else {
		if (q==0) {
			return Vol * contrast;
		} else {
			return Vol
				* contrast / (gsl_pow_3(rad)/3.+rad*gsl_pow_2(sigma)/6.0)
				* ( (rad/gsl_pow_2(sigma)+1/sigma) * cos(v)/gsl_pow_4(q)
				   +(rad/gsl_pow_2(sigma)-1/sigma) * cos(w)/gsl_pow_4(q)
				   -3*sin(v)/t
				   -3*sin(w)/t
				   -2*rad*cos(u)/s
				   +6*sin(u)/t
				  );
		}
	}
}

scalar sasfit_ff_coreshellmicrogel(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);

	// insert your code here
	return gsl_pow_2(sasfit_ff_coreshellmicrogel_f(q,param));
}

scalar sasfit_ff_coreshellmicrogel_f(scalar q, sasfit_param * param)
{
	scalar u,v,w,t,s;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	 return		ampl(q,R_OUT,SIGMA_OUT,(ETA_SHELL-ETA_SOL))
			-	ampl(q,R_SHIN,SIGMA_SHIN,(ETA_SHELL-ETA_SOL))
			+	ampl(q,R_CORE,SIGMA_CORE,(ETA_CORE -ETA_SOL));
}

scalar sasfit_ff_coreshellmicrogel_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 4.*M_PI*gsl_pow_3(R_OUT+SIGMA_OUT)/3.;
}

