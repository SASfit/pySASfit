/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define R	param->p[0]

#define L	param->p[2]
#define ETA	param->p[3]


static void porod_long_coeffs(scalar a, scalar b, scalar *c)
{   /* small-q series of the long-cylinder formula, a = L, b = 2R (batch 7, 2026-11-24) */
    c[0] = 1;
    c[1] = -1.0/2304.0*(64*pow(a, 4) + 96*pow(a, 2)*pow(b, 2) + 5*pow(b, 4))/pow(a, 2);
    c[2] = (1.0/1843200.0)*(1024*pow(a, 6) + 1280*pow(a, 4)*pow(b, 2) + 2000*pow(a, 2)*pow(b, 4) + 105*pow(b, 6))/pow(a, 2);
    c[3] = -1.0/17340825600.0*(122880*pow(a, 8) + 172032*pow(a, 6)*pow(b, 2) + 313600*pow(a, 4)*pow(b, 4) + 274400*pow(a, 2)*pow(b, 6) + 12789*pow(b, 8))/pow(a, 2);
    c[4] = (1.0/7491236659200.0)*(458752*pow(a, 10) + 737280*pow(a, 8)*pow(b, 2) + 1935360*pow(a, 6)*pow(b, 4) + 1975680*pow(a, 4)*pow(b, 6) + 1111320*pow(a, 2)*pow(b, 8) + 45045*pow(b, 10))/pow(a, 2);
    c[5] = -1.0/116024273377689600.0*(44040192*pow(a, 12) + 80740352*pow(a, 10)*pow(b, 2) + 297369600*pow(a, 8)*pow(b, 4) + 437133312*pow(a, 6)*pow(b, 6) + 286868736*pow(a, 4)*pow(b, 8) + 112698432*pow(a, 2)*pow(b, 10) + 3978117*pow(b, 12))/pow(a, 2);
    c[6] = (1.0/18823778112796360704000.0)*(33218887680*pow(a, 14) + 68702699520*pow(a, 12)*pow(b, 2) + 341127987200*pow(a, 10)*pow(b, 4) + 703576473600*pow(a, 8)*pow(b, 6) + 664879767552*pow(a, 6)*pow(b, 8) + 304736560128*pow(a, 4)*pow(b, 10) + 88428019680*pow(a, 2)*pow(b, 12) + 2737609875*pow(b, 14))/pow(a, 2);
}

static scalar porod_omega(scalar sig)
{   /* 8/sig^2 (3 J2 + J0 - 1): series for sig < 1, the closed form cancels for small sig */
    scalar h2, pw, ser, fk, fkm1, fkp1, ck;
    int k;
    if (sig >= 1.0) return 8.0/gsl_pow_2(sig)*(3.0*gsl_sf_bessel_Jn(2,sig)+gsl_sf_bessel_J0(sig)-1.0);
    h2 = 0.25*sig*sig; pw = 1.0; ser = 0.0;
    for (k = 1; k < 16; k++) {
        fk = gsl_sf_fact(k); fkm1 = gsl_sf_fact(k-1); fkp1 = gsl_sf_fact(k+1);
        ck = ((k % 2) ? -1.0 : 1.0)*(1.0/(fk*fk) - 3.0/(fkm1*fkp1));
        ser += 2.0*ck*pw;
        pw *= h2;
    }
    return ser;
}


static scalar porod_poly_l(const scalar *c, scalar q)
{
    scalar u = q*q, s = 0.0;
    int k;
    for (k = 6; k >= 0; k--) s = s*u + c[k];
    return s;
}

scalar sasfit_ff_longcylinder(scalar q, sasfit_param * param)
{
    scalar mu, sigma, pi_mu, G1, G2, I_sp, Sum, V, omega;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((R < 0.0), param, "R(%lg) < 0",R); // modify condition to your needs
	SASFIT_CHECK_COND1((L < 0.0), param, "L(%lg) < 0",L); // modify condition to your needs

	// insert your code here
    if (R == 0.0) return 0.0;
	if (L == 0.0) return 0.0;

	mu = L*q;
	sigma = 2.0*R*q;
	V = M_PI*R*R*L;

	if (R==0 || L==0) return 0;
	if (q*GSL_MAX(L,2.0*R) < 0.3) {   // 2026-11-24: series (the closed form lost up to 1e-4 for small q)
		scalar c[7];
		porod_long_coeffs(L, 2.0*R, c);
		return ETA*ETA*V*V*porod_poly_l(c, q);
	}

	pi_mu = gsl_sf_Si(mu)+cos(mu)/mu+sin(mu)/mu/mu;
	G1 = 2.0/(0.5*sigma) *  gsl_sf_bessel_J1(0.5*sigma);
	G2 = 8.0/sigma/sigma * gsl_sf_bessel_Jn(2,sigma);
//	I_sp = 3.0 * (sin(sigma*0.5)-0.5*sigma*cos(0.5*sigma)) / pow(sigma/2.0,3);
//	I_sp = I_sp*I_sp;
	omega = porod_omega(sigma);   // 2026-11-24: series for small sigma

//	Sum = 2.0/mu * (pi_mu*G1*G1 - 1.0/mu*(2.0*G2-I_sp) - sin(mu)/mu/mu);
	Sum = 2.0/mu * (pi_mu*G1*G1 - omega/mu - sin(mu)/mu/mu);

	return ETA*ETA *V*V* Sum;
}

scalar sasfit_ff_longcylinder_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_longcylinder_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return M_PI*gsl_pow_2(R)*L;
}

