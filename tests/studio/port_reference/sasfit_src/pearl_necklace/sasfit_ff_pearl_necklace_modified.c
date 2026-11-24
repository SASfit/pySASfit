/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 *   8.12.2014
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define R	param->p[0]
#define DUMMY1 param->p[1] // nu
#define DUMMY2 param->p[2] // d
#define L	param->p[3]
#define BETA_R	param->p[4]
#define BETA_S	param->p[5]
#define NP	param->p[6]
#define DUMMY3 param->p[7]

scalar sasfit_ff_pearl_necklace_mod(scalar q, sasfit_param * param)
{
    scalar  qR,ql,qA,Fpsi,Ppsi, LA,Mr,mr,ms,beta,Sr,Sp,Sm,sinqA_qA,sinql2_ql2,A;
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((R < 0.0), param, "R(%lg) < 0",R); // modify condition to your needs
	SASFIT_CHECK_COND1((L < 0.0), param, "L(%lg) < 0",L); // modify condition to your needs
	// SASFIT_CHECK_COND1((BETA_R < 0.0), param, "beta_r(%lg) < 0",BETA_R); // modify condition to your needs   // removed 2026-11-24: scattering length densities / contrasts may be negative
	// SASFIT_CHECK_COND1((BETA_S < 0.0), param, "beta_s(%lg) < 0",BETA_S); // modify condition to your needs   // removed 2026-11-24: scattering length densities / contrasts may be negative
	SASFIT_CHECK_COND1((NP < 0.0), param, "Np(%lg) < 0",NP); // modify condition to your needs

	// insert your code here

    A=L+2.*R;
	qR = q*R;
	ql = q*L;
	qA = q*A;
	Mr=NP-1.0;
	Fpsi = Psi(qR);
	Ppsi=gsl_pow_2(Fpsi);
	if (ql == 0.0) {
        LA=1.0;
        sinql2_ql2= 1.0;
	} else {
	    LA=gsl_sf_Si(ql)/(ql);
	    sinql2_ql2 = sin(ql/2.0)*2/ql;
	}
	if (qA == 0.0) {
        sinqA_qA=1.;
	} else {
	    sinqA_qA=sin(qA)/(qA);
	}


    beta = LambdaExt(qR,ql,param);
	// pn_geo_real: the closed forms N/(1-s)-s(1-|s|^N)/(1-s)^2 etc. as series for s -> 1 (0/0 at q = 0 before)
    Sp=2*pow(BETA_S,2)*Ppsi*(pn_geo_real(NP,1,sinqA_qA)-NP/2);
    Sr=pow(BETA_R,2)*(Mr*(2*LA-pow(sinql2_ql2,2))+2*pow(beta,2)*pn_geo_real(Mr,0,sinqA_qA));
    Sm=BETA_R*beta*BETA_S*Fpsi*4*pn_geo_real(NP-1,1,sinqA_qA);
	// SASfit convention I(0) = (m*beta_r + n*beta_s)^2 (beta = excess scattering lengths); was normalised to P(0) = 1
	// until 2026-11-18
	return Sp+Sr+Sm;
}

scalar sasfit_ff_pearl_necklace_mod_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_pearl_necklace_mod_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

