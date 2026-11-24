/* IDebye, S_SB, S_EXV, fcorr, IRod of thin_objects/sasfit_ao_utils.c (unchanged; fRg2 and falpha2 come from
   spherical_ellipsoidal_micelles, identical code). Reference copy for the Python ports. */
#include "include/private.h"

scalar IDebye(scalar q, scalar Rg)
// Scattering function (Debye function) for Gaussian Coils with radius of gyration Rg
{
	scalar u;
	u = gsl_pow_2(q*Rg);
	if (u < 0.05) { // series (1 - u/3 below 0.01 was 8e-6 off; 2026-11-23)
		return 1.0-u/3.0*(1.0-u/4.0*(1.0-u/5.0*(1.0-u/6.0*(1.0-u/7.0*(1.0-u/8.0)))));
	} else {
		return 2.0*(gsl_expm1(-u)+u)/gsl_pow_2(u);
	}
}

scalar SB_term(scalar u)
// 4/15 + 7/(15u) - (11/15 + 7/(15u)) exp(-u) without the cancellation of 7/(15u)(1 - exp(-u)) for small u
{
	if (u == 0.0) return 0.0;
	return 4./15. - 11./15.*exp(-u) - 7./15.*gsl_expm1(-u)/u;
}

scalar S_SB(scalar q, scalar Rg, sasfit_param *param)
// The scattering function calculated for the
// Daniels approximation by Sharp and Bloomfield
{
	scalar u, nb;
	nb = L/B;
	u = gsl_pow_2(q*Rg);
	return IDebye(q,Rg)+SB_term(u)/nb;
}

scalar S_EXV(scalar q, scalar Rg, sasfit_param *param)
/*
The expression given for the scattering function for worm like chains with excluded volume effects
by Utiyama et al.21 consists of a low-q and a large-q expansion. In order to get good agreement at
intermediate q, a large number of terms is included in the high-q expansion and this makes the
numerical calculations rather slow. An approximate expression has therefore been constructed using
this approximation:
*/
{
	scalar SDebye, C1,C2,C3,w,x,nu,epsilon;
    SDebye = IDebye(q,Rg);
	C1 = 1.2220;
	C2 = 0.4288;
	C3 = -1.651;
	epsilon = 0.170;
	nu = (1.0+epsilon)/2.0;
	x = Rg*q;
	w = (1.+tanh((x-1.523)/0.1477))/2.0;
	return  (1-w)* SDebye
		  + (  w)* (  C1*pow(x,-1.0/nu)
			         +C2*pow(x,-2.0/nu)
			         +C3*pow(x,-3.0/nu))*fcorr(q,Rg,param);
}


scalar fcorr(scalar q, scalar RG, sasfit_param *param)
{
/*
	Correction function as described in the paper
	Wei-Ren Chen, Paul D. Butler, and Linda J. Magid
	Incorporating Intermicellar Interactions in the Fitting of SANS Data
	from Cationic Wormlike Micelles
	Langmuir 2006, 22, 6539-6548
	DOI: 10.1021/la0530440
	f_corr(Q) = 0 for dS(q)/dq > 0
	f_corr(Q) = 1 for dS(q)/dq <= 0
*/
	scalar dS;
	scalar C1, C2, C3, C4, C5, epsilon, nu;

	C1 = 1.2220;
	C2 = 0.4288;
	C3 = -1.651;
	epsilon = 0.170;
	nu = (1.0+epsilon)/2.0;
	C4 = 1.523;
	C5 = 0.1477;

	dS = -(1.0-pow(tanh((q*RG-C4)/C5),2.0))/(RG*RG*RG)/C5*(2.0*exp(-q*q*RG*RG
)+2.0*q*q*RG*RG-2.0)/(q*q*q*q)/2.0+(1.0/2.0-tanh((q*RG-C4)/C5)/2.0)*(-4.0*q*RG*
RG*exp(-q*q*RG*RG)+4.0*q*RG*RG)/(q*q*q*q)/(RG*RG*RG*RG)-4.0*(1.0/2.0-tanh((q*RG
-C4)/C5)/2.0)*(2.0*exp(-q*q*RG*RG)+2.0*q*q*RG*RG-2.0)/(q*q*q*q*q)/(RG*RG*RG*RG)
+(1.0-pow(tanh((q*RG-C4)/C5),2.0))*RG/C5*(C1*pow(q*RG,-1.0/nu)+C2*pow(q*RG,-2.0
/nu)+C3*pow(q*RG,-3.0/nu))/2.0+(1.0/2.0+tanh((q*RG-C4)/C5)/2.0)*(-C1*pow(q*RG,
-1.0/nu)/nu/q-2.0*C2*pow(q*RG,-2.0/nu)/nu/q-3.0*C3*pow(q*RG,-3.0/nu)/nu/q)+CLB*
(-14.0/15.0/(q*q*q)/(RG*RG)+14.0/15.0/(q*q*q)/(RG*RG)*exp(-q*q*RG*RG)+2.0*(11.0
/15.0+7.0/15.0/(q*q)/(RG*RG))*q*RG*RG*exp(-q*q*RG*RG))*B/L;

	if (dS<=0) {
		return 1;
	} else {
		return 0;
	}
}

scalar IRod(scalar q, scalar H)
//  ROD SCATTERING FUNCTION
{
	return 2.0*gsl_sf_Si(q*H)/(q*H)-gsl_pow_2(sin(q*H/2.)/(q*H/2.));
}




