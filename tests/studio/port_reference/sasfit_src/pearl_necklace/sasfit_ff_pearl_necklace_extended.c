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
#define G param->p[7]

// N pearls; between neighbouring pearls g freely jointed rods of length l. The first rod starts on the surface of
// pearl k with its axis through the centre of pearl k, the last rod ends on the surface of pearl k+1 with its axis
// through that centre, i.e. the bonds have the lengths R+l, l, ..., l, l+R (one bond l+2R for g = 1, which is the
// original pearl necklace). Freely jointed bonds: every pair term is a product of sin(qL)/(qL) over the bonds in
// between; the double sum is done by a backward recursion (O(N g)).
// Rewritten 2026-11-18: the former formula used rods of length A = l+2R with Lambda(qA) in the rod-rod terms and did
// not reduce to the original pearl necklace for g = 1.
static scalar PN(scalar q, scalar F, scalar P, int g, int n, sasfit_param *param) {
    scalar qR, ql, lam, be, rod, S, acc, bL, bin, bout;
    int k, j;

	qR = q*R;
	ql = q*L;
	lam = Lambda(ql,param);
    be = LambdaExt(qR,ql,param);
    rod = 2*lam-gsl_pow_2(gsl_sf_bessel_j0(ql/2));
    S = n*gsl_pow_2(BETA_S)*P + g*(n-1)*gsl_pow_2(BETA_R)*rod;
    acc = 0.0;
    for (k=n-1;k>=0;k--) {
        S += 2*BETA_S*F*acc;
        acc += BETA_S*F;
        if (k==0) break;
        for (j=g-1;j>=0;j--) {
            if (g==1) {
                bL = L+2*R; bin = be; bout = be;
            } else if (j==0) {
                bL = R+L; bin = be; bout = lam;
            } else if (j==g-1) {
                bL = L+R; bin = lam; bout = be;
            } else {
                bL = L; bin = lam; bout = lam;
            }
            S += 2*BETA_R*bout*acc;
            acc = BETA_R*bin + gsl_sf_bessel_j0(q*bL)*acc;
        }
    }
    // SASfit convention I(0) = (m*beta_r + n*beta_s)^2 (beta = excess scattering lengths); was normalised to P(0) = 1
    // until 2026-11-18
    return S;
}

scalar sasfit_ff_pearl_necklace_ext(scalar q, sasfit_param * param)
{
    scalar  res, w, wg, F, P;
    int n,g;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((R < 0.0), param, "R(%lg) < 0",R); // modify condition to your needs
	SASFIT_CHECK_COND1((L < 0.0), param, "L(%lg) < 0",L); // modify condition to your needs
	SASFIT_CHECK_COND1((BETA_R < 0.0), param, "beta_r(%lg) < 0",BETA_R); // modify condition to your needs
	SASFIT_CHECK_COND1((BETA_S < 0.0), param, "beta_s(%lg) < 0",BETA_S); // modify condition to your needs
	SASFIT_CHECK_COND1((NP < 1.0), param, "Np(%lg) < 1",NP); // modify condition to your needs
    SASFIT_CHECK_COND1((G < 1.0), param, "g(%lg) < 1",G); // modify condition to your needs

	// insert your code here
    n = lround(floor(NP));
	w = NP-n;
	g = lround(floor(G));
	wg = G-g;

	F = Psi(q*R);
	P = F*F;

    res = (1-wg)*(1-w)*PN(q,F,P,g,n,param);
	if (wg!=0) {
        res = res+wg*(1-w)*PN(q,F,P,g+1,n,param);
 	}
	if (w!=0) {
        res=res+(1-wg)*w*PN(q,F,P,g,n+1,param);
    	if (wg!=0) {
            res=res+wg*w*PN(q,F,P,g+1,n+1,param);
        }
	}
	return res;
}

scalar sasfit_ff_pearl_necklace_ext_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_pearl_necklace_ext_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

