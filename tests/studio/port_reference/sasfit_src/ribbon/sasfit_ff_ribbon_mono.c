/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables
#define A	param->p[0]
#define B	param->p[2]
#define C	param->p[4]
#define ETA	param->p[6]

// F(x) = 1F2(1/2; 3/2, 2; -x^2) = 2 int_0^1 (1-t) J0(2xt) dt = [int_0^{2x} J0(t) dt - J1(2x)]/x
//      = [J1(2x) + 2 sum_{k>=1} J_{2k+1}(2x)]/x      (Neumann series int_0^z J0 = 2 sum_k J_{2k+1}(z)),
// the in-plane orientation average of sinc^2(x cos(phi)). The odd-order Bessel sum is computed by Miller's
// downward recurrence normalised with J0 + 2 sum J_2k = 1 (exact to rounding for all x).
// 2026-11-18: replaces sasfit_pfq for qB/2 <= 3 and the asymptote 2/(qB) above (errors up to 2.3 %).
scalar ribbon_f1f2(scalar x)
{
    scalar z, jp, j, jm, norm, sodd, jone;
    int n, m;
    x = fabs(x);
    if (x < 1e-4) return 1.0-x*x/6.0+gsl_pow_4(x)/60.0;
    z = 2.0*x;
    n = (int)(z+10.0*cbrt(z)+30.0);
    if (n % 2) n++;
    jp = 0.0;
    j = 1e-280;
    norm = 0.0;
    sodd = 0.0;
    jone = 0.0;
    for (m=n; m>0; m--) {      // j = J_m (unnormalised) -> J_{m-1}
        jm = 2.0*m/z*j-jp;
        if (m % 2 == 0) {
            norm += 2.0*j;
        } else {
            sodd += j;
            if (m == 1) jone = j;
        }
        jp = j;
        j = jm;
        if (fabs(j) > 1e250) {
            jp *= 1e-250; j *= 1e-250; norm *= 1e-250; sodd *= 1e-250; jone *= 1e-250;
        }
    }
    norm += j;                 // + J0
    return (jone+2.0*(sodd-jone))/norm/x;
}

scalar sasfit_ff_ribbon_mono(scalar q, sasfit_param * param)
{
    scalar rod;
    sasfit_param subParam;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((A < 0.0), param, "A(%lg) < 0",A);
	SASFIT_CHECK_COND1((B < 0.0), param, "B(%lg) < 0",B);
	SASFIT_CHECK_COND1((C < 0.0), param, "C(%lg) < 0",C);

	sasfit_init_param(&subParam);
	subParam.p[0] = C;
	subParam.p[3] = fabs(ETA);
	rod = sasfit_ff_rod(q,&subParam);

	return rod*gsl_pow_2(A*B*gsl_sf_bessel_j0(q*A*0.5))*ribbon_f1f2(q*B*0.5);
}

scalar sasfit_ff_ribbon_mono_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	return 0.0;
}

scalar sasfit_ff_ribbon_mono_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	return A*B*C;
}
