/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_sq.h>

// define shortcuts for local parameters/variables


scalar Lambda(scalar qb, sasfit_param *param) {
    scalar Lbda;
	if (fabs(qb)<1e-6) {
		Lbda = 1 - gsl_pow_2(qb)/18. + gsl_pow_4(qb)/600. - gsl_pow_6(qb)/35280. + gsl_pow_8(qb)/3265920.;
	} else {
		Lbda = gsl_sf_Si(qb)/qb;
	}
	return Lbda;
}

scalar LambdaExt(scalar qR, scalar qL, sasfit_param *param) {
    // (Si(qR+qL)-Si(qR))/qL = mean of sin(t)/t over [qR, qR+qL]
    // qL < 2: 10-point Gauss-Legendre (the difference of Si values loses digits for small qL)
    static const double x[5] = {0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
                                0.8650633666889845, 0.9739065285171717};
    static const double w[5] = {0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
                                0.1494513491505806, 0.0666713443086881};
    scalar c, h, s;
    int i;
    if (qL == 0.0) return gsl_sf_bessel_j0(qR);
    if (qL >= 2.0) return (gsl_sf_Si(qL+qR)-gsl_sf_Si(qR))/qL;
    c = qR+qL/2.;
    h = qL/2.;
    s = 0.0;
    for (i=0;i<5;i++) {
        s += w[i]*(gsl_sf_bessel_j0(c-h*x[i])+gsl_sf_bessel_j0(c+h*x[i]));
    }
    return s/2.;
}

scalar Psi(scalar x) {
    // series below 0.05 (cancellation in sin(x)-x*cos(x)); the x^4 coefficient was 1/240 instead of 1/280
    scalar x2;
    if (fabs(x)<0.05) {
        x2 = x*x;
        return 1.-x2/10.*(1.-x2/28.*(1.-x2/54.*(1.-x2/88.)));
	} else {
        return 3.*(sin(x)-x*cos(x))/gsl_pow_3(x);
	}
}

scalar pn_sum_u(int m, scalar g) {
    // sum_{k=0}^{m-1} (m-k) g^k (0 for m <= 0), finite form of the geometric closed forms, stable for g -> 1
    scalar s=0.0, gk=1.0;
    int k;
    for (k=0;k<m;k++) {
        s += (m-k)*gk;
        gk *= g;
    }
    return s;
}

scalar pn_sum_geo(int m, scalar g) {
    // sum_{k=0}^{m-1} g^k = (1-g^m)/(1-g)
    scalar s=0.0, gk=1.0;
    int k;
    for (k=0;k<m;k++) {
        s += gk;
        gk *= g;
    }
    return s;
}

scalar pn_geo_real(scalar x, int j, scalar s) {
    // j = 1: x/(1-s) - s (1-|s|^x)/(1-s)^2,  j = 0: x/(1-s) - (1-|s|^x)/(1-s)^2  (real x);
    // binomial series sum_k (-1)^k C(x+j,k+2) (1-s)^k for small 1-s (the closed form cancels there)
    scalar e, y, c, ek, res;
    int k;
    e = 1.0-s;
    if (fabs(e) < 0.25 && fabs(x*e) < 0.5) {
        y = x+j;
        c = y*(y-1.)/2.;
        ek = 1.0;
        res = 0.0;
        for (k=0;k<40;k++) {
            res += c*ek;
            c *= -(y-k-2.)/(k+3.);
            ek *= e;
        }
        return res;
    }
    return x/e - (j==1 ? s : 1.0)*(1.0-pow(fabs(s),x))/(e*e);
}
