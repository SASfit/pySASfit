/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_peaks.h>
#include <math.h>
#include <stdlib.h>

#define A	param->p[0]
#define SIGMA_A	param->p[1]
#define B	param->p[2]
#define SIGMA_B	param->p[3]
#define C	param->p[4]
#define DUMMY	param->p[5]
#define ETA	param->p[6]

// 12-point Gauss-Legendre on [-1, 1] (positive half)
static const double ribbon_glx[6] = {0.1252334085114689, 0.3678314989981802, 0.5873179542866175,
                                     0.7699026741943047, 0.9041172563704749, 0.9815606342467192};
static const double ribbon_glw[6] = {0.2491470458134028, 0.2334925365383548, 0.2031674267230659,
                                     0.1600783285433462, 0.1069393259953184, 0.0471753363865118};

// K(w) = (2/pi)[ln((1+sqrt(1-w^2))/w) - sqrt(1-w^2)]:  F(x) = 2 int_0^1 K(w) cos(2xw) dw
static scalar ribbon_kern(scalar w)
{
    scalar s = sqrt(1.0-w*w);
    return M_2_PI*(log((1.0+s)/w)-s);
}

// <b^2 cos(q b w)> for a Schulz (gamma) distribution, shape k, scale th
static scalar ribbon_schulz_re(scalar k, scalar th, scalar q, scalar w)
{
    scalar u = q*th*w;
    return k*(k+1.0)*th*th*exp(-0.5*(k+2.0)*log1p(u*u))*cos((k+2.0)*atan(u));
}

// 12-point rule on [a, b] (type 0), or on [a, 1] with w = 1 - (1-a) tau^2, tau in [0, 1] (type 1)
static scalar ribbon_panel(scalar a, scalar b, int type, scalar k, scalar th, scalar q)
{
    scalar s = 0.0, c, h, t, w, jac;
    int i, sgn;
    c = 0.5*(a+b);
    h = 0.5*(b-a);
    if (type == 1) { c = 0.5; h = 0.5; }
    for (i=0; i<6; i++) {
        for (sgn=-1; sgn<=1; sgn+=2) {
            t = c+sgn*h*ribbon_glx[i];
            if (type == 1) {
                w = 1.0-(1.0-a)*t*t;
                jac = 2.0*(1.0-a)*t;
            } else {
                w = t;
                jac = 1.0;
            }
            s += ribbon_glw[i]*h*jac*ribbon_kern(w)*ribbon_schulz_re(k,th,q,w);
        }
    }
    return s;
}

// <b^2 F(qb/2)> for a Schulz distribution of b (mean bm, standard deviation sig):
//   = 2 int_0^1 K(w) Re[k(k+1) th^2 (1 - i q th w)^-(k+2)] dw,  k = bm^2/sig^2, th = sig^2/bm,
// which equals the former (bm^2+sig^2) 3F2(1/2, 1+k/2, 3/2+k/2; 3/2, 2; -(q sig^2/bm)^2). The integrand has at most
// (k+2)/4 oscillations whatever q. Panels: at most 1.5 rad of the phase (k+2) atan(q th w) each, geometric (factor 2)
// where the amplitude decays, 50 geometric subpanels towards w = 0 (log singularity of K), w = 1 - h tau^2 on the
// last panel. Same algorithm as pysasfit.studio.ports.ribbon (2026-11-18, replaces sasfit_pfq and the asymptote).
static int ribbon_cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static scalar ribbon_schulz_b2f(scalar q, scalar bm, scalar sig)
{
    scalar k, th, u, phase, w, w1, tot, *e;
    int n, j, ne, i, jj, nmax;
    k = gsl_pow_2(bm/sig);
    th = sig*sig/bm;
    u = q*th;
    if (q*bm*sqrt(1.0+2.0/k) < 1e-3) {
        return (bm*bm+sig*sig)*(1.0-q*q*(bm*bm+2*sig*sig)*(bm*bm+3*sig*sig)/(24*bm*bm));
    }
    // b^2 F(qb/2) = 2b/q + oscillating terms ~ b^(1/2) cos(qb + ...): their Schulz averages are damped by
    // (1+u^2)^(-(k+1/2)/2); below 1e-17 only the smooth part <2b/q> = 2 bm/q remains
    if (k > 1.0 && -0.5*(k-1.0)*log1p(u*u) < -39.0) return 2.0*bm/q;
    phase = (k+2.0)*atan(u);
    n = (int)(phase/1.5)+1;
    nmax = n+(int)(log(u+1.0)/M_LN2)+4;
    e = (scalar *)malloc(nmax*sizeof(scalar));
    if (e == NULL) return 2.0*bm/q;
    ne = 0;
    for (j=1; j<n; j++) e[ne++] = tan(j*phase/n/(k+2.0))/u;
    for (w=1.0/u; w<1.0 && ne<nmax; w*=2.0) e[ne++] = w;
    qsort(e, ne, sizeof(scalar), ribbon_cmp);
    jj = 0;                                      // keep 0 < e < 1, unique
    for (i=0; i<ne; i++) {
        if (e[i] <= 0.0 || e[i] >= 1.0) continue;
        if (jj > 0 && e[i] == e[jj-1]) continue;
        e[jj++] = e[i];
    }
    ne = jj;
    if (ne == 0) { e[0] = 0.5; ne = 1; }
    w1 = e[0];
    tot = 0.0;
    for (j=0; j<50; j++) tot += ribbon_panel(w1*pow(0.5,j+1), w1*pow(0.5,j), 0, k, th, q);
    for (i=0; i+1<ne; i++) tot += ribbon_panel(e[i], e[i+1], 0, k, th, q);
    tot += ribbon_panel(e[ne-1], 1.0, 1, k, th, q);
    free(e);
    return 2.0*tot;
}

scalar sasfit_peaks_ribbon_w_sd(scalar q, sasfit_param * param)
{
    scalar rod, ResA, ResB, ka, tha, ua, lg, ph;
    sasfit_param subParam;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((A <= 0.0), param, "A(%lg) <= 0",A);
	SASFIT_CHECK_COND1((SIGMA_A <= 0.0), param, "sigma_a(%lg) <= 0",SIGMA_A);
	SASFIT_CHECK_COND1((B < 0.0), param, "B(%lg) < 0",B);
	SASFIT_CHECK_COND1((SIGMA_B < 0.0), param, "sigma_B(%lg) < 0",SIGMA_B);
	SASFIT_CHECK_COND1((C < 0.0), param, "C(%lg) < 0",C);

	sasfit_init_param(&subParam);
	subParam.p[0] = C;
	subParam.p[3] = fabs(ETA);
	rod = sasfit_ff_rod(q,&subParam);

	// A: <a^2 j0^2(qa/2)> = 2 (1 - Re(1 - i q th_a)^-k_a)/q^2  (Schulz, mean A, standard deviation sigma_A)
	if (q*sqrt(A*A+SIGMA_A*SIGMA_A) < 1e-6) {
        ResA = A*A+SIGMA_A*SIGMA_A;
    } else {
        ka = gsl_pow_2(A/SIGMA_A);
        tha = SIGMA_A*SIGMA_A/A;
        ua = q*tha;
        lg = -0.5*ka*log1p(ua*ua);
        ph = ka*atan(ua);
        // 1 - e^lg cos(ph) = -expm1(lg) cos(ph) + 2 sin^2(ph/2)  (no cancellation for small q)
        ResA = 2.0*(-expm1(lg)*cos(ph)+2.0*gsl_pow_2(sin(0.5*ph)))/(q*q);
    }
	// B: <b^2 F(qb/2)>
	if (B == 0.0) {
        ResB = 0.0;
	} else if (SIGMA_B == 0.0) {
        ResB = B*B*ribbon_f1f2(q*B*0.5);
	} else {
        ResB = ribbon_schulz_b2f(q, B, SIGMA_B);
	}
	return rod*ResA*ResB;
}

scalar sasfit_peaks_ribbon_w_sd_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	return 0.0;
}

scalar sasfit_peaks_ribbon_w_sd_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	return 0.0;
}
