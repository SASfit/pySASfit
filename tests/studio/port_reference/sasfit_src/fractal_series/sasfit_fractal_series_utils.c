/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>
#include <gsl/gsl_roots.h>

// define shortcuts for local parameters/variables

scalar fractalSD_single(scalar x, scalar a,scalar fd, scalar xmin, scalar xmax,sasfit_param * param) {
    return a*pow(x,-(1+fd))*fd/(pow(xmin,-fd)-pow(xmax,-fd));
}

scalar sc_ExpIntegralE(double n, double x) {
    if (x >=0) {
        return pow(x,n-1)*gsl_sf_gamma_inc(1-n,x);
    } else {
        return 0.0;
    }
}

static scalar fractalFF_kernel(scalar u, sasfit_param *sp) {
    // integrand in u = ln x of int SD(x) x^6/(1+Q^2x^2)^2 dx (without the normalisation of SD)
    scalar x = exp(u);
    return pow(x, 6.-sp->p[1])/gsl_pow_2(1.+gsl_pow_2(sp->p[0]*x));
}

scalar fractalFF_single(scalar Q, scalar a,scalar fd, scalar xmin, scalar xmax,sasfit_param * param) {
    // int_xmin^xmax SD(x) x^6/(1+Q^2 x^2)^2 dx, SD(x) = a fd x^(-1-fd)/(xmin^-fd - xmax^-fd), integrated in ln x.
    // Replaces the closed forms (special cases fd = 1..8, three-term series for Q xmax < 0.32 with errors up
    // to ~1e-2, a branch condition Q2*xmin > 3 instead of Q2*xmin2, 2F1 outside |z| < 1).
    sasfit_param sp;
    scalar norm;
    if (a == 0 || xmin == xmax) return 0.0;
    sasfit_init_param(&sp);
    sp.p[0] = Q;
    sp.p[1] = fd;
    norm = a*fd/(pow(xmin,-fd)-pow(xmax,-fd));
    if (xmin < xmax) return norm*sasfit_integrate(log(xmin), log(xmax), fractalFF_kernel, &sp);
    return -norm*sasfit_integrate(log(xmax), log(xmin), fractalFF_kernel, &sp);
}
