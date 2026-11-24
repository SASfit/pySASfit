/* shim: replaces the SASfit private.h of plugin fuzzysphere (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_FUZZYSPHERE_PRIVATE_H
#define SHIM_FUZZYSPHERE_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_expshell(scalar q, sasfit_param * param);
scalar sasfit_ff_expshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_expshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_kshexp1(scalar q, sasfit_param * param);
scalar sasfit_kshexp2(scalar q, sasfit_param * param);
scalar sasfit_ff_linshell(scalar q, sasfit_param * param);
scalar sasfit_ff_linshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_linshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_linshell2(scalar q, sasfit_param * param);
scalar sasfit_ff_linshell2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_linshell2_v(scalar q, sasfit_param * param, int dist);
scalar ampl(scalar q, scalar rad, scalar sigma, scalar contrast);
scalar sasfit_ff_coreshellmicrogel(scalar q, sasfit_param * param);
scalar sasfit_ff_coreshellmicrogel_f(scalar q, sasfit_param * param);
scalar sasfit_ff_coreshellmicrogel_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_fuzzysphere(scalar q, sasfit_param * param);
scalar sasfit_ff_fuzzysphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fuzzysphere_v(scalar q, sasfit_param * param, int dist);
scalar expprof(scalar x, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_expshell(scalar r, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_expshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_expshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_radial_profile_of_linshell(scalar r, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_linshell_f(scalar q, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_linshell_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_radial_profile_of_linshell2(scalar r, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_linshell2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_linshell2_v(scalar q, sasfit_param * param, int dist);
scalar prof(scalar x,scalar ir,scalar isigma);
scalar sasfit_ff_radial_profile_of_coreshellmicrogel(scalar r, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_coreshellmicrogel_f(scalar q, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_coreshellmicrogel_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_radial_profile_of_fuzzysphere(scalar r, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_fuzzysphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_radial_profile_of_fuzzysphere_v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
//
// stable helpers (2026-11-23): sphere amplitude phi(x) = 3(sin x - x cos x)/x^3 and
// chi(x) = (2x sin x - (x^2-2) cos x - 2)/x^4 = int_0^1 t^3 sinc(x t) dt by series for x < 1,
// radial zone integrals 4 pi int_a^b rho(r) r^2 sinc(q r) dr by 16-point Gauss-Legendre on panels of at most 2 rad
//
static const double fz_gl_x[8] = {0.09501250983763744, 0.2816035507792589, 0.45801677765722737, 0.6178762444026438,
	0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499};
static const double fz_gl_w[8] = {0.18945061045506864, 0.18260341504492364, 0.16915651939500265, 0.1495959888165767,
	0.12462897125553407, 0.0951585116824926, 0.062253523938647456, 0.027152459411754176};

static double fz_phi(double x)
{
	double x2, t, s;
	int k;
	x = fabs(x);
	if (x >= 1.0) return 3.0*(sin(x)-x*cos(x))/(x*x*x);
	x2 = x*x; t = 1.0; s = 1.0/3.0;
	for (k = 1; k < 12; k++) {
		t *= -x2/((2.0*k)*(2.0*k+1.0));
		s += t/(2.0*k+3.0);
	}
	return 3.0*s;
}

static double fz_chi(double x)
{
	double x2, t, s;
	int k;
	x = fabs(x);
	if (x >= 1.0) return (2.0*x*sin(x)-(x*x-2.0)*cos(x)-2.0)/gsl_pow_4(x);
	x2 = x*x; t = 1.0; s = 0.25;
	for (k = 1; k < 12; k++) {
		t *= -x2/((2.0*k)*(2.0*k+1.0));
		s += t/(2.0*k+4.0);
	}
	return s;
}

static double fz_sinc(double x)
{
	return (fabs(x) < 1e-4) ? 1.0 - x*x/6.0 : sin(x)/x;
}

// contrast * 4/3 pi r^3 phi(q r)
static double fz_sphere(double q, double r, double contrast)
{
	r = fabs(r);
	return contrast*4.0/3.0*M_PI*r*r*r*fz_phi(q*r);
}

// 4 pi int_a^b rho(r) r^2 sinc(q r) dr
static double fz_zone(double q, double a, double b, double (*rho)(double, void *), void *par)
{
	double h, c, r, s, sum;
	int n, i, k, j;
	if (a < 0) a = 0;
	if (!(b > a)) return 0.0;
	n = (int) ceil(q*(b-a)/2.0);
	if (n < 1) n = 1;
	h = 0.5*(b-a)/n;
	sum = 0.0;
	for (i = 0; i < n; i++) {
		c = a + (2.0*i+1.0)*h;
		for (k = 0; k < 8; k++) {
			for (j = -1; j <= 1; j += 2) {
				r = c + j*h*fz_gl_x[k];
				sum += fz_gl_w[k]*r*r*rho(r, par)*fz_sinc(q*r);
			}
		}
	}
	return 4.0*M_PI*h*sum;
}
#endif
