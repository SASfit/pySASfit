/* mock libsasfit: common-function table, precision setters, built-in functions */
#include "mock_common.h"
static scalar twice(scalar x) { return 2.0 * x; }
static void set_err(sasfit_param *p, const char *msg) {
    p->errStatus = 1; strncpy(p->errStr, msg, STRLEN - 1); p->errLen = (int)strlen(p->errStr);
}
static sasfit_common_stubs_t stubs = {twice, set_err};
const sasfit_common_stubs_t *sasfit_common_stubs_ptr(void) { return &stubs; }

static scalar eps_nriq = 0.0; static int int_strategy = -1;
void sasfit_eps_set_nriq(scalar v) { eps_nriq = v; }
scalar sasfit_eps_get_nriq(void) { return eps_nriq; }
void sasfit_set_int_strategy(int v) { int_strategy = v; }
int sasfit_get_int_strategy(void) { return int_strategy; }
int sasfit_param_size(void) { return (int)sizeof(sasfit_param); }

scalar sasfit_ff_sphere_f(scalar q, sasfit_param *param) {
    scalar R = param->p[0], eta = param->p[3];
    if (R < 0) { set_err(param, "R < 0"); return 0.0; }
    if (q * R < 1e-4) return eta * 4.0 / 3.0 * M_PI * R * R * R;
    return eta * 4.0 * M_PI * (sin(q * R) - q * R * cos(q * R)) / (q * q * q);
}
scalar sasfit_ff_sphere(scalar q, sasfit_param *param) { scalar f = sasfit_ff_sphere_f(q, param); return f * f; }
scalar sasfit_ff_sphere_v(scalar x, sasfit_param *param, int distr) {
    if (distr != 0) x = param->p[0];
    return 4.0 / 3.0 * M_PI * x * x * x;
}
scalar sasfit_sd_LogNorm(scalar x, sasfit_param *param) {
    scalar N = param->p[0], s = param->p[1], P = param->p[2], R = param->p[3];
    if (x <= 0) return 0.0;
    scalar c = 1.0 / sqrt(M_PI * 2.0) / fabs(s) / pow(R, 1.0 - P) / exp(0.5 * pow(s * (1.0 - P), 2.0));
    return N * c * pow(x, -P) * exp(-0.5 * pow(log(x / R) / s, 2.0));
}
/* Percus-Yevick hard spheres, radius p[0], volume fraction p[1] */
scalar sasfit_sq_HardSphere(scalar q, sasfit_param *param) {
    scalar R = param->p[0], f = param->p[1];
    scalar a = 2.0 * q * R;
    if (f == 0.0) return 1.0;
    scalar al = pow(1 + 2 * f, 2) / pow(1 - f, 4), be = -6 * f * pow(1 + f / 2, 2) / pow(1 - f, 4), ga = f * al / 2;
    if (a < 1e-3) return pow(1 - f, 4) / pow(1 + 2 * f, 2);
    scalar s = sin(a), c = cos(a);
    scalar G = al * (s - a * c) / (a * a) + be * (2 * a * s + (2 - a * a) * c - 2) / (a * a * a)
             + ga * (-pow(a, 4) * c + 4 * ((3 * a * a - 6) * c + (pow(a, 3) - 6 * a) * s + 6)) / pow(a, 5);
    return 1.0 / (1.0 + 24.0 * f * G / a);
}

/* sasfit_integrate_ctm and sasfit_hankel stand-ins (Simpson rule up to where f has decayed): only the plumbing of the
   Python wrappers is tested with them, not SASfit's strategies. */
typedef double hankel_f(double, void *);
static double cutoff(double (*g)(double, void *), void *p) {
    double kmax = 0.0, peak = 0.0, k;
    for (k = 1e-6; k < 1e6; k *= 1.1) { double v = fabs(k * g(k, p)); if (v > peak) peak = v; }
    for (k = 1e-6; k < 1e6; k *= 1.1) { double v = fabs(k * g(k, p)); if (v > 1e-13 * peak) kmax = k; }
    return kmax > 0 ? kmax * 1.5 : 1.0;
}
static double simpson(double (*g)(double, void *), void *p, double a, double b, double x, int bessel) {
    int n = 8000, i;
    double h = (b - a) / n, s = 0.0;
    for (i = 0; i <= n; i++) {
        double k = a + i * h, w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
        double v = g(k, p) * (bessel ? k * j0(k * x) : 1.0);
        s += w * v;
    }
    return s * h / 3.0;
}
double sasfit_hankel(double nu, hankel_f *f, double x, void *param) {
    (void)nu;
    return simpson(f, param, 0.0, cutoff(f, param), x, 1);
}
typedef double one_f(double, sasfit_param *);
double sasfit_integrate_ctm(double a, double b, one_f *f, sasfit_param *param, int limit, double epsabs, double epsrel) {
    (void)limit; (void)epsabs; (void)epsrel;
    if (isinf(b)) b = cutoff((double (*)(double, void *))f, param);
    return simpson((double (*)(double, void *))f, param, a, b, 0.0, 0);
}
