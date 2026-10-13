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
