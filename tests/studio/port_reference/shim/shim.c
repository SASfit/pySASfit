#include <stdlib.h>
/* Support functions of sasfit_shim.h: psi override and sasfit_integrate (graded GSL QAGS, rel. tol. 1e-13). */
#include <stdarg.h>
#include <string.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include "sasfit_shim.h"

static int psi_override = 0;
static double psi_value = 0.0;

void shim_set_psi(double rad) { psi_override = 1; psi_value = rad; }
void shim_clear_psi(void) { psi_override = 0; }
scalar sasfit_param_override_get_psi(scalar default_val) { return psi_override ? psi_value : default_val; }

void sasfit_init_param(sasfit_param *param) { memset(param, 0, sizeof(*param)); }

void sasfit_param_set_err(sasfit_param *param, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    vsnprintf(param->errStr, 256, format, ap);
    va_end(ap);
    param->errStatus = 1;
}

/* integrator: 0 = reference (graded QAGS, default), 1 = plain QAGS on [a,b] with rel. tol. 1e-8, like an
 * adaptive SASfit strategy (used to show that a function no longer depends on the integrator) */
static int shim_mode = 0;
void shim_set_integrator(int mode) { shim_mode = mode; }

typedef struct { scalar (*f)(scalar, sasfit_param *); sasfit_param *param; } shim_fp;
static double shim_kernel(double x, void *v) { shim_fp *fp = (shim_fp *)v; return fp->f(x, fp->param); }

scalar sasfit_integrate(scalar a, scalar b, scalar (*f)(scalar, sasfit_param *), sasfit_param *param)
{
    /* Reference quality, independent of the integrand: GSL QAGS (rel. tol. 1e-13) on subintervals graded
     * geometrically towards both end points (a + (b-a) 10^-k, k = 1..16), so that narrow end-point layers (ring
     * kernels at large q Rg) are resolved. Plain QAGS returns 0 there, CQUAD is off by ~1e-3. */
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
    shim_fp fp = { f, param };
    gsl_function F = { &shim_kernel, &fp };
    double pts[40], sum = 0.0;
    if (isinf(b)) {  /* semi-infinite (SASfit: QAGIU); [a, a+1] graded as below, then QAGIU from a+1 */
        double res = 0.0, err = 0.0;
        gsl_integration_workspace_free(w);
        sum = sasfit_integrate(a, a + 1.0, f, param);
        w = gsl_integration_workspace_alloc(4000);
        gsl_integration_qagiu(&F, a + 1.0, 0.0, 1e-12, 4000, w, &res, &err);
        gsl_integration_workspace_free(w);
        return sum + res;
    }
    if (shim_mode == 1) {
        double res = 0.0, err = 0.0;
        gsl_integration_qags(&F, a, b, 0.0, 1e-8, 2000, w, &res, &err);
        gsl_integration_workspace_free(w);
        return res;
    }
    int n = 0, k;
    pts[n++] = a;
    for (k = 16; k >= 1; k--) pts[n++] = a + (b - a) * pow(10.0, -k);
    pts[n++] = a + 0.5 * (b - a);
    for (k = 1; k <= 16; k++) pts[n++] = b - (b - a) * pow(10.0, -k);
    pts[n++] = b;
    for (k = 0; k + 1 < n; k++) {
        double res = 0.0, err = 0.0;
        if (pts[k + 1] <= pts[k]) continue;
        gsl_integration_qags(&F, pts[k], pts[k + 1], 0.0, 1e-13, 2000, w, &res, &err);
        sum += res;
    }
    gsl_integration_workspace_free(w);
    return sum;
}

void sasfit_out(const char *format, ...) { (void)format; }
void sasfit_err(const char *format, ...) { (void)format; }
scalar sasfit_eps_get_nriq(void) { return 1e-12; }
scalar sasfit_eps_get_aniso(void) { return 1e-12; }

/* sasfit_cubature for ndim = 1 or 2: nested reference integrals (graded QAGS in x[0], QAGS in x[1]) */
typedef struct { sasfit_cubature_kernel f; void *param; size_t ndim; double y; } cub_fp;
static double cub_x(double x, void *v)
{
    cub_fp *c = (cub_fp *)v;
    double xy[2] = { x, c->y };
    return c->f(xy, c->ndim, c->param);
}
static double graded(double (*fn)(double, void *), void *v, double a, double b, double rel)
{
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
    gsl_function F = { fn, v };
    double pts[40], sum = 0.0;
    int n = 0, k;
    pts[n++] = a;
    for (k = 16; k >= 1; k--) pts[n++] = a + (b - a) * pow(10.0, -k);
    pts[n++] = a + 0.5 * (b - a);
    for (k = 1; k <= 16; k++) pts[n++] = b - (b - a) * pow(10.0, -k);
    pts[n++] = b;
    for (k = 0; k + 1 < n; k++) {
        double res = 0.0, err = 0.0;
        if (pts[k + 1] <= pts[k]) continue;
        gsl_integration_qags(&F, pts[k], pts[k + 1], 0.0, rel, 2000, w, &res, &err);
        sum += res;
    }
    gsl_integration_workspace_free(w);
    return sum;
}
typedef struct { cub_fp *c; double xmin, xmax; } cub_outer;
static double cub_y(double y, void *v)
{
    cub_outer *o = (cub_outer *)v;
    o->c->y = y;
    return graded(&cub_x, o->c, o->xmin, o->xmax, 1e-12);
}
/* ndim = 3: plain nested QAGS (x[2] outer) */
typedef struct { sasfit_cubature_kernel f; void *param; double y, z; } cub3;
static double c3_x(double x, void *v) { cub3 *c = (cub3 *)v; double xyz[3] = { x, c->y, c->z }; return c->f(xyz, 3, c->param); }
typedef struct { cub3 *c; const double *xmin, *xmax; } cub3o;
static double qags(double (*fn)(double, void *), void *v, double a, double b, double rel)
{
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
    gsl_function F = { fn, v };
    double r = 0.0, e = 0.0;
    gsl_integration_qags(&F, a, b, 0.0, rel, 2000, w, &r, &e);
    gsl_integration_workspace_free(w);
    return r;
}
static double c3_y(double y, void *v) { cub3o *o = (cub3o *)v; o->c->y = y; return qags(&c3_x, o->c, o->xmin[0], o->xmax[0], 1e-10); }
static double c3_z(double z, void *v) { cub3o *o = (cub3o *)v; o->c->z = z; return qags(&c3_y, o, o->xmin[1], o->xmax[1], 1e-9); }

int sasfit_cubature(size_t ndim, const double *xmin, const double *xmax, sasfit_cubature_kernel f, void *param,
                    double eps, double *sum, double *err)
{
    cub_fp c = { f, param, ndim, 0.0 };
    if (ndim == 3) {
        cub3 c3 = { f, param, 0.0, 0.0 };
        cub3o o = { &c3, xmin, xmax };
        *err = 0.0;
        *sum = qags(&c3_z, &o, xmin[2], xmax[2], 1e-8);
        return 0;
    }
    (void)eps;
    *err = 0.0;
    if (ndim == 1) {
        *sum = graded(&cub_x, &c, xmin[0], xmax[0], 1e-13);
    } else {
        cub_outer o = { &c, xmin[0], xmax[0] };
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
        gsl_function F = { &cub_y, &o };
        double e2 = 0.0;
        gsl_integration_qags(&F, xmin[1], xmax[1], 0.0, 1e-11, 2000, w, sum, &e2);
        gsl_integration_workspace_free(w);
    }
    return 0;
}

void sasfit_get_param(sasfit_param *param, int num, ...)
{
    va_list ap;
    int i;
    va_start(ap, num);
    for (i = 0; i < num; i++) {
        scalar *v = va_arg(ap, scalar *);
        if (v) *v = param->p[i];
    }
    va_end(ap);
}

scalar sasfit_jinc(scalar x)
{
    if (fabs(x) <= 1e-6)
        return 0.5 - gsl_pow_2(x)/16.0 + gsl_pow_4(x)/384.0 - gsl_pow_6(x)/18432.0 + gsl_pow_8(x)/1474560.0;
    return gsl_sf_bessel_J1(x) / x;
}

scalar sasfit_rwbrush_w(scalar Q, scalar R)
{
    scalar u = Q*Q * R*R, u2 = u*u;
    if (fabs(u) <= 1.0e-4) return (1 - u/10. + u2/60.)/(1. + (2.*u)/5 + u2/20.);
    return (-gsl_expm1(-u)) / u;
}

scalar sasfit_gauss_fc(scalar Q, scalar R)  /* fixed version: series up to u = 0.5 (was 1e-4) */
{
    scalar u = Q*Q * R*R, t, s;
    int k;
    if (fabs(u) < 0.5) {
        t = 1.0; s = 0.0;
        for (k = 0; k < 18; k++) { s += t; t *= -u/(k+3.); }
        return s;
    }
    return 2 * (gsl_expm1(-u) + u) / (u*u);
}

/* robertus_p < 0 selects the closed-form structure factor of stackdiscs; PORTREF_ROBERTUS_P overrides (default 1) */
scalar sasfit_eps_get_robertus_p(void)
{
    const char *e = getenv("PORTREF_ROBERTUS_P");
    return e ? atof(e) : 1.0;
}

/* Gauss hypergeometric function for x < 1 (Pfaff transformation for x < 0, GSL for 0 <= x < 1) */
scalar sasfit_2f1(scalar a, scalar b, scalar c, scalar x, sasfit_param *param)
{
    (void)param;
    if (x < 0.0) return pow(1.0 - x, -a) * gsl_sf_hyperg_2F1(a, c - b, c, x / (x - 1.0));
    return gsl_sf_hyperg_2F1(a, b, c, x);
}

/* sasfit_sq_p__q___worm_ps3_: from sasfit_src/thin_objects_worms (2026-11-23) */

/* sasfit_common/sasfit_utils.c (fixed version) */
scalar find_LogNorm_int_range(scalar dim, scalar x0, scalar sigma, scalar *Xstart, scalar *Xend, sasfit_param *param)
{
    scalar a2 = fabs(sigma), a3 = 1.0, a4 = fabs(x0), X_0, X_n;
    (void)param;
    X_0 = a4 * exp(-a2 * a2 * (a3 - 2.0 * dim));
    X_n = a4 * exp(-a2 * a2 * (a3 - 2.0 * dim) + sqrt(2.0 * a2 * a2 * log(100.0 / 0.1)));
    *Xstart = 0.0;
    *Xend = X_n;
    if ((X_n - X_0) < X_0) { *Xend = X_n; *Xstart = 2.0 * X_0 - X_n; }
    return 0;
}
/* LogNorm(x; N, s, p, mu) for p = 1 (normalised) */
scalar sasfit_sd_LogNorm(scalar x, sasfit_param *param)
{
    scalar s = param->p[1], mu = param->p[3];
    if (x <= 0) return 0.0;
    return param->p[0] / (sqrt(2.0 * M_PI) * s * x) * exp(-gsl_pow_2(log(x / mu)) / (2.0 * s * s));
}
/* erfi(z) = 2z/sqrt(pi) int_0^1 exp(z^2 s^2) ds (Gauss-Legendre, enough for the arguments used here) */
static void cerfi(double x, double y, double *re, double *im)
{
    static gsl_integration_glfixed_table *t = NULL;
    double sr = 0.0, si = 0.0, zr2 = x * x - y * y, zi2 = 2.0 * x * y;
    size_t i;
    if (!t) t = gsl_integration_glfixed_table_alloc(600);
    for (i = 0; i < 600; i++) {
        double xi, wi, s2, m;
        gsl_integration_glfixed_point(0.0, 1.0, i, &xi, &wi, t);
        s2 = xi * xi;
        m = exp(zr2 * s2);
        sr += wi * m * cos(zi2 * s2);
        si += wi * m * sin(zi2 * s2);
    }
    *re = 2.0 / sqrt(M_PI) * (x * sr - y * si);
    *im = 2.0 / sqrt(M_PI) * (x * si + y * sr);
}
double re_cerfi_z(double x, double y) { double r, i; cerfi(x, y, &r, &i); return r; }
double im_cerfi_z(double x, double y) { double r, i; cerfi(x, y, &r, &i); return i; }


/* sasfit_common/sasfit_utils.c (fixed version) */
scalar sasfit_sphere_fc(scalar Q, scalar R)
{
    scalar u = Q * R, u2 = u * u, u4 = u2 * u2;
    if (fabs(u) <= 0.1)  /* fixed 2026-11-18: was 1e-4 (cancellation in sin(u)-u cos(u) up to u ~ 0.1) */
        return (1. - (211.*u2)/2860. + (2647.*u4)/2162160.) / (1. + (15. *u2)/572. + (17. *u4)/61776.);
    return 3.0 * (sin(u) - u * cos(u)) / (u * u * u);
}

__attribute__((constructor)) static void shim_init(void) { gsl_set_error_handler_off(); }


/* ---- Ooura DE quadrature for oscillatory integrands (sasfit_common/sasfit_intde2.c), sasfit_erfinv (sasfit_utils.c) */
#include <float.h>
#ifndef MAXDOUBLE
#define MAXDOUBLE DBL_MAX
#endif
#ifndef CENTRAL_RANGE
#define CENTRAL_RANGE 0.7
#endif
void sasfit_intdeoini(int lenaw, double tiny, double eps, double *aw)
{
    /* ---- adjustable parameter ---- */
    int lmax = 5;
    double efs = 0.1, enoff = 0.40, pqoff = 2.9, ppoff = -0.72;
    /* ------------------------------ */
    int noff0, nk0, noff, k, nk, j;
    double pi4, tinyln, epsln, frq4, per2, pp, pq, ehp, ehm, h, t, 
        ep, em, tk, xw, wg, xa;
    
    pi4 = atan(1.0);
    tinyln = -log(tiny);
    epsln = 1 - log(efs * eps);
    frq4 = 1 / (2 * pi4);
    per2 = 4 * pi4;
    pq = pqoff / epsln;
    pp = ppoff - log(pq * pq * frq4);
    ehp = exp(2 * pq);
    ehm = 1 / ehp;
    aw[3] = lmax;
    aw[4] = eps;
    aw[5] = sqrt(efs * eps);
    noff0 = 6;
    nk0 = 1 + (int) (enoff * epsln);
    aw[1] = nk0;
    noff = 2 * nk0 + noff0;
    wg = 0;
    xw = 1;
    for (k = 1; k <= nk0; k++) {
        wg += xw;
        aw[noff - 2 * k] = wg;
        aw[noff - 2 * k + 1] = xw;
        xw = xw * (nk0 - k) / k;
    }
    wg = per2 / wg;
    for (k = noff0; k <= noff - 2; k += 2) {
        aw[k] *= wg;
        aw[k + 1] *= wg;
    }
    xw = exp(pp - 2 * pi4);
    aw[noff] = sqrt(xw * (per2 * 0.5));
    aw[noff + 1] = xw * pq;
    aw[noff + 2] = per2 * 0.5;
    h = 2;
    nk = 0;
    k = noff + 3;
    do {
        t = h * 0.5;
        do {
            em = exp(2 * pq * t);
            ep = pi4 * em;
            em = pi4 / em;
            tk = t;
            j = k;
            do {
                xw = exp(pp - ep - em);
                wg = sqrt(frq4 * xw + tk * tk);
                xa = xw / (tk + wg);
                wg = (pq * xw * (ep - em) + xa) / wg;
                aw[j] = xa;
                aw[j + 1] = xw * pq;
                aw[j + 2] = wg;
                ep *= ehp;
                em *= ehm;
                tk += 1;
                j += 3;
            } while (ep < tinyln && j <= lenaw - 3);
            t += h;
            k += nk;
        } while (t < 1);
        h *= 0.5;
        if (nk == 0) {
            if (j > lenaw - 6) j -= 3;
            nk = j - noff;
            k += nk;
            aw[2] = nk;
        }
    } while (2 * k - noff - 3 <= lenaw);
    aw[0] = k - 3;
}


void sasfit_intdeo(double (*f)(double, void *), double a, double omega, double *aw, 
    double *i, double *err, void *fparams)
{
    int lenawm, nk0, noff0, nk, noff, lmax, m, k, j, jm, l;
    double eps, per, perw, w02, ir, h, iback, irback, t, tk, 
        xa, fm, fp, errh, s0, s1, s2, errd;
    
    
    lenawm = (int) (aw[0] + 0.5);
    nk0 = (int) (aw[1] + 0.5);
    noff0 = 6;
    nk = (int) (aw[2] + 0.5);
    noff = 2 * nk0 + noff0;
    lmax = (int) (aw[3] + 0.5);
    eps = aw[4];
    per = 1 / fabs(omega);
    w02 = 2 * aw[noff + 2];
    perw = per * w02;
    
    *i = (*f)(a + aw[noff] * per,fparams);
    
    ir = *i * aw[noff + 1];
    
    *i *= aw[noff + 2];
    
    *err = fabs(*i);
    
    h = 2;
    m = 1;
    k = noff;
    do {
        iback = *i;
        irback = ir;
        
        t = h * 0.5;
        do {
            if (k == noff) {
                tk = 1;
                k += nk;
                j = noff;
                do {
                    j += 3;
                    xa = per * aw[j];
                    fm = (*f)(a + xa,fparams);
                    fp = (*f)(a + xa + perw * tk,fparams);
                    ir += (fm + fp) * aw[j + 1];
                    fm *= aw[j + 2];
                    fp *= w02 - aw[j + 2];
                    *i += fm + fp;
                    *err += fabs(fm) + fabs(fp);
                    tk += 1;
                } while (aw[j] > eps && j < k);
                errh = *err * aw[5];
                *err *= eps;
                jm = j - noff;
            } else {
                tk = t;
                for (j = k + 3; j <= k + jm; j += 3) {
                    xa = per * aw[j];
                    fm = (*f)(a + xa,fparams);
                    fp = (*f)(a + xa + perw * tk,fparams);
                    ir += (fm + fp) * aw[j + 1];
                    fm *= aw[j + 2];
                    fp *= w02 - aw[j + 2];
                    *i += fm + fp;
                    tk += 1;
                }
                j = k + jm;
                k += nk;
            }
            while (fabs(fm) > *err && j < k) {
                j += 3;
                fm = (*f)(a + per * aw[j],fparams);
                ir += fm * aw[j + 1];
                fm *= aw[j + 2];
                *i += fm;
            }
            fm = (*f)(a + perw * tk,fparams);
            s2 = w02 * fm;
            *i += s2;
            if (fabs(fp) > *err || fabs(s2) > *err) {
                l = 0;
                for (;;) {
                    l++;
                    s0 = 0;
                    s1 = 0;
                    s2 = fm * aw[noff0 + 1];
                    for (j = noff0 + 2; j <= noff - 2; j += 2) {
                        tk += 1;
                        fm = (*f)(a + perw * tk,fparams);
                        s0 += fm;
                        s1 += fm * aw[j];
                        s2 += fm * aw[j + 1];
                    }
                    if (s2 <= *err || l >= lmax) break;
                    *i += w02 * s0;
                }
                *i += s1;
                if (s2 > *err) *err = s2;
            }
            t += h;
        } while (t < 1);
        if (m == 1) {
            errd = 1 + 2 * errh;
        } else {
            errd = h * (fabs(*i - 2 * iback) + fabs(ir - 2 * irback));
        }
        h *= 0.5;
        m *= 2;
    } while (errd > errh && 2 * k - noff <= lenawm);
    *i *= h * per;
    if (errd > errh) {
        *err = -errd * per;
    } else {
        *err *= per * m * 0.5;
    }
}

scalar sasfit_erfinv(scalar y)
{
        double x,z,num,dem; /*working variables */
        /* coefficients in rational expansion */
        double a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
        double b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
        double c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
        double d[2]={ 3.543889200,  1.637067800};
        if(fabs(y) > 1.0) return (atof("NaN"));  /* This needs IEEE constant*/
        if(fabs(y) == 1.0) return((copysign(1.0,y))*MAXDOUBLE);
        if( fabs(y) <= CENTRAL_RANGE )
        {
                z = y*y;
                num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
                dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0])*z + 1.0);
                x = y*num/dem;
        }
        else if( (fabs(y) > CENTRAL_RANGE) && (fabs(y) < 1.0) )
        {
                z = sqrt(-log((1.0-fabs(y))/2.0));
                num = ((c[3]*z + c[2])*z + c[1])*z + c[0];
                dem = (d[1]*z + d[0])*z + 1.0;
                x = (copysign(1.0,y))*num/dem;
        }
        /* Two steps of Newton-Raphson correction */
        x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));
        x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));

        return(x);
}



/* hcubature / pcubature (fdim = 1, dim <= 2) by the nested reference integrals of sasfit_cubature */
typedef struct { integrand f; void *fdata; } jc_wrap;
static double jc_kernel(const double *x, size_t ndim, void *v)
{
    jc_wrap *w = (jc_wrap *)v;
    double val = 0.0;
    w->f((unsigned)ndim, x, w->fdata, 1, &val);
    return val;
}
int sasfit_get_int_strategy(void) { return P_CUBATURE; }
int hcubature(unsigned fdim, integrand f, void *fdata, unsigned dim, const double *xmin, const double *xmax,
              size_t maxEval, double reqAbsError, double reqRelError, error_norm norm, double *val, double *err)
{
    jc_wrap w = { f, fdata };
    (void)fdim; (void)maxEval; (void)reqAbsError; (void)reqRelError; (void)norm;
    return sasfit_cubature(dim, xmin, xmax, &jc_kernel, &w, 1e-12, val, err);
}
int pcubature(unsigned fdim, integrand f, void *fdata, unsigned dim, const double *xmin, const double *xmax,
              size_t maxEval, double reqAbsError, double reqRelError, error_norm norm, double *val, double *err)
{
    return hcubature(fdim, f, fdata, dim, xmin, xmax, maxEval, reqAbsError, reqRelError, norm, val, err);
}


/* ---- batch 7: Struve H0, H1 (series for x < 6, else H - Y by the Laplace integrals
 *      H0 - Y0 = 2/pi int_0^inf e^{-xt} / sqrt(1+t^2) dt,  H1 - Y1 = 2x/pi int_0^inf e^{-xt} sqrt(1+t^2) dt) */
#include <gsl/gsl_sf_bessel.h>
typedef struct { double x; int nu; } struve_lp;
static double struve_kernel(double t, void *v)
{
    struve_lp *s = (struve_lp *)v;
    double r = sqrt(1.0 + t * t);
    return exp(-s->x * t) * (s->nu == 0 ? 1.0 / r : r);
}
static double struve_series(double x, int nu)
{
    /* H_nu(x) = sum_k (-1)^k (x/2)^(2k+nu+1) / (Gamma(k+3/2) Gamma(k+nu+3/2)) */
    long double h = x / 2.0L, term, sum = 0.0L;
    long double g1 = sqrtl(M_PI) / 2.0L;                    /* Gamma(3/2) */
    long double g2 = (nu == 0) ? g1 : g1 * 1.5L;            /* Gamma(nu + 3/2) */
    int k;
    term = powl(h, nu + 1) / (g1 * g2);
    for (k = 0; k < 200; k++) {
        sum += term;
        term *= -h * h / ((k + 1.5L) * (k + nu + 1.5L));
        if (fabsl(term) < 1e-21L * fabsl(sum)) break;
    }
    return (double)sum;
}
static double struve(double x, int nu)
{
    if (x < 6.0) return struve_series(x, nu);
    {
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(2000);
        struve_lp s = { x, nu };
        gsl_function F = { &struve_kernel, &s };
        double r = 0.0, e = 0.0;
        gsl_integration_qagiu(&F, 0.0, 0.0, 1e-12, 2000, w, &r, &e);
        gsl_integration_workspace_free(w);
        return nu == 0 ? gsl_sf_bessel_Y0(x) + 2.0 / M_PI * r : gsl_sf_bessel_Y1(x) + 2.0 * x / M_PI * r;
    }
}
scalar sasfit_struve_h0(scalar *x) { return struve(*x, 0); }
scalar sasfit_struve_h1(scalar *x) { return struve(*x, 1); }

/* sasfit_orient_avg: mean of f(theta, phi) over cos(theta) in [cos(theta_max), 1], phi in [0, phi_max]
 * (nested reference integrals, uniform in cos(theta), as the cubature branch of sasfit_orient_avg_ctm) */
static double polar_phi = 2.0 * M_PI, polar_theta = M_PI;
void sasfit_param_set_polar_phi(scalar v) { polar_phi = v; }
void sasfit_param_set_polar_theta(scalar v) { polar_theta = v; }
scalar sasfit_param_get_polar_phi(void) { return polar_phi; }
scalar sasfit_param_get_polar_theta(void) { return polar_theta; }
typedef struct { sasfit_func_two_t *f; sasfit_param *param; } oavg;
static double oavg_kernel(const double *x, size_t ndim, void *v)
{
    oavg *o = (oavg *)v;
    (void)ndim;
    return o->f(acos(x[0]), 2.0 * M_PI * x[1], o->param);
}
scalar sasfit_orient_avg(sasfit_func_two_t *f, sasfit_param *param)
{
    oavg o = { f, param };
    double xmin[2] = { cos(polar_theta), 0.0 }, xmax[2] = { 1.0, polar_phi / (2.0 * M_PI) }, s = 0.0, e = 0.0;
    sasfit_cubature(2, xmin, xmax, &oavg_kernel, &o, 1e-12, &s, &e);
    return s / ((1.0 - cos(polar_theta)) * polar_phi / (2.0 * M_PI));
}

/* ---- batch 9: sasfit_invert_func_v (root of f(y, dist) = u on [bl, bu], bisection to full precision),
 * eps_get_comp, erfcinv, gammaincinv (not needed by the compared functions: NaN), r8_besk (dead code in GIG) */
scalar sasfit_invert_func_v(scalar u, sasfit_func_vol_t *f, int dist, scalar bl, scalar bu, sasfit_param *param)
{
    scalar a = bl, b = bu, fa, m, fm;
    int i;
    fa = f(a, param, dist) - u;
    if (fa >= 0) return a;
    if (f(b, param, dist) - u <= 0) return b;
    for (i = 0; i < 200; i++) {
        m = 0.5 * (a + b);
        if (m == a || m == b) break;
        fm = f(m, param, dist) - u;
        if ((fm < 0) == (fa < 0)) { a = m; fa = fm; } else { b = m; }
    }
    return 0.5 * (a + b);
}
scalar sasfit_eps_get_comp(void) { return 1e-14; }
scalar sasfit_erfcinv(scalar y) { return sasfit_erfinv(1.0 - y); }
scalar sasfit_gammaincinv(scalar a, scalar y) { (void)a; (void)y; return NAN; }
double r8_besk(double nu, double x) { return gsl_sf_bessel_Knu(fabs(nu), x); }
