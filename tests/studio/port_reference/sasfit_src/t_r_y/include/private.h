/* shim: replaces the SASfit private.h of plugin t_r_y (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_T_R_Y_PRIVATE_H
#define SHIM_T_R_Y_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_egu_w_l_profile(scalar x, sasfit_param * param);
scalar sasfit_ff_egu_w_l_profile_f(scalar x, sasfit_param * param);
scalar sasfit_ff_egu_w_l_profile_v(scalar u, sasfit_param * param, int dist);
scalar egu_w_l_sphere(scalar y, sasfit_param *param);
scalar sasfit_ff_egu_w_l_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_egu_w_l_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_egu_w_l_sphere_v(scalar u, sasfit_param * param, int dist);
scalar sasfit_ff_fgld_profile(scalar x, sasfit_param * param);
scalar sasfit_ff_fgld_profile_f(scalar x, sasfit_param * param);
scalar sasfit_ff_fgld_profile_v(scalar u, sasfit_param * param, int dist);
scalar fgld_profile_sphere(scalar y, sasfit_param *param);
scalar fgld_profile_void(scalar y, void *para);
scalar sasfit_ff_fgld_profile_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_fgld_profile_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fgld_profile_sphere_v(scalar u, sasfit_param * param, int dist);
scalar fgld_sphere(scalar y, sasfit_param *param);
scalar sasfit_ff_fgld_sphere(scalar q, sasfit_param * param);
scalar sasfit_ff_fgld_sphere_f(scalar q, sasfit_param * param);
scalar sasfit_ff_fgld_sphere_v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_e_l_ll_(scalar q, sasfit_param * param);
scalar sasfit_sd_e_l_ll__f(scalar q, sasfit_param * param);
scalar sasfit_sd_e_l_ll__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_egu_w_l(scalar x, sasfit_param * param);
scalar sasfit_sd_egu_w_l_f(scalar x, sasfit_param * param);
scalar sasfit_sd_egu_w_l_v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_fgld_(scalar x, sasfit_param * param);
scalar sasfit_sd_fgld__f(scalar x, sasfit_param * param);
scalar sasfit_sd_fgld__v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_g_c_e__(scalar q, sasfit_param * param);
scalar sasfit_sd_g_c_e___f(scalar q, sasfit_param * param);
scalar sasfit_sd_g_c_e___v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_ll_l_gl__(scalar q, sasfit_param * param);
scalar sasfit_sd_ll_l_gl___f(scalar q, sasfit_param * param);
scalar sasfit_sd_ll_l_gl___v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_n_l_c_(scalar x, sasfit_param * param);
scalar sasfit_sd_n_l_c__f(scalar x, sasfit_param * param);
scalar sasfit_sd_n_l_c__v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_n_u_gl__(scalar x, sasfit_param * param);
scalar sasfit_sd_n_u_gl___f(scalar x, sasfit_param * param);
scalar sasfit_sd_n_u_gl___v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_p_w_gl__(scalar q, sasfit_param * param);
scalar sasfit_sd_p_w_gl___f(scalar q, sasfit_param * param);
scalar sasfit_sd_p_w_gl___v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_u_e_gl__(scalar x, sasfit_param * param);
scalar sasfit_sd_u_e_gl___f(scalar x, sasfit_param * param);
scalar sasfit_sd_u_e_gl___v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sd_w_l_e_(scalar x, sasfit_param * param);
scalar sasfit_sd_w_l_e__f(scalar x, sasfit_param * param);
scalar sasfit_sd_w_l_e__v(scalar u, sasfit_param * param, int dist);
scalar Q_Uniform(scalar p);
scalar F_Uniform(scalar x);
scalar f_Uniform(scalar x);
scalar Q_Weibull(scalar p, scalar alpha, scalar beta);
scalar F_Weibull(scalar x, scalar alpha, scalar beta);
scalar f_Weibull(scalar x, scalar alpha, scalar beta);
scalar Q_Exponential(scalar p, scalar lambda);
scalar F_Exponential(scalar x, scalar lambda);
scalar f_Exponential(scalar x, scalar lambda);
scalar Q_ExtremeValue(scalar p, scalar a, scalar b);
scalar Q_Lindley(scalar p, scalar theta);
scalar f_Lindley(scalar x, scalar theta);
scalar F_Lindley(scalar x, scalar theta);
scalar Q_LogLogistic(scalar p, scalar a, scalar b);
scalar f_LogLogistic(scalar x, scalar a, scalar b);
scalar F_LogLogistic(scalar x, scalar a, scalar b);
scalar Q_Logistic(scalar p, scalar mu, scalar sigma);
scalar f_Logistic(scalar x, scalar mu, scalar sigma);
scalar F_Logistic(scalar x, scalar mu, scalar sigma);
scalar Q_Cauchy(scalar p);
scalar f_Cauchy(scalar x);
scalar F_Cauchy(scalar x);
scalar Q_Gumbel(scalar p);
scalar f_Gumbel(scalar x);
scalar F_Gumbel(scalar x);
scalar Q_Gamma(scalar y, scalar alpha, scalar beta);
scalar f_Gamma(scalar x, scalar alpha, scalar beta);
scalar F_Gamma(scalar x, scalar alpha, scalar beta);
scalar Q_Frechet(scalar p, scalar k);
scalar f_Frechet(scalar x, scalar k);
scalar F_Frechet(scalar x, scalar k);
scalar Q_Rayleigh(scalar p);
scalar f_Rayleigh(scalar x);
scalar F_Rayleigh(scalar x);
scalar Q_Pareto(scalar p, scalar alpha);
scalar f_Pareto(scalar x, scalar alpha);
scalar F_Pareto(scalar x, scalar alpha);
scalar KWm1(scalar Z, scalar theta);
scalar Qx(scalar y, sasfit_func_one_t *Fy, sasfit_func_one_t *QT, sasfit_param *param);
scalar QxWLE(scalar y, sasfit_param *param);
scalar Lindley_f_R(scalar x, scalar theta);
scalar Lindley_F_R(scalar x, scalar theta);

//
// add local defines here:
// #define P0 param->p[0]
//

#define N	param->p[0]

#define BL	GSL_MIN(param->p[1],param->p[2])
#define BU	GSL_MAX(param->p[1],param->p[2])

#define DIM	    param->p[3]
#define RSCALE	param->p[3]

#define KAPPA	fabs(param->p[4])
#define THETA	param->p[4]
#define S	    param->p[4]
#define W	    param->p[4]
#define LAMBDA2	param->p[4]

#define ALPHA	param->p[5]
#define C	    param->p[5]
#define LAMBDA3	param->p[5]

#define BETA	param->p[6]
#define LAMBDA4	param->p[6]

#define GAMMA	param->p[7]
#define MU		param->p[7]

#define DELTA	param->p[8]
#define SIGMA	param->p[8]
#define LAMBDA 	param->p[8]




#define Q       param->p[MAXPAR-1]
#define YSTART  param->p[MAXPAR-2]
#define P       param->p[MAXPAR-3]


#define Power(x, y)	(pow((scalar)(x), (scalar)(y)))
#define Sqrt(x)		(sqrt((scalar)(x)))
#define Sqrtl(x)        (sqrt((scalar)(x)))

#define Abs(x)		(fabs((scalar)(x)))

#define Exp(x)		(exp((scalar)(x)))
#define Log(x)		(log((scalar)(x)))

#define Sin(x)		(sin((scalar)(x)))
#define Cos(x)		(cos((scalar)(x)))
#define Tan(x)		(tan((scalar)(x)))

#define ArcSin(x)       (asin((scalar)(x)))
#define ArcCos(x)       (acos((scalar)(x)))
#define ArcTan(x)       (atan((scalar)(x)))
#define ArcCot(x)       (atan((scalar)(1/x)))

#define Sinh(x)          (sinh((scalar)(x)))
#define Cosh(x)          (cosh((scalar)(x)))
#define Tanh(x)          (tanh((scalar)(x)))

#define Cot(x)          (1./tan((scalar)(x)))
#define Csc(x)          (1./sin((scalar)(x)))
#define Sec(x)          (1./cos((scalar)(x)))

#define E       2.71828182845904523536029
#define Pi      3.14159265358979323846264
#define Degree      0.01745329251994329576924

#define GSLPOWN(x,y)    (gsl_pow_int(x,y))
#define InverseErfc(x)  (sasfit_erfcinv(x))
#define ProductLog(x)   (gsl_sf_lambert_Wm1(x))


enum TRY_QY_fY_type {
    Y_UNIFORM,
    Y_FRECHET,
    Y_LOGISTIC,
    Y_LOGLOGISTIC,
    Y_PARETO,
    Y_RAYLEIGH,
    Y_EXTREMEVALUE,
    Y_CAUCHY,
    Y_EXPONENTIAL,
    Y_WEIBULL,
    Y_GAMMA
};

scalar KWm1(scalar, scalar);
scalar Lindley_f_R(scalar, scalar);
scalar Lindley_F_R(scalar, scalar);
scalar f_Gamma(scalar, scalar, scalar);
scalar F_Gamma(scalar, scalar, scalar);
#endif
