/* shim: replaces the SASfit private.h of plugin stochastic_models (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_STOCHASTIC_MODELS_PRIVATE_H
#define SHIM_STOCHASTIC_MODELS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_boolean_intersection_model(scalar q, sasfit_param * param);
scalar sasfit_ff_boolean_intersection_model_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boolean_intersection_model_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_boolean_union_model(scalar q, sasfit_param * param);
scalar sasfit_ff_boolean_union_model_f(scalar q, sasfit_param * param);
scalar sasfit_ff_boolean_union_model_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_1(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_2(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_3(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_3_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_4(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_4_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_4_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_5(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_5_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_5_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_1(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_2(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_3(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_3_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_4(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_4_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_4_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_5(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_5_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_5_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_1(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_1_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_1_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_2(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_2_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_2_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_3(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_3_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_3_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_4(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_4_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_4_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_5(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_5_f(scalar q, sasfit_param * param);
scalar sasfit_ff_clipped_random_waves_twocut_intersection_5_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_dead_leaves_model(scalar q, sasfit_param * param);
scalar sasfit_ff_dead_leaves_model_f(scalar q, sasfit_param * param);
scalar sasfit_ff_dead_leaves_model_v(scalar q, sasfit_param * param, int dist);
scalar gy1(scalar r, sasfit_param *param);
scalar gy2(scalar r, sasfit_param *param);
scalar gy3(scalar r, sasfit_param *param);
scalar gy5(scalar r, sasfit_param *param);
scalar C11kernel(scalar x, sasfit_param *param);
scalar C11_twocut_kernel(scalar x, sasfit_param *param);
scalar gama(scalar r, void *pam);
scalar gamaY1(scalar r, void *pam);
scalar gamaY2(scalar r, void *pam);
scalar gamaY3(scalar r, void *pam);
scalar gamaY4(scalar r, void *pam);
scalar gamaY5(scalar r, void *pam);
scalar gamaY1_twocut(scalar r, void *pam);
scalar gamaY2_twocut(scalar r, void *pam);
scalar gamaY3_twocut(scalar r, void *pam);
scalar gamaY4_twocut(scalar r, void *pam);
scalar gamaY5_twocut(scalar r, void *pam);
double ThetaK(scalar r, sasfit_param * param);
scalar gammaBoolean(scalar r, void *pam);
scalar gammaDeadLeaves(scalar r, void *pam);

typedef scalar sasfit_sm_f(scalar , sasfit_param *);

typedef struct
{
	sasfit_param *param;
	sasfit_sm_f *gy;
	sasfit_sm_f *C11kernel;
	int sm_intersect;
} sm_param;
//
// add local defines here:
// #define P0 param->p[0]
//
#define SCALE	param->p[0]
#define FP	param->p[1]
#define MU	param->p[2]
#define DD	param->p[2]
#define SIGMA	param->p[3]
#define XI	param->p[3]
#define RM	param->p[4]
#define Q   param->p[MAXPAR-1]
#define ALPHA param->p[MAXPAR-2]
#define ASIN_GR param->p[MAXPAR-3]
#define AA	(2.*M_PI/DD)
#define BB (1./XI)
#define CC (1./RM)
scalar gammaBoolean(scalar, void *);
scalar gammaDeadLeaves(scalar, void *);
double ThetaK(scalar , sasfit_param *);
scalar gamaY1(scalar, void *);
scalar gamaY2(scalar, void *);
scalar gamaY3(scalar, void *);
scalar gamaY4(scalar, void *);
scalar gamaY5(scalar, void *);

scalar gy1(scalar, sasfit_param *);
scalar gy2(scalar, sasfit_param *);
scalar gy3(scalar, sasfit_param *);
scalar gy4(scalar, sasfit_param *);
scalar gy5(scalar, sasfit_param *);

scalar C11kernel(scalar, sasfit_param *);
scalar C11_twocut_kernel(scalar, sasfit_param *);
scalar gama(scalar, void *);
#endif
