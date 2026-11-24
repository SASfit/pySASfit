/* shim: replaces the SASfit private.h of plugin metalog (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_METALOG_PRIVATE_H
#define SHIM_METALOG_PRIVATE_H
#include <sasfit_shim.h>

scalar mk(int k, scalar y, scalar *a, size_t n, sasfit_param * param);
scalar Mk(int k, scalar y, scalar *a, size_t n, sasfit_param * param);
scalar MLogit(int k, scalar y, scalar *a, size_t n,scalar bl, scalar bu, sasfit_param * param);
scalar MLog(int k, scalar y, scalar *a, size_t n, scalar bl, sasfit_param * param);
scalar MNLog(int k, scalar y, scalar *a, size_t n, scalar bu, sasfit_param * param);
scalar mLogit(int k, scalar y, scalar *a, size_t n, scalar bl, scalar bu, sasfit_param * param);
scalar mLog(int k, scalar y, scalar *a, size_t n, sasfit_param * param);
scalar mNLog(int k, scalar y, scalar *a, size_t n, sasfit_param * param);
scalar sasfit_sd_metalog_clipped(scalar x, sasfit_param * param);
scalar sasfit_sd_metalog_clipped_f(scalar q, sasfit_param * param);
scalar sasfit_sd_metalog_clipped_v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_metalog_log_clipped(scalar x, sasfit_param * param);
scalar sasfit_sd_metalog_log_clipped_f(scalar q, sasfit_param * param);
scalar sasfit_sd_metalog_log_clipped_v(scalar u, sasfit_param * param, int dist);
scalar sasfit_sd_metalog_logit(scalar x, sasfit_param * param);
scalar sasfit_sd_metalog_logit_f(scalar q, sasfit_param * param);
scalar sasfit_sd_metalog_logit_v(scalar u, sasfit_param * param, int dist);

//
// add local defines here:
#define N	param->p[0]
#define BL	GSL_MIN(param->p[1],param->p[2])
#define BU	GSL_MAX(param->p[1],param->p[2])
#define ALPHA	param->p[3]
#define A1	param->p[4]
#define A2	param->p[5]
#define A3	param->p[6]
#define A4	param->p[7]
#define A5	param->p[8]
#define A6	param->p[9]
#define A7	param->p[10]
#define A8	param->p[11]
#define A9	param->p[12]
#define A10	param->p[13]
#define Q   param->p[MAXPAR-1]

scalar Mk(int , scalar , scalar *, size_t , sasfit_param *);
scalar MLog(int , scalar , scalar *, size_t , scalar, sasfit_param *);
scalar MNLog(int , scalar , scalar *, size_t , scalar, sasfit_param *);
scalar MLogit(int , scalar , scalar *, size_t , scalar, scalar, sasfit_param *);

scalar mk(int , scalar , scalar *, size_t , sasfit_param *);
scalar mLog(int , scalar , scalar *, size_t , sasfit_param *);
scalar mNLog(int , scalar , scalar *, size_t , sasfit_param *);
scalar mLogit(int , scalar , scalar *, size_t ,  scalar, scalar, sasfit_param *);
#endif
