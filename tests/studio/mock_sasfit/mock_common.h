/* Minimal stand-ins for the SASfit headers, with the same binary layout. */
#include <math.h>
#include <string.h>
#include <stdio.h>
#define MAXPAR 50
#define STRLEN 256
typedef double scalar;
typedef struct {
    scalar p[MAXPAR];
    int kernelSelector, kernelSubSelector;
    char errStr[STRLEN];
    int errLen;
    int errStatus; /* bool is int in SASfit (bool.h) */
    double *xarr, *yarr;
    void *moreparam;
    scalar more_p[MAXPAR];
    double (*function)(double, void *);
} sasfit_param;
typedef scalar sasfit_func_one_t(scalar, sasfit_param *);
typedef scalar sasfit_func_vol_t(scalar, sasfit_param *, int);
typedef struct { int len; const char *name; sasfit_func_one_t *func, *func_f; sasfit_func_vol_t *func_v; } sasfit_plugin_func_t;
typedef struct { int num; sasfit_plugin_func_t functions[]; } sasfit_plugin_info_t;
typedef int sasfit_plugin_search_t(sasfit_plugin_info_t *);
typedef struct { scalar (*twice)(scalar); void (*set_err)(sasfit_param *, const char *); } sasfit_common_stubs_t;
