/* shim: replaces the SASfit private.h of plugin blob (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_BLOB_PRIVATE_H
#define SHIM_BLOB_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_blob__beaucage_(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__beaucage__f(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__beaucage__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_blob__dendrimer_(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__dendrimer__f(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__dendrimer__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_blob__star_(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__star__f(scalar q, sasfit_param * param);
scalar sasfit_ff_blob__star__v(scalar q, sasfit_param * param, int dist);

//
// add local defines here:
// #define P0 param->p[0]
//
#endif
