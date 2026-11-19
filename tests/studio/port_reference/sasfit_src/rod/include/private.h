/* shim: replaces the SASfit private.h of plugin rod (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_ROD_PRIVATE_H
#define SHIM_ROD_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_rod(scalar q, sasfit_param * param);
scalar sasfit_ff_rod_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rod_v(scalar q, sasfit_param * param, int dist);

#endif
