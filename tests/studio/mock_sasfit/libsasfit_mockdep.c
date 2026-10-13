/* mock plugin importing ff_mock_scaled from another plugin (alphabetically loaded first: 'mockdep' < 'mockplug') */
#include "mock_common.h"
static sasfit_plugin_info_t plugin_import = {1, {{14, "ff_mock_scaled", 0, 0, 0}}};
scalar sasfit_sq_mock_dep(scalar q, sasfit_param *param) {
    return 1.0 + 0.01 * plugin_import.functions[0].func(q, param);
}
static sasfit_plugin_info_t plugin_export = {1, {{11, "sq_mock_dep", sasfit_sq_mock_dep, 0, 0}}};
int get_ver(void) { return 1; }
int do_init(const sasfit_plugin_info_t **exp_ptr, const sasfit_common_stubs_t *ptr, sasfit_plugin_search_t *searchf) {
    if (!exp_ptr || !ptr || !searchf) return 0;
    *exp_ptr = &plugin_export;
    return searchf(&plugin_import);
}
