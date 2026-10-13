/* mock plugin: uses the common-function table, so it only works after do_init() */
#include "mock_common.h"
static const sasfit_common_stubs_t *stub_ptr = 0;
scalar sasfit_ff_mock_scaled(scalar q, sasfit_param *param) {
    if (param->p[0] < 0) { stub_ptr->set_err(param, "A(<0) not allowed"); return 0.0; }
    return stub_ptr->twice(param->p[0]) * exp(-q * param->p[1]);
}
scalar sasfit_ff_mock_scaled_f(scalar q, sasfit_param *param) { return sqrt(sasfit_ff_mock_scaled(q, param)); }
scalar sasfit_ff_mock_scaled_v(scalar x, sasfit_param *param, int d) { return d == 1 ? x : param->p[1]; }
/* box distribution on [p1, p2] with integral p0 */
scalar sasfit_sd_mock_box(scalar x, sasfit_param *param) {
    return (x >= param->p[1] && x <= param->p[2]) ? param->p[0] / (param->p[2] - param->p[1]) : 0.0;
}
static sasfit_plugin_info_t plugin_export = {2, {
    {14, "ff_mock_scaled", sasfit_ff_mock_scaled, sasfit_ff_mock_scaled_f, sasfit_ff_mock_scaled_v},
    {11, "sd_mock_box", sasfit_sd_mock_box, 0, 0},
}};
int get_ver(void) { return 1; }
int do_init(const sasfit_plugin_info_t **exp_ptr, const sasfit_common_stubs_t *ptr, sasfit_plugin_search_t *searchf) {
    static sasfit_plugin_info_t plugin_import = {0};
    if (!exp_ptr || !ptr || !searchf) return 0;
    *exp_ptr = &plugin_export;
    stub_ptr = ptr;
    return searchf(&plugin_import);
}
