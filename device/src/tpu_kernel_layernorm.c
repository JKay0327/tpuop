#include "common.h"
#include "tpu_kernel.h"
#include "tpu_api_protocol.h"
#include "tpu_device_op.h"

extern void nodechip_layer_norm(
    global_addr_t input_global_addr,
    global_addr_t weight_global_addr,
    global_addr_t bias_global_addr,
    global_addr_t output_global_addr,
    global_addr_t mean_global_addr,
    global_addr_t rstd_global_addr,
    const int *shape,
    int dims,
    int axis,
    float eps,
    int affine,
    bool need_mean,
    bool need_rstd,
    data_type_t dtype);

void tpu_kernel_layer_norm(const void *args) {
    tpu_kernel_api_layer_norm_t *api = (tpu_kernel_api_layer_norm_t *)args;
    tpu_initialize();
    nodechip_layer_norm(
        api->input_global_offset,
        api->weight_global_offset,
        api->bias_global_offset,
        api->output_global_offset,
        api->mean_global_offset,
        api->rstd_global_offset,
        api->shape,
        api->dims,
        api->axis,
        api->eps,
        api->affine,
        api->need_mean,
        api->need_rstd,
        (data_type_t)api->dtype);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_layer_norm);