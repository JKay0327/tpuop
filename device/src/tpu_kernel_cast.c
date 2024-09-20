#include "common.h"
#include "tpu_kernel.h"
#include "tpu_api_protocol.h"
#include "tpu_device_op.h"

extern void nodechip_cast(
    global_addr_t in_global_addr,
    global_addr_t out_global_addr,
    const int* shape,
    int shape_dim,
    data_type_t src_dtype,
    data_type_t dst_dtype,
    rounding_mode_t round_mode);

void tpu_kernel_api_cast(const void* api_buf) {
    tpu_kernel_api_cast_t *api = (tpu_kernel_api_cast_t*)api_buf;
    tpu_initialize();
    nodechip_cast(
        api->input_global_offset,
        api->output_global_offset,
        api->shape,
        api->dims,
        (data_type_t)api->src_dtype,
        (data_type_t)api->dst_dtype,
        (rounding_mode_t)api->round_mode);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_cast);