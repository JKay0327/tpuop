#include "common.h"
#include "tpu_kernel.h"
#include "tpu_api_protocol.h"
#include "tpu_device_op.h"


extern void nodechip_bcbinary_fp(
    global_addr_t A_global_addr,
    global_addr_t B_global_addr,
    global_addr_t res_global_addr,
    const int* A_shape,
    const int* B_shape,
    int A_dim,
    int B_dim,
    int binary_type,
    data_type_t dtype,
    int if_relu,
    float relu_upper_limit);

void tpu_kernel_api_bcbinary(const void* api_buf) {
    tpu_kernel_api_bcbinary_t *api = (tpu_kernel_api_bcbinary_t*)api_buf;
    tpu_initialize();
    nodechip_bcbinary_fp(
        api->a_global_addr,
        api->b_global_addr,
        api->c_global_addr,
        api->a_shape,
        api->b_shape,
        api->a_dim,
        api->b_dim,
        api->binary_type,
        (data_type_t)api->dtype,
        api->if_relu,
        api->relu_upper_limit);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_bcbinary);