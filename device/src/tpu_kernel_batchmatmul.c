#include "common.h"
#include "tpu_kernel.h"
#include "tpu_api_protocol.h"
#include "tpu_device_op.h"


extern void nodechip_batch_matmul_float(
  global_addr_t L_global_addr,
  global_addr_t R_global_addr,
  global_addr_t bias_global_addr,
  global_addr_t Y_global_addr,
  data_type_t in_dtype,
  data_type_t out_dtype,
  const int* L_shape,
  const int* R_shape,
  int L_dim,
  int R_dim,
  int* Y_shape,
  int* Y_dim,
  int L_trans,
  int R_trans,
  int hdim_is_batch,
  int has_bias,
  bool do_relu,
  float relu_upper_limit);

void tpu_kernel_batch_matmul(const void* args){
    tpu_kernel_api_batch_matmul_t *api = (tpu_kernel_api_batch_matmul_t*)args;
    tpu_initialize();
    nodechip_batch_matmul_float(
        api->l_global_offset,
        api->r_global_offset,
        api->bias_global_offset,
        api->output_global_offset,
        (data_type_t)api->in_dtype,
        (data_type_t)api->out_dtype,
        api->l_shape,
        api->r_shape,
        api->l_dim,
        api->r_dim,
        NULL,
        NULL,
        api->l_trans,
        api->r_trans,
        api->hdim_is_batch,
        api->has_bias,
        (bool)api->if_relu,
        api->relu_upper_limit);
    tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_batch_matmul);
