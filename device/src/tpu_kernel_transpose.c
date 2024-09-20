#include "tpu_kernel.h"
#include "tpu_api_protocol.h"

extern void nodechip_transpose(
    global_addr_t                input_global_addr,
    global_addr_t                output_global_addr,
    const int*                   input_shape,
    const int*                   order,
    int                          dims,
    global_addr_t                buffer_global_addr,
    unsigned long long*          buffer_size, //if not NULL, just calculate buffer_size, not compute
    data_type_t                  dtype
);

void tpu_kernel_api_transpose(const void* args) {
  tpu_kernel_api_transpose_t* api = (tpu_kernel_api_transpose_t*)args;
  int input_shape_fix[MAX_SHAPE_DIMS];
  int order_fix[MAX_SHAPE_DIMS];
  TPUKERNEL_ASSERT(api->dims <= 4);
  int insert_dim = 4 - api->dims;
  for (int i = 0; i < 4; i++) {
    if (i < insert_dim) {
      input_shape_fix[i] = 1;
      order_fix[i] = i;
    } else {
      input_shape_fix[i] = api->input_shape[i - insert_dim];
      order_fix[i] = api->order[i - insert_dim] + insert_dim;
    }
  }
  tpu_initialize();
  // printf("tpu_kernel_api_transpose start");
  nodechip_transpose(
    api->input_global_addr, 
    api->output_global_addr, 
    input_shape_fix, 
    order_fix, 
    4, 
    api->buffer_global_addr, 
    0,  
    api->dtype);
  // dim4 shape = {api->input_shape[0], api->input_shape[1], api->input_shape[2], api->input_shape[3]};
  // tpu_gdma_cpy_S2S(api->output_global_addr, api->input_global_addr, &shape, NULL, NULL, DT_FP32);
  // printf("tpu_kernel_api_transpose end");
  tpu_poll();
}

TPUKERNEL_FUNC_REGISTER(tpu_kernel_api_transpose);
