
#include "stdio.h"
#include "assert.h"
#include "bmdef.h"
#include "bmlib_runtime.h"
#include "bmruntime_interface.h"
#include "tpu_api_protocol.h"
#include "help.h"

#define DEBUG_PRINT 1

#ifdef __cplusplus
extern "C" {
#endif

struct tensor {
    void* data;
    int dt_dtype;
    int source_dtype;
    int dims;
    int shape[MAX_SHAPE_DIMS];
    int device_id;
    bm_device_mem_t bm_mem;
};

u64 get_tensor_device_address(struct tensor* tensor);
struct tensor* create_tensor( void* data, int dtype, int dims, int* shape, int device_id);
inline void tpuop_copy_tensor_into_device(struct tensor* cur_tensor);
void tpuop_copy_tensor_into_host(struct tensor* cur_tensor);
void tensor_free(struct tensor* tensor);
void tpuop_cast( struct TPU_Kernel* tpu_kernel, 
                 struct tensor*    input_tensor, 
                 struct tensor*    output_tensor, 
                 int round_mode);
void tpuop_layer_norm( struct TPU_Kernel* tpu_kernel, 
                        struct tensor*    input_tensor, 
                        struct tensor*    weight_tensor, 
                        struct tensor*    bias_tensor, 
                        struct tensor*    output_tensor, 
                        struct tensor*    mean_tensor, 
                        struct tensor*    rstd_tensor, 
                        int axis, 
                        float eps, 
                        int affine, 
                        int need_mean, 
                        int need_rstd);

inline u64 get_tensor_device_address(struct tensor* tensor){
    return tensor->bm_mem.u.device.device_addr;
}
// 1 fp32 2 fp16 3 bf16 4 int32 5 int8 6 uint8

inline size_t tpuop_get_tensor_sizes(struct tensor* cur_tensor){
    size_t size = 1;
    for(int i=0;i<cur_tensor->dims;i++){ size *= cur_tensor->shape[i]; }
    switch(cur_tensor->source_dtype){
        case 1: size *= sizeof(float); break;
        case 2: size *= sizeof(short); break;
        case 3: size *= sizeof(short); break;
        case 4: size *= sizeof(int); break;
        case 5: size *= sizeof(char); break;
        case 6: size *= sizeof(unsigned char); break;
        default: assert(-1); break;
    }
    return size;
}

inline void tpuop_copy_tensor_into_device(struct tensor* cur_tensor){
    bm_handle_t handle = get_handle(cur_tensor->device_id);
    bm_device_mem_t bm_mem;
    bm_malloc_device_byte(handle, &bm_mem, tpuop_get_tensor_sizes(cur_tensor));
    bm_memcpy_s2d(handle, bm_mem, cur_tensor->data);
    cur_tensor->bm_mem = bm_mem;
    if (DEBUG_PRINT) {
        printf("copy tensor into device: %p\n", cur_tensor->data);
    }
}

void tpuop_copy_tensor_into_host(struct tensor* cur_tensor){
    bm_handle_t handle = get_handle(cur_tensor->device_id);
    bm_memcpy_d2s(handle, cur_tensor->data, cur_tensor->bm_mem);
    if (DEBUG_PRINT) {
        printf("copy tensor into host: %p\n", cur_tensor->data);
    }
}

struct tensor* create_tensor( void* data, int dtype, int dims, int* shape, int device_id){
    // printf(">>>>>>>>> data prt: %p\n", data);
    struct tensor* tensor = (struct tensor*) calloc(1, sizeof(struct tensor));
    bm_handle_t handle = get_handle(device_id);
    tensor->dims = dims;
    for(int i=0;i<dims;i++){ tensor->shape[i] = shape[i]; }
    tensor->device_id = device_id;
    tensor->data = data;
    tensor->source_dtype = dtype;
    switch(dtype){
        case 1: tensor->dt_dtype = DT_FP32; break;
        case 2: tensor->dt_dtype = DT_FP16; break;
        case 3: tensor->dt_dtype = DT_BFP16; break;
        case 4: tensor->dt_dtype = DT_INT32; break;
        case 5: tensor->dt_dtype = DT_INT8; break;
        case 6: tensor->dt_dtype = DT_UINT8; break;
        default: assert(-1); break;
    }
    tpuop_copy_tensor_into_device(tensor);
    return tensor;
}

void tensor_free(struct tensor* tensor){
    bm_handle_t handle = get_handle(tensor->device_id);
    bm_free_device(handle, tensor->bm_mem);
    free(tensor);
}

void tpuop_cast( struct TPU_Kernel* tpu_kernel, 
                 struct tensor*    input_tensor, 
                 struct tensor*    output_tensor, 
                 int round_mode){
    tpu_kernel_api_cast_t cast_param;
    cast_param.input_global_offset  = get_tensor_device_address(input_tensor);
    cast_param.output_global_offset = get_tensor_device_address(output_tensor);
    cast_param.dims = input_tensor->dims;
    for(int i=0;i<input_tensor->dims;i++){ cast_param.shape[i] = input_tensor->shape[i]; }
    cast_param.src_dtype  = input_tensor->dt_dtype;
    cast_param.dst_dtype  = output_tensor->dt_dtype;
    cast_param.round_mode = round_mode;
    int func_id = search_func_id(tpu_kernel, "tpu_kernel_api_cast");
    run_tpu_kernel_fun(tpu_kernel, func_id, &cast_param, sizeof(cast_param));
    // copy to cpu
    tpuop_copy_tensor_into_host(output_tensor);
}

void tpuop_layer_norm( struct TPU_Kernel* tpu_kernel, 
                        struct tensor*    input_tensor, 
                        struct tensor*    weight_tensor, 
                        struct tensor*    bias_tensor, 
                        struct tensor*    output_tensor, 
                        struct tensor*    mean_tensor, 
                        struct tensor*    rstd_tensor, 
                        int axis, 
                        float eps, 
                        int affine, 
                        int need_mean, 
                        int need_rstd){
    tpu_kernel_api_layer_norm_t layer_norm_param;
    layer_norm_param.input_global_offset  = get_tensor_device_address(input_tensor);
    layer_norm_param.weight_global_offset = get_tensor_device_address(weight_tensor);
    layer_norm_param.bias_global_offset   = get_tensor_device_address(bias_tensor);
    layer_norm_param.output_global_offset = get_tensor_device_address(output_tensor);
    layer_norm_param.mean_global_offset   = get_tensor_device_address(mean_tensor);
    layer_norm_param.rstd_global_offset   = get_tensor_device_address(rstd_tensor);
    layer_norm_param.dims = input_tensor->dims;
    for(int i=0;i<input_tensor->dims;i++){ layer_norm_param.shape[i] = input_tensor->shape[i]; }
    layer_norm_param.axis = axis;
    layer_norm_param.eps = eps;
    layer_norm_param.affine = affine;
    layer_norm_param.need_mean = need_mean;
    layer_norm_param.need_rstd = need_rstd;
    layer_norm_param.dtype = output_tensor->dt_dtype;
    int func_id = search_func_id(tpu_kernel, "tpu_kernel_layer_norm");
    run_tpu_kernel_fun(tpu_kernel, func_id, &layer_norm_param, sizeof(layer_norm_param));
    tpuop_copy_tensor_into_host(output_tensor);
}

void tpuop_transpose( struct TPU_Kernel* tpu_kernel, 
                        struct tensor*    input_tensor, 
                        struct tensor*    output_tensor, 
                        struct tensor*    buffer_tensor, 
                        int* order){
    tpu_kernel_api_transpose_t transpose_param;
    transpose_param.input_global_addr  = get_tensor_device_address(input_tensor);
    transpose_param.output_global_addr = get_tensor_device_address(output_tensor);
    transpose_param.buffer_global_addr = get_tensor_device_address(buffer_tensor);
    uint64_t buffer_size = 0;
    transpose_param.buffer_size_ptr = &buffer_size;
    // assert(input_tensor->dims == 3 || input_tensor->dims == 4);
    // transpose_param.dims = 4;
    // int insert_dim = 4 - input_tensor->dims;
    // for (int i = 0; i < 4; i++) {
    //     if (i < insert_dim) {
    //         transpose_param.input_shape[i] = 1;
    //         transpose_param.order[i] = i;
    //     } else {
    //         transpose_param.input_shape[i] = input_tensor->shape[i - insert_dim];
    //         transpose_param.order[i] = order[i - insert_dim] + insert_dim;
    //     }
    // }
    for (int i = 0; i < input_tensor->dims; i++) {
        transpose_param.input_shape[i] = input_tensor->shape[i];
        transpose_param.order[i] = order[i];
    }
    transpose_param.dims = input_tensor->dims;
    transpose_param.dtype = output_tensor->dt_dtype;
    int func_id = search_func_id(tpu_kernel, "tpu_kernel_api_transpose");
    run_tpu_kernel_fun(tpu_kernel, func_id, &transpose_param, sizeof(transpose_param));
    tpuop_copy_tensor_into_host(output_tensor);
}

#ifdef __cplusplus
}
#endif
