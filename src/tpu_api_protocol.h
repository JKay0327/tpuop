#ifndef __TPU_API_PROTOCOL_H__
#define __TPU_API_PROTOCOL_H__

#ifdef __cplusplus
extern "C" {
#endif

// type define's
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned short u16;

#define sg_min(x, y) (((x)) < ((y)) ? (x) : (y))
#define sg_max(x, y) (((x)) > ((y)) ? (x) : (y))
#define UNUSED(x) (void)(x)
#define NO_USE 0

#define MAX_ATTR_NUM 16
#define MAX_ID_ATTR_NUM 512
#define MAX_MULTI_CROP_NUM 10
#define MAX_SHAPE_DIMS 8

#define TPU_KERNEL_MAX_IMAGE_DIM 3
#define TPU_KERNEL_MAX_IMAGE_CHANNELS 3
#define MAX_bm_image_CHANNEL 4
#define MAX_PROPOSAL_LAYER_OUTPUT_ROI_NUM 1000


typedef struct {
    u64 a_global_addr;
    u64 b_global_addr;
    u64 c_global_addr;
    int a_shape[MAX_SHAPE_DIMS];
    int b_shape[MAX_SHAPE_DIMS];
    int a_dim;
    int b_dim;
    int binary_type;
    int dtype;
    int if_relu;
    float relu_upper_limit;
} __attribute__((packed)) tpu_kernel_api_bcbinary_t;

typedef struct {
    u64 input_global_offset;
    u64 output_global_offset;
    int shape[MAX_SHAPE_DIMS];
    int dims;
    float const_value;
    int inversed;
    int binary_type;
    int dtype;
    int if_relu;
    float relu_upper_limit;
} __attribute__((packed)) tpu_kernel_api_const_binary_t;

typedef struct {
    u64 input_global_offset;
    u64 output_global_offset;
    int shape[MAX_SHAPE_DIMS];
    int dims;
    int src_dtype;
    int dst_dtype;
    int round_mode;
} __attribute__((packed)) tpu_kernel_api_cast_t;

typedef struct {
    u64 l_global_offset;
    u64 r_global_offset;
    u64 bias_global_offset;
    u64 output_global_offset;
    int l_shape[MAX_SHAPE_DIMS];
    int r_shape[MAX_SHAPE_DIMS];
    int output_shape[MAX_SHAPE_DIMS];
    int l_dim;
    int r_dim;
    int output_dim;
    int in_dtype;
    int out_dtype;
    int l_trans;
    int r_trans;
    int hdim_is_batch;
    int has_bias;
    int if_relu;
    float relu_upper_limit;
} __attribute__((packed)) tpu_kernel_api_batch_matmul_t;

typedef struct {
    u64 input_global_offset;
    u64 weight_global_offset;
    u64 bias_global_offset;
    u64 output_global_offset;
    u64 mean_global_offset;
    u64 rstd_global_offset;
    int shape[MAX_SHAPE_DIMS];
    int dims;
    int axis;
    float eps;
    int affine;
    int need_mean;
    int need_rstd;
    int dtype;
} __attribute__((packed)) tpu_kernel_api_layer_norm_t;

typedef struct {
    u64 input_global_addr;
    u64 output_global_addr;
    int input_shape[MAX_SHAPE_DIMS];
    int order[MAX_SHAPE_DIMS];
    int dims;
    u64 buffer_global_addr;
    uint64_t *buffer_size_ptr;
    int dtype;
} __attribute__((packed)) tpu_kernel_api_transpose_t;


#ifdef __cplusplus
}
#endif

#endif
