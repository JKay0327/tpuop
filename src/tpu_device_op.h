#include "common.h"
#include "tpu_kernel.h"
#include "tpu_api_protocol.h"

typedef enum {
  BINARY_ADD          = 0,
  BINARY_SUB          = 1,
  BINARY_MUL          = 2,
  BINARY_DIV          = 3,
  BINARY_MAX          = 4,
  BINARY_MIN          = 10000,
  BINARY_GT           = 10001,
  BINARY_GE           = 10002,
  BINARY_LT           = 10003,
  BINARY_LE           = 10004,
  BINARY_EQ           = 10005,
  BINARY_NE           = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD    = 10008,
  BINARY_FLOOR_DIV    = 10009
} sg_binary_type_t;

typedef void(*binary_int_func)(
    local_addr_t, local_addr_t, local_addr_t,
    const dim4*, const dim4*, const dim4*, const dim4*,
    data_type_t, data_type_t, data_type_t,
    char, rounding_mode_t, bool);

typedef void(*binary_cmp_func)(
    local_addr_t, local_addr_t, local_addr_t, scalar_t,
    const dim4*, const dim4*, const dim4*, const dim4*,
    data_type_t, data_type_t);

typedef void(*binary_limit_func)(
    local_addr_t, local_addr_t, local_addr_t,
    const dim4*, const dim4*, const dim4*, const dim4*,
    data_type_t);

typedef void(*const_binary_int_func)(
    local_addr_t, local_addr_t, scalar_t,
    const dim4*, const dim4*, const dim4*,
    data_type_t, data_type_t, data_type_t,
    char, rounding_mode_t, bool);

typedef void(*const_binary_cmp_func)(
    local_addr_t, local_addr_t, scalar_t, scalar_t,
    const dim4*, const dim4*, const dim4*,
    data_type_t, data_type_t);

typedef void(*const_binary_limit_func)(
    local_addr_t, local_addr_t, scalar_t,
    const dim4*, const dim4*, const dim4*,
    data_type_t);

typedef void(*binary_fp_func)(
    local_addr_t, local_addr_t, local_addr_t,
    const dim4*, const dim4*, const dim4*,
    const dim4*, data_type_t);

typedef void(*const_binary_fp_func)(
    local_addr_t, local_addr_t, scalar_t,
    const dim4*, const dim4*, const dim4*,
    data_type_t);

typedef void(*binary_fp32_func)(
    local_addr_t, local_addr_t, local_addr_t,
    const dim4*, const dim4*, const dim4*,
    const dim4*);

typedef void(*const_binary_fp32_func)(
    local_addr_t, local_addr_t, float,
    const dim4*, const dim4*, const dim4*);

static inline binary_int_func get_binary_int_func(int binary_type) {
  binary_int_func func = NULL;
  if (binary_type == BINARY_ADD) func = tpu_bdc_int_add;
  else if (binary_type == BINARY_SUB) func = tpu_bdc_int_sub;
  else if (binary_type == BINARY_MUL) func = tpu_bdc_int_mul;
  return func;
}

static inline binary_cmp_func get_binary_cmp_func(int binary_type) {
  binary_cmp_func func = NULL;
  if (binary_type == BINARY_EQ) func = tpu_bdc_equal;
  else if (binary_type == BINARY_NE) func = tpu_bdc_not_equal;
  else if (binary_type == BINARY_LE) func = tpu_bdc_less_equal;
  else if (binary_type == BINARY_LT) func = tpu_bdc_less;
  else if (binary_type == BINARY_GE) func = tpu_bdc_greater_equal;
  else if (binary_type == BINARY_GT) func = tpu_bdc_greater;
  return func;
}

static inline binary_limit_func get_binary_limit_func(int binary_type) {
  binary_limit_func func = NULL;
  if (binary_type == BINARY_MIN) func = tpu_bdc_min;
  else if (binary_type == BINARY_MAX) func = tpu_bdc_max;
  return func;
}

static inline const_binary_int_func get_const_binary_int_func(int binary_type, int inversed) {
  const_binary_int_func func = NULL;
  if (binary_type == BINARY_ADD) func = tpu_bdc_int_add_C;
  else if (binary_type == BINARY_MUL) func = tpu_bdc_int_mul_C;
  else if (binary_type == BINARY_SUB && !inversed) func = tpu_bdc_int_sub_C;
  else if (binary_type == BINARY_SUB && inversed) func = tpu_bdc_int_C_sub;
  return func;
}

static inline const_binary_cmp_func get_const_binary_cmp_func(int binary_type) {
  const_binary_cmp_func func = NULL;
  if (binary_type == BINARY_EQ) func = tpu_bdc_equal_C;
  else if (binary_type == BINARY_NE) func = tpu_bdc_not_equal_C;
  else if (binary_type == BINARY_LE) func = tpu_bdc_less_equal_C;
  else if (binary_type == BINARY_LT) func = tpu_bdc_less_C;
  else if (binary_type == BINARY_GE) func = tpu_bdc_greater_equal_C;
  else if (binary_type == BINARY_GT) func = tpu_bdc_greater_C;
  return func;
}

static inline const_binary_limit_func get_const_binary_limit_func(int binary_type) {
  const_binary_limit_func func = NULL;
  if (binary_type == BINARY_MIN) func = tpu_bdc_min_C;
  else if (binary_type == BINARY_MAX) func = tpu_bdc_max_C;
  return func;
}

static inline binary_fp_func get_binary_fp_func(int binary_type) {
  binary_fp_func func = NULL;
  if (binary_type == BINARY_ADD) func = tpu_bdc_fp_add;
  else if (binary_type == BINARY_SUB) func = tpu_bdc_fp_sub;
  else if (binary_type == BINARY_MUL) func = tpu_bdc_fp_mul;
  else if (binary_type == BINARY_DIV) func = tpu_bdc_fp_div;
  return func;
}

static inline const_binary_fp_func get_const_binary_fp_func(int binary_type, int inversed) {
  const_binary_fp_func func = NULL;
  if (binary_type == BINARY_ADD) func = tpu_bdc_fp_add_C;
  else if (binary_type == BINARY_MUL) func = tpu_bdc_fp_mul_C;
  else if (binary_type == BINARY_SUB && !inversed) func = tpu_bdc_fp_sub_C;
  else if (binary_type == BINARY_SUB && inversed) func = tpu_bdc_fp_C_sub;
  return func;
}

static inline const_binary_fp_func get_const_binary_div_func(int inversed) {
  const_binary_fp_func func = NULL;
  if (!inversed) func = tpu_bdc_fp_div_C;
  else if (inversed) func = tpu_bdc_fp_C_div;
  return func;
}

static inline bool is_cmp_op(int binary_type) {
  return (binary_type == BINARY_EQ || binary_type == BINARY_NE ||
          binary_type == BINARY_LE || binary_type == BINARY_LT ||
          binary_type == BINARY_GE || binary_type == BINARY_GT);
}