#ifndef __TPUOPS_HELP_H__
#define __TPUOPS_HELP_H__
#include <stdio.h>
#include <cstdlib>
#include "tpu_api_protocol.h"
#include "bmlib_runtime.h"
#include "tpu_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TPU_Kernel {
    int                 device_id;
    tpu_kernel_module_t tpu_module;
    bm_handle_t         bm_handle;
    tpu_kernel_function_t _multi_fullnet_func_id  ;
    tpu_kernel_function_t _dynamic_fullnet_func_id;
    tpu_kernel_function_t _enable_profile_func_id ;
    tpu_kernel_function_t _get_profile_func_id    ;
};

void                free_handle(bm_handle_t bm_handle);
bm_handle_t         get_handle(int device_id);
struct TPU_Kernel*  create_a_tpu_kernel(int device_id, const char* filename);
void                free_a_tpu_kernel(struct TPU_Kernel* tpu_kernel);
int                 search_func_id(struct TPU_Kernel* tpu_kernel, const char* func_name);


void free_handle(bm_handle_t bm_handle){
    bm_dev_free(bm_handle);
}

bm_handle_t get_handle(int device_id){
    bm_handle_t bm_handle;
    bm_dev_request(&bm_handle, device_id);
    return bm_handle;
}


struct TPU_Kernel* create_a_tpu_kernel(int device_id, const char* filename){
    struct TPU_Kernel* tpu_kernel = (struct TPU_Kernel*)malloc(sizeof(struct TPU_Kernel));
    tpu_kernel->device_id  = device_id;
    tpu_kernel->bm_handle  = get_handle(device_id);
    tpu_kernel->tpu_module = tpu_kernel_load_module_file(tpu_kernel->bm_handle, filename);
    tpu_kernel->_multi_fullnet_func_id   = search_func_id(tpu_kernel, "sg_api_multi_fullnet");
    tpu_kernel->_dynamic_fullnet_func_id = search_func_id(tpu_kernel, "sg_api_dynamic_fullnet");
    tpu_kernel->_enable_profile_func_id  = search_func_id(tpu_kernel, "sg_api_enable_profile");
    return tpu_kernel;
}

void free_a_tpu_kernel(struct TPU_Kernel* tpu_kernel){
    tpu_kernel_unload_module(tpu_kernel->bm_handle, tpu_kernel->tpu_module);
    free_handle(tpu_kernel->bm_handle);
    free(tpu_kernel);
}

int search_func_id(struct TPU_Kernel* tpu_kernel, const char* func_name){
    int func_id = tpu_kernel_get_function(tpu_kernel->bm_handle, tpu_kernel->tpu_module, func_name);
    printf("search_func_id func_name: %s, func_id: %d\n", func_name, func_id);
    return func_id;
}

void run_tpu_kernel_fun(struct TPU_Kernel* tpu_kernel, int func_id, void* args, size_t args_size){
    printf("run_tpu_kernel_fun fun_id: %d\n", func_id);
    bm_status_t status_ = tpu_kernel_launch(tpu_kernel->bm_handle, func_id, args, args_size);
    if(status_ != BM_SUCCESS){
        exit(-1);
    }
    bm_thread_sync(tpu_kernel->bm_handle);
}

#ifdef __cplusplus
}
#endif

#endif // __TPUOPS_HELP_H__