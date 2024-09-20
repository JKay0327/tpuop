import ctypes
import numpy as np
import torch
import torch.nn as nn
import os
import platform
from typing import Any
import pdb
import copy

def print_ctype_floats(ptr, num):
    FloatArrayNum = ctypes.c_float * num
    float_array_ptr = ctypes.cast(ptr, ctypes.POINTER(FloatArrayNum))
    float_array = float_array_ptr.contents
    for i in range(num):
        print(f"Float {i}: {float_array[i]}")
    return

def make_np2c(np_array:np.ndarray):
    assert np_array.flags['C_CONTIGUOUS'], "np_array must be contiguous"
    return np_array.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

def make_torch2c(tensor:torch.Tensor):
    assert tensor.is_contiguous(), "tensor must be contiguous"
    ptr = tensor.data_ptr()
    return ctypes.c_void_p(ptr)

int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_void_p
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool
null_ptr  = ctypes.c_void_p(None)
ref       = lambda x: ctypes.byref(x)

def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)

def make2_c_int_list(my_list:list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point:ctypes.c_char_p):
    return ctypes.string_at(char_point).decode('utf-8')

def str2char_point(string:str):
    return ctypes.c_char_p(string.encode('utf-8'))

def make2_c_point_list(my_list:list):
    return (ctypes.c_void_p * len(my_list))(*my_list)
    
def build_c_torch_lists(args):
    # need be torch
    return make2_c_point_list( [ make_torch2c(i) for i in args] )

class Builder:

    def __init__(self, so_path:str="./lib/libtpuop.so"):
        self.so_path = os.path.join(os.path.dirname(__file__), so_path)
        self.lib = ctypes.CDLL(self.so_path)
        self.lib_init()

    def lib_init(self):
        # struct tensor* create_tensor( void* data, int dtype, int dims, int* shape, int device_id);
        # void tpuop_copy_tensor_into_host(struct tensor* cur_tensor);

        self.lib.create_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.create_tensor.restype = ctypes.c_void_p

        self.lib.tpuop_copy_tensor_into_host.argtypes = [ctypes.c_void_p]
        self.lib.tpuop_copy_tensor_into_host.restype = None

        # void tensor_free(struct tensor* tensor);
        self.lib.tensor_free.argtypes = [ctypes.c_void_p]
        self.lib.tensor_free.restype = None


        # struct TPU_Kernel* create_a_tpu_kernel(int device_id, const char* filename)
        self.lib.create_a_tpu_kernel.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.lib.create_a_tpu_kernel.restype = ctypes.c_void_p

        # void free_a_tpu_kernel(struct TPU_Kernel* tpu_kernel)
        self.lib.free_a_tpu_kernel.argtypes = [ctypes.c_void_p]
        self.lib.free_a_tpu_kernel.restype = None

        # void tpuop_cast( struct TPU_Kernel* tpu_kernel, 
        #                 struct tensor*    input_tensor, 
        #                 struct tensor*    output_tensor, 
        #                 int round_mode);
        self.lib.tpuop_cast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.lib.tpuop_cast.restype = None

        # void tpuop_layer_norm( struct TPU_Kernel* tpu_kernel, 
        #                         struct tensor*    input_tensor, 
        #                         struct tensor*    weight_tensor, 
        #                         struct tensor*    bias_tensor, 
        #                         struct tensor*    output_tensor, 
        #                         struct tensor*    mean_tensor, 
        #                         struct tensor*    rstd_tensor, 
        #                         int axis, 
        #                         float eps, 
        #                         int affine, 
        #                         int need_mean, 
        #                         int need_rstd);
        self.lib.tpuop_layer_norm.argtypes = [ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_void_p, 
                                              ctypes.c_int, 
                                              ctypes.c_float, 
                                              ctypes.c_int, 
                                              ctypes.c_int, 
                                              ctypes.c_int]
        
        # void tpuop_transpose( struct TPU_Kernel* tpu_kernel, 
        #                 struct tensor*    input_tensor, 
        #                 struct tensor*    output_tensor, 
        #                 int* order);
        self.lib.tpuop_transpose.argtypes = [ctypes.c_void_p, 
                                             ctypes.c_void_p, 
                                             ctypes.c_void_p, 
                                             ctypes.c_void_p, 
                                             ctypes.POINTER(ctypes.c_int)]
    

class TpuOp:

    def __init__(self, model_so_path:str="./lib/libcmodel.so", device:int = 0 ):
        self.builder = Builder()
        self.device = device
        self.model_so_path = os.path.join(os.path.dirname(__file__), model_so_path)
        self.tpu_kernel = self.builder.lib.create_a_tpu_kernel(self.device, str2char_point(self.model_so_path))

        self.torch_dtype_map = {
            torch.float32: 1,
            torch.float16: 2,
            torch.bfloat16: 3,
            torch.int32: 4,
            torch.int8: 5,
            torch.uint8: 6
        }

        self.np_dtype_map = {
            np.float32: 1,
            np.float16: 2,
            np.int32: 4,
            np.int8: 5,
            np.uint8: 6
        }

    def tpuop_make_torch2optensor(self, tensor:torch.Tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        data_ptr = make_torch2c(tensor)
        # print_ctype_floats(data_ptr, 16)
        dtype = self.torch_dtype_map[tensor.dtype]
        shapes = list(tensor.shape)
        dims = len(shapes)
        shape = make2_c_int_list(shapes)
        # print_ctype_floats(data_ptr, 16)
        # print(tensor)
        
        return self.builder.lib.create_tensor(data_ptr, dtype, dims, shape, self.device)

    def tpuop_make_np2optensor(self, np_array:np.ndarray):
        if np_array.flags['CONTIGUOUS'] == False:
            # info users
            np_array = np.ascontiguousarray(np_array)
        data_ptr = make_np2c(np_array)
        dtype = self.np_dtype_map[np_array.dtype]
        shapes = [i for i in np_array.shape]
        dims = len(shapes)
        shape = make2_c_int_list(shapes)
        return self.builder.lib.create_tensor(data_ptr, dtype, dims, shape, self.device)

    def tpuop_cast(self, input_tensor, output_tensor, round_mode):
        input_optensor = self.tpuop_make_torch2optensor(input_tensor)
        output_optensor = self.tpuop_make_torch2optensor(output_tensor)
        return self.builder.lib.tpuop_cast(self.tpu_kernel, 
                                    input_optensor, 
                                    output_optensor, 
                                    round_mode)

    def tpuop_layernorm(self, 
                        input_tensor, 
                        weight_tensor, 
                        bias_tensor, 
                        output_tensor, 
                        axis, 
                        eps):
        assert weight_tensor is not None
        assert bias_tensor is not None
        input_optensor = self.tpuop_make_torch2optensor(input_tensor)
        weight_optensor = self.tpuop_make_torch2optensor(weight_tensor)
        bias_optensor = self.tpuop_make_torch2optensor(bias_tensor)
        output_optensor = self.tpuop_make_torch2optensor(output_tensor)
        affine = (1<<0) + (1<<1)
        need_mean = 0
        need_rstd = 0
        mean_tensor = torch.zeros_like(input_tensor)
        rstd_tensor = torch.zeros_like(input_tensor)
        mean_optensor = self.tpuop_make_torch2optensor(mean_tensor)
        rstd_optensor = self.tpuop_make_torch2optensor(rstd_tensor)
        return self.builder.lib.tpuop_layer_norm(self.tpu_kernel,
                                            input_optensor, 
                                            weight_optensor, 
                                            bias_optensor, 
                                            output_optensor, 
                                            mean_optensor, 
                                            rstd_optensor, 
                                            axis, 
                                            eps, 
                                            affine, 
                                            need_mean, 
                                            need_rstd)
    
    def tpuop_transpose(self, input_tensor, output_tensor, buffer_tensor, order):
        input_optensor = self.tpuop_make_torch2optensor(input_tensor)
        output_optensor = self.tpuop_make_torch2optensor(output_tensor)
        buffer_optensor = self.tpuop_make_torch2optensor(buffer_tensor)
        order = make2_c_int_list(order)
        return self.builder.lib.tpuop_transpose(
            self.tpu_kernel,
            input_optensor,
            output_optensor,
            buffer_optensor,
            order)
