import os
import pathlib
import sys
import numpy as np
from functools import reduce
from typing import List

# CUDA stuff
# TODO: move from pvnrtc module to cuda entirely to reduce dependencies
from pynvrtc.compiler import Program
from cuda import cuda, nvrtc

from .arg_info import ArgInfo, verify_args
from .gpu_headers import CUDA_HEADER_MAP
from .hat_file import Function


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(cuda.cuGetErrorString(err)[1].decode('utf-8')))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def _find_cuda_incl_path() -> pathlib.Path:
    "Tries to find the CUDA include path."
    cuda_path = os.getenv("CUDA_PATH")
    if not cuda_path:
        if sys.platform == 'linux':
            cuda_path = pathlib.Path("/usr/local/cuda/include")
            if not (cuda_path.exists() and cuda_path.is_dir()):
                cuda_path = None
        elif sys.platform == 'win32':
            ...
        elif sys.platform == 'darwin':
            ...
    else:
        cuda_path /= "include"

    return cuda_path


def compile_cuda_program(cuda_src_path: pathlib.Path, func_name):
    src = cuda_src_path.read_text()

    prog = Program(src=src, name=func_name, headers=CUDA_HEADER_MAP.values(), include_names=CUDA_HEADER_MAP.keys())
    ptx = prog.compile([
        '-use_fast_math',
        '-default-device',
        '-std=c++11',
        '-arch=sm_52',    # TODO: is this needed?
    ])

    return ptx


def initialize_cuda():
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Retrieve handle for device 0
    # TODO: add support for multiple CUDA devices?
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)


def get_func_from_ptx(ptx, func_name):
    # Note: Incompatible --gpu-architecture would be detected here
    err, ptx_mod = cuda.cuModuleLoadData(ptx.encode('utf-8'))
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(ptx_mod, func_name.encode('utf-8'))
    ASSERT_DRV(err)

    return kernel


def _arg_size(arg_info: ArgInfo):
    return arg_info.element_num_bytes * reduce(lambda x, y: x * y, arg_info.numpy_shape)


def transfer_mem_host_to_cuda(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'input' in arg_info.usage.value:
            err, = cuda.cuMemcpyHtoD(device_arg, host_arg.ctypes.data, _arg_size(arg_info))
            ASSERT_DRV(err)


def transfer_mem_cuda_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'output' in arg_info.usage.value:
            err, = cuda.cuMemcpyDtoH(host_arg.ctypes.data, device_arg, _arg_size(arg_info))
            ASSERT_DRV(err)


def allocate_cuda_mem(arg_infos: List[ArgInfo]):
    device_mem = []
    for arg in arg_infos:
        err, mem = cuda.cuMemAlloc(_arg_size(arg))
        ASSERT_DRV(err)
        device_mem.append(mem)

    return device_mem


def device_args_to_ptr_list(device_args: List):
    # CUDA python example says this is subject to change
    ptrs = [np.array([int(d_arg)], dtype=np.uint64) for d_arg in device_args]
    ptrs = np.array([ptr.ctypes.data for ptr in ptrs], dtype=np.uint64)

    return ptrs


def create_loader_for_device_function(device_func: Function, hat_dir_path: str):
    if not device_func.provider:
        raise RuntimeError("Expected a provider for the device function")

    cuda_src_path: pathlib.Path = pathlib.Path(hat_dir_path) / device_func.provider
    func_name = device_func.name

    ptx = compile_cuda_program(cuda_src_path, func_name)

    initialize_cuda()

    kernel = get_func_from_ptx(ptx, func_name)

    hat_arg_descriptions = device_func.arguments
    arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
    launch_parameters = device_func.launch_parameters

    def f(*args):
        verify_args(args, arg_infos, func_name)
        device_mem = allocate_cuda_mem(arg_infos)
        transfer_mem_host_to_cuda(device_args=device_mem, host_args=args, arg_infos=arg_infos)
        ptrs = device_args_to_ptr_list(device_mem)

        err, stream = cuda.cuStreamCreate(0)
        ASSERT_DRV(err)

        err, = cuda.cuLaunchKernel(
            kernel,
            *launch_parameters,    # [ grid[x-z], block[x-z] ]
            0,    # dynamic shared memory
            stream,    # stream
            ptrs.ctypes.data,    # kernel arguments
            0,    # extra (ignore)
        )
        ASSERT_DRV(err)
        err, = cuda.cuStreamSynchronize(stream)
        ASSERT_DRV(err)

        transfer_mem_cuda_to_host(device_args=device_mem, host_args=args, arg_infos=arg_infos)

    return f
