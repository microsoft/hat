import os
import pathlib
import sys
import numpy as np
from typing import List

# CUDA stuff
# TODO: move from pvnrtc module to cuda entirely to reduce dependencies
from pynvrtc.compiler import Program
from cuda import cuda, nvrtc

from .arg_info import ArgInfo, verify_args
from .callable_func import CallableFunc
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
        '-std=c++11'
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


def _cuda_transfer_mem(usage, func, source_args: List, dest_args: List, arg_infos: List[ArgInfo], stream=None):
    for source_arg, dest_arg, arg_info in zip(source_args, dest_args, arg_infos):
        if usage in arg_info.usage.value:
            if stream:
                err, = func(dest_arg, source_arg, arg_info.total_byte_size, stream)
            else:
                err, = func(dest_arg, source_arg, arg_info.total_byte_size)
            ASSERT_DRV(err)


def transfer_mem_host_to_cuda(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo], stream=None):
    _cuda_transfer_mem(
        usage='input',
        func=cuda.cuMemCpyHtoDAsync if stream else cuda.cuMemcpyHtoD,
        source_args=[a.ctypes.data for a in host_args],
        dest_args=device_args,
        arg_infos=arg_infos,
        stream=stream
    )


def transfer_mem_cuda_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo], stream=None):
    _cuda_transfer_mem(
        usage='output',
        func=cuda.cuMemcpyDtoHAsync if stream else cuda.cuMemcpyDtoH,
        source_args=device_args,
        dest_args=[a.ctypes.data for a in host_args],
        arg_infos=arg_infos,
        stream=stream
    )


def allocate_cuda_mem(arg_infos: List[ArgInfo], stream=None):
    device_mem = []

    for arg in arg_infos:
        size = arg.total_byte_size
        err, mem = cuda.cuMemAllocAsync(size, stream) if stream else cuda.cuMemAlloc(size)
        try:
            ASSERT_DRV(err)
        except:
            free_cuda_mem(device_mem, stream)
            raise
        device_mem.append(mem)

    return device_mem


def free_cuda_mem(args, stream=None):
    for arg in args:
        cuda.cuMemFreeAsync(arg, stream) if stream else cuda.cuMemFree(arg)


def device_args_to_ptr_list(device_args: List):
    # CUDA python example says this is subject to change
    ptrs = [np.array([int(d_arg)], dtype=np.uint64) for d_arg in device_args]
    ptrs = np.array([ptr.ctypes.data for ptr in ptrs], dtype=np.uint64)

    return ptrs


_PTX_CACHE = {}


class CudaCallableFunc(CallableFunc):

    def __init__(self, func: Function, cuda_src_path: str) -> None:
        super().__init__()
        self.hat_func = func
        self.func_name = func.name
        self.kernel = None
        hat_arg_descriptions = func.arguments
        self.arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
        self.launch_params = func.launch_parameters
        self.device_mem = None
        self.ptrs = None
        self.start_event = None
        self.stop_event = None
        self.exec_time = 0.
        self.cuda_src_path = cuda_src_path

    def init_runtime(self):
        initialize_cuda()

        ptx = _PTX_CACHE.get(self.cuda_src_path)
        if not ptx:
            _PTX_CACHE[self.cuda_src_path] = ptx = compile_cuda_program(self.cuda_src_path, self.func_name)

        self.kernel = get_func_from_ptx(ptx, self.func_name)

    def cleanup_runtime(self):
        pass

    def init_main(self, warmup_iters=0, args=[]):
        verify_args(args, self.arg_infos, self.func_name)
        self.device_mem = allocate_cuda_mem(self.arg_infos)
        transfer_mem_host_to_cuda(device_args=self.device_mem, host_args=args, arg_infos=self.arg_infos)
        self.ptrs = device_args_to_ptr_list(self.device_mem)

        err, self.start_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)
        ASSERT_DRV(err)
        err, self.stop_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)
        ASSERT_DRV(err)

        for _ in range(warmup_iters):
            err, = cuda.cuLaunchKernel(
                self.kernel,
                *self.launch_params,    # [ grid[x-z], block[x-z] ]
                0,    # dynamic shared memory
                0,    # stream
                self.ptrs.ctypes.data,    # kernel arguments
                0,    # extra (ignore)
            )
            ASSERT_DRV(err)
        else:
            err, = cuda.cuCtxSynchronize()
            ASSERT_DRV(err)

    def main(self, iters=1, batch_size=1, args=[]) -> float:
        batch_timings: List[float] = []
        for _ in range(batch_size):
            err, = cuda.cuEventRecord(self.start_event, 0)
            ASSERT_DRV(err)

            for _ in range(iters):
                err, = cuda.cuLaunchKernel(
                    self.kernel,
                    *self.launch_params,    # [ grid[x-z], block[x-z] ]
                    0,    # dynamic shared memory
                    0,    # stream
                    self.ptrs.ctypes.data,    # kernel arguments
                    0,    # extra (ignore)
                )
                ASSERT_DRV(err)

            err, = cuda.cuEventRecord(self.stop_event, 0)
            ASSERT_DRV(err)
            err, = cuda.cuEventSynchronize(self.stop_event)
            ASSERT_DRV(err)
            err, batch_time = cuda.cuEventElapsedTime(self.start_event, self.stop_event)
            ASSERT_DRV(err)
            batch_timings.append(batch_time)
            self.exec_time += batch_time

            err, = cuda.cuCtxSynchronize()
            ASSERT_DRV(err)

        self.exec_time /= (iters * batch_size)
        return batch_timings

    def cleanup_main(self, args=[]):
        # If there's no device mem, that means allocation during initialization failed, which means nothing else needs to be cleaned up either
        if self.device_mem:
            transfer_mem_cuda_to_host(device_args=self.device_mem, host_args=args, arg_infos=self.arg_infos)
            free_cuda_mem(self.device_mem)

        err, = cuda.cuCtxSynchronize()
        ASSERT_DRV(err)

        if self.start_event:
            cuda.cuEventDestroy(self.start_event)

        if self.stop_event:
            cuda.cuEventDestroy(self.stop_event)


def create_loader_for_device_function(device_func: Function, hat_dir_path: str) -> CallableFunc:
    if not device_func.provider:
        raise RuntimeError("Expected a provider for the device function")

    cuda_src_path: pathlib.Path = pathlib.Path(hat_dir_path) / device_func.provider

    return CudaCallableFunc(device_func, cuda_src_path)
