import os
import pathlib
import sys
import numpy as np
from typing import List
from cuda import cuda, nvrtc
from .arg_info import ArgInfo
from .callable_func import CallableFunc
from .function_info import FunctionInfo
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
    if cuda_path is None:
        if sys.platform == 'linux':
            cuda_path = pathlib.Path("/usr/local/cuda/include")
            if not (cuda_path.exists() and cuda_path.is_dir()):
                cuda_path = None
        elif sys.platform == 'win32':
            ...
        elif sys.platform == 'darwin':
            ...
    else:
        cuda_path = pathlib.Path(cuda_path)
        cuda_path /= "include"

    return cuda_path


def _get_compute_capability(gpu_id) -> int:
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, gpu_id)
    ASSERT_DRV(err)

    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, gpu_id)
    ASSERT_DRV(err)

    return (major * 10) + minor


def compile_cuda_program(cuda_src_path: pathlib.Path, func_name, gpu_id):
    src = cuda_src_path.read_text()

    cuda_incl_path = _find_cuda_incl_path()
    if not cuda_incl_path:
        raise RuntimeError("Unable to determine CUDA include path. Please set CUDA_PATH environment variable.")

    opts = [
    # https://docs.nvidia.com/cuda/nvrtc/index.html#group__options
        f'--gpu-architecture=compute_{_get_compute_capability(gpu_id)}'.encode(),
        b'--ptxas-options=--warn-on-spills',    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-passing-specific-phase-options-ptxas-options
        b'-use_fast_math',
        b'--include-path=' + str(cuda_incl_path).encode(),
        b'-std=c++17',
        b'-default-device',
    #b'--restrict',
    #b'--device-int128'
    ]

    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(src), func_name.encode('utf-8'), 0, [], [])
    ASSERT_DRV(err)

    # Compile program
    err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        ASSERT_DRV(err)

        log = "0" * log_size
        e_log = log.encode('utf-8')
        err = nvrtc.nvrtcGetProgramLog(prog, e_log)
        print(e_log.decode('utf-8'))

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    err = nvrtc.nvrtcGetPTX(prog, ptx)

    return ptx


def initialize_cuda(gpu_id):
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    err, cuDevice = cuda.cuDeviceGet(gpu_id)
    ASSERT_DRV(err)

    # Create context
    # TODO: USE cuDevicePrimaryCtxRetain for faster intialization
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ASSERT_DRV(err)
    return context


def get_func_from_ptx(ptx, func_name):
    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    err, ptx_mod = cuda.cuModuleLoadData(ptx)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(ptx_mod, func_name.encode('utf-8'))
    ASSERT_DRV(err)

    return kernel


def _cuda_transfer_mem(usage, func, source_args: List, dest_args: List, arg_infos: List[ArgInfo]):
    for source_arg, dest_arg, arg_info in zip(source_args, dest_args, arg_infos):
        if usage in arg_info.usage.value:
            err, = func(dest_arg, source_arg, arg_info.total_byte_size)
            ASSERT_DRV(err)


def transfer_mem_host_to_cuda(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    _cuda_transfer_mem(
        usage='input',
        func=cuda.cuMemcpyHtoD,
        source_args=[a.ctypes.data for a in host_args],
        dest_args=device_args,
        arg_infos=arg_infos
    )


def transfer_mem_cuda_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    _cuda_transfer_mem(
        usage='output',
        func=cuda.cuMemcpyDtoH,
        source_args=device_args,
        dest_args=[a.ctypes.data for a in host_args],
        arg_infos=arg_infos
    )


def allocate_cuda_mem(arg_infos: List[ArgInfo]):
    device_mem = []

    for arg in arg_infos:
        size = arg.total_byte_size
        err, mem = cuda.cuMemAlloc(size)
        try:
            ASSERT_DRV(err)
        except:
            free_cuda_mem(device_mem)
            raise
        device_mem.append(mem)

    return device_mem


def free_cuda_mem(args):
    for arg in args:
        cuda.cuMemFree(arg)


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
        self.func_info = FunctionInfo(func)
        self.kernel = None
        self.device_mem = None
        self.ptrs = None
        self.start_event = None
        self.stop_event = None
        self.exec_time = 0.
        self.cuda_src_path = cuda_src_path
        self.context = None

    def init_runtime(self, benchmark: bool, gpu_id: int):
        self.context = initialize_cuda(gpu_id)

        ptx = _PTX_CACHE.get(self.cuda_src_path)
        if not ptx:
            _PTX_CACHE[self.cuda_src_path] = ptx = compile_cuda_program(self.cuda_src_path, self.func_info.name, gpu_id)

        self.kernel = get_func_from_ptx(ptx, self.func_info.name)

    def cleanup_runtime(self, benchmark: bool):
        cuda.cuCtxDestroy(self.context)

    def init_main(self, benchmark: bool, warmup_iters=0, args=[], gpu_id: int = 0):
        self.func_info.verify(args)
        self.device_mem = allocate_cuda_mem(self.func_info.arguments)

        if not benchmark:
            transfer_mem_host_to_cuda(device_args=self.device_mem, host_args=args, arg_infos=self.func_info.arguments)

        self.ptrs = device_args_to_ptr_list(self.device_mem)

        if self.hat_func.dynamic_shared_mem_bytes > 0:
            err, = cuda.cuFuncSetAttribute(self.kernel, cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.hat_func.dynamic_shared_mem_bytes)
            ASSERT_DRV(err)

        err, self.start_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)
        ASSERT_DRV(err)
        err, self.stop_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)
        ASSERT_DRV(err)

        for _ in range(warmup_iters):
            err, = cuda.cuLaunchKernel(
                self.kernel,
                *self.hat_func.launch_parameters,    # [ grid[x-z], block[x-z] ]
                self.hat_func.dynamic_shared_mem_bytes,
                0,    # stream
                self.ptrs.ctypes.data,    # kernel arguments
                0,    # extra (ignore)
            )

            if not benchmark:
                ASSERT_DRV(err)

    def main(self, benchmark: bool, iters=1, batch_size=1, args=[]) -> float:
        batch_timings: List[float] = []
        for _ in range(batch_size):
            err, = cuda.cuEventRecord(self.start_event, 0)
            ASSERT_DRV(err)

            for _ in range(iters):
                err, = cuda.cuLaunchKernel(
                    self.kernel,
                    *self.hat_func.launch_parameters,    # [ grid[x-z], block[x-z] ]
                    self.hat_func.dynamic_shared_mem_bytes,
                    0,    # stream
                    self.ptrs.ctypes.data,    # kernel arguments
                    0,    # extra (ignore)
                )

                if not benchmark:
                    ASSERT_DRV(err)

            err, = cuda.cuEventRecord(self.stop_event, 0)
            ASSERT_DRV(err)
            err, = cuda.cuEventSynchronize(self.stop_event)
            ASSERT_DRV(err)
            err, batch_time = cuda.cuEventElapsedTime(self.start_event, self.stop_event)
            ASSERT_DRV(err)
            batch_timings.append(batch_time)
            self.exec_time += batch_time

            if not benchmark:
                err, = cuda.cuCtxSynchronize()
                ASSERT_DRV(err)

        self.exec_time /= (iters * batch_size)
        return batch_timings

    def cleanup_main(self, benchmark: bool, args=[]):
        # If there's no device mem, that means allocation during initialization failed, which means nothing else needs to be cleaned up either
        if not benchmark and self.device_mem:
            transfer_mem_cuda_to_host(device_args=self.device_mem, host_args=args, arg_infos=self.func_info.arguments)
        if self.device_mem:
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
