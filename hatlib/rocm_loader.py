import ctypes
import pathlib
import numpy as np
from typing import List

from .arg_info import ArgInfo
from .callable_func import CallableFunc
from .function_info import FunctionInfo
from .hat_file import Function
from .gpu_headers import ROCM_HEADER_MAP
from .pyhip.hip import *
from .pyhip.hiprtc import *


def initialize_rocm():
    # Initialize ROCM Driver API
    hipInit(0)


def compile_rocm_program(rocm_src_path: pathlib.Path, func_name):
    src = rocm_src_path.read_text()

    prog = hiprtcCreateProgram(
        source=src,
        name=func_name + ".cu",
        header_names=ROCM_HEADER_MAP.keys(),
        header_sources=ROCM_HEADER_MAP.values()
    )
    device_properties = hipGetDeviceProperties(0)
    hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}', '-D__HIP_PLATFORM_AMD__'])
    code = hiprtcGetCode(prog)

    return code


def get_func_from_rocm_program(rocm_program, func_name):
    rocm_module = hipModuleLoadData(rocm_program)
    kernel = hipModuleGetFunction(rocm_module, func_name)
    return kernel


cached_mem = []


def allocate_rocm_mem(benchmark: bool, arg_infos: List[ArgInfo], gpu_id: int):
    device_mem = []
    for arg in arg_infos:
        try:
            memory_cache = cached_mem[gpu_id]
            if benchmark and arg.total_byte_size in memory_cache:
                mem = memory_cache[arg.total_byte_size]
            else:
                mem = hipMalloc(arg.total_byte_size)
                if benchmark:
                    memory_cache[arg.total_byte_size] = mem
        except:
            free_rocm_mem(device_mem)
            raise
        device_mem.append(mem)

    return device_mem


def free_rocm_mem(args):
    for arg in args:
        hipFree(arg)


def transfer_mem_host_to_rocm(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'input' in arg_info.usage.value:
            hipMemcpy_htod(dst=device_arg, src=host_arg.ctypes.data, count=arg_info.total_byte_size)


def transfer_mem_rocm_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'output' in arg_info.usage.value:
            hipMemcpy_dtoh(dst=host_arg.ctypes.data, src=device_arg, count=arg_info.total_byte_size)


def device_args_to_ptr_list(device_args: List):
    ptrs = [np.array([int(d_arg)], dtype=np.uint64) for d_arg in device_args]
    ptrs = np.array([ptr.ctypes.data for ptr in ptrs], dtype=np.uint64)

    return ptrs


_HSACO_CACHE = {}


class RocmCallableFunc(CallableFunc):

    def __init__(self, func: Function, rocm_src_path: str) -> None:
        super().__init__()
        self.hat_func = func
        self.func_info = FunctionInfo(func)
        self.kernel = None
        self.device_mem = None
        self.ptrs = None
        self.stream = None
        self.start_event = None
        self.stop_event = None
        self.exec_time = 0.
        self.rocm_src_path = rocm_src_path

    def init_runtime(self, benchmark: bool, gpu_id: int):
        if not benchmark:
            initialize_rocm()

        hipSetDevice(gpu_id)

        # Add a separate cache for each gpu since device memory is not shareable (duh!)
        while len(cached_mem) <= gpu_id:
            cached_mem.append({})

        rocm_program = _HSACO_CACHE.get(self.rocm_src_path)
        if not rocm_program:
            _HSACO_CACHE[self.rocm_src_path
                         ] = rocm_program = compile_rocm_program(self.rocm_src_path, self.func_info.name)

        self.kernel = get_func_from_rocm_program(rocm_program, self.func_info.name)

    def cleanup_runtime(self, benchmark: bool):
        pass

    def init_main(self, benchmark: bool, warmup_iters=0, args=[], gpu_id: int = 0):
        self.func_info.verify(args)
        self.device_mem = allocate_rocm_mem(benchmark, self.func_info.arguments, gpu_id)

        if not benchmark:
            transfer_mem_host_to_rocm(device_args=self.device_mem, host_args=args, arg_infos=self.func_info.arguments)

        class DataStruct(ctypes.Structure):
            _fields_ = [(f"arg{i}", ctypes.c_void_p) for i in range(len(self.func_info.arguments))]

        self.data = DataStruct(*self.device_mem)

        self.start_event = hipEventCreate()
        self.stop_event = hipEventCreate()

        for _ in range(warmup_iters):
            hipModuleLaunchKernel(
                self.kernel,
                *self.hat_func.launch_parameters,    # [ grid[x-z], block[x-z] ]
                self.hat_func.dynamic_shared_mem_bytes,
                0,    # stream
                self.data,    # data
            )

    def main(self, benchmark: bool, iters=1, batch_size=1, args=[]) -> float:
        batch_timings: List[float] = []
        for _ in range(batch_size):
            hipEventRecord(self.start_event)

            for _ in range(iters):
                hipModuleLaunchKernel(
                    self.kernel,
                    *self.hat_func.launch_parameters,    # [ grid[x-z], block[x-z] ]
                    self.hat_func.dynamic_shared_mem_bytes,    # dynamic shared memory
                    0,    # stream
                    self.data,    # data
                )

            hipEventRecord(self.stop_event)
            hipEventSynchronize(self.stop_event)
            batch_time = hipEventElapsedTime(self.start_event, self.stop_event)
            batch_timings.append(batch_time)
            self.exec_time += batch_time

            if not benchmark:
                hipDeviceSynchronize()

        self.exec_time /= (iters * batch_size)
        return batch_timings

    def cleanup_main(self, benchmark: bool, args=[]):
        # If there's no device mem, that means allocation during initialization failed, which means nothing else needs to be cleaned up either
        if not benchmark and self.device_mem:
            transfer_mem_rocm_to_host(device_args=self.device_mem, host_args=args, arg_infos=self.func_info.arguments)
            free_rocm_mem(self.device_mem)
        hipDeviceSynchronize()

        if self.start_event:
            hipEventDestroy(self.start_event)

        if self.stop_event:
            hipEventDestroy(self.stop_event)


def create_loader_for_device_function(device_func: Function, hat_dir_path: str):
    if not device_func.provider:
        raise RuntimeError("Expected a provider for the device function")

    rocm_src_path: pathlib.Path = pathlib.Path(hat_dir_path) / device_func.provider

    return RocmCallableFunc(device_func, rocm_src_path)
