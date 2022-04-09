import ctypes
import pathlib
import numpy as np
from functools import reduce
from typing import List

from .arg_info import ArgInfo, verify_args
from .callable_func import CallableFunc
from .hat_file import Function
from .gpu_headers import ROCM_HEADER_MAP
from .pyhip.hip import *
from .pyhip.hiprtc import *


def _arg_size(arg_info: ArgInfo):
    return arg_info.element_num_bytes * reduce(lambda x, y: x * y, arg_info.numpy_shape)


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
    print(hiprtcGetProgramLog(prog))
    code = hiprtcGetCode(prog)

    return code


def get_func_from_rocm_program(rocm_program, func_name):
    rocm_module = hipModuleLoadData(rocm_program)
    kernel = hipModuleGetFunction(rocm_module, func_name)
    return kernel


def allocate_rocm_mem(arg_infos: List[ArgInfo]):
    device_mem = []
    for arg in arg_infos:
        mem = hipMalloc(_arg_size(arg))
        device_mem.append(mem)

    return device_mem


def free_rocm_mem(args):
    for arg in args:
        hipFree(arg)


def transfer_mem_host_to_rocm(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'input' in arg_info.usage.value:
            hipMemcpy_htod(dst=device_arg, src=host_arg.ctypes.data, count=_arg_size(arg_info))


def transfer_mem_rocm_to_host(device_args: List, host_args: List[np.array], arg_infos: List[ArgInfo]):
    for device_arg, host_arg, arg_info in zip(device_args, host_args, arg_infos):
        if 'output' in arg_info.usage.value:
            hipMemcpy_dtoh(dst=host_arg.ctypes.data, src=device_arg, count=_arg_size(arg_info))


def device_args_to_ptr_list(device_args: List):
    ptrs = [np.array([int(d_arg)], dtype=np.uint64) for d_arg in device_args]
    ptrs = np.array([ptr.ctypes.data for ptr in ptrs], dtype=np.uint64)

    return ptrs


class RocmCallableFunc(CallableFunc):

    def __init__(self, func: Function, kernel) -> None:
        super().__init__()
        self.hat_func = func
        self.func_name = func.name
        self.kernel = kernel
        hat_arg_descriptions = func.arguments
        self.arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
        self.launch_params = func.launch_parameters
        self.device_mem = None
        self.ptrs = None
        self.stream = None
        self.start_event = None
        self.stop_event = None
        self.exec_time = 0.

    def init_runtime(self):
        pass

    def cleanup_runtime(self):
        pass

    def init_main(self, warmup_iters=0, args=[]):
        verify_args(args, self.arg_infos, self.func_name)
        self.device_mem = allocate_rocm_mem(self.arg_infos)
        transfer_mem_host_to_rocm(device_args=self.device_mem, host_args=args, arg_infos=self.arg_infos)

        class DataStruct(ctypes.Structure):
            _fields_ = [(f"arg{i}", ctypes.c_void_p) for i in range(len(self.arg_infos))]

        self.data = DataStruct(*self.device_mem)

        self.stream = hipStreamCreate()

        self.start_event = hipEventCreate()
        self.stop_event = hipEventCreate()

        for _ in range(warmup_iters):
            hipModuleLaunchKernel(
                self.kernel,
                *self.launch_params,    # [ grid[x-z], block[x-z] ]
                0,    # dynamic shared memory
                0,    # stream
                self.data,    # data
            )
        else:
            hipStreamSynchronize(self.stream)

    def main(self, iters=1, batch_size=1, args=[]) -> float:
        batch_timings:List[float] = []
        for _ in range(batch_size):
            hipEventRecord(self.start_event)

            for _ in range(iters):
                hipModuleLaunchKernel(
                    self.kernel,
                    *self.launch_params,    # [ grid[x-z], block[x-z] ]
                    0,    # dynamic shared memory
                    0,    # stream
                    self.data,    # data
                )

            hipEventRecord(self.stop_event)
            hipEventSynchronize(self.stop_event)
            batch_time = hipEventElapsedTime(self.start_event, self.stop_event)
            batch_timings.append(batch_time)
            self.exec_time += batch_time

        self.exec_time /= (iters * batch_size)
        return batch_timings

    def cleanup_main(self, args=[]):
        transfer_mem_rocm_to_host(device_args=self.device_mem, host_args=args, arg_infos=self.arg_infos)
        free_rocm_mem(self.device_mem)
        hipEventDestroy(self.start_event)
        hipEventDestroy(self.stop_event)
        hipStreamDestroy(self.stream)


initialize_rocm()


def create_loader_for_device_function(device_func: Function, hat_dir_path: str):
    if not device_func.provider:
        raise RuntimeError("Expected a provider for the device function")

    rocm_src_path: pathlib.Path = pathlib.Path(hat_dir_path) / device_func.provider
    func_name = device_func.name

    rocm_program = compile_rocm_program(rocm_src_path, func_name)

    kernel = get_func_from_rocm_program(rocm_program, func_name)

    return RocmCallableFunc(device_func, kernel)
