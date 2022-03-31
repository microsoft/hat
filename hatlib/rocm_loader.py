import ctypes
import pathlib
import numpy as np
from functools import reduce
from typing import List

from .arg_info import ArgInfo, verify_args
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


def create_loader_for_device_function(device_func: Function, hat_dir_path: str):
    if not device_func.provider:
        raise RuntimeError("Expected a provider for the device function")

    rocm_src_path: pathlib.Path = pathlib.Path(hat_dir_path) / device_func.provider
    func_name = device_func.name

    rocm_program = compile_rocm_program(rocm_src_path, func_name)

    initialize_rocm()

    kernel = get_func_from_rocm_program(rocm_program, func_name)

    hat_arg_descriptions = device_func.arguments
    arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
    launch_parameters = device_func.launch_parameters

    class DataStruct(ctypes.Structure):
        _fields_ = [(f"arg{i}", ctypes.c_void_p) for i in range(len(arg_infos))]

    def f(*args):
        verify_args(args, arg_infos, func_name)
        device_mem = allocate_rocm_mem(arg_infos)
        transfer_mem_host_to_rocm(device_args=device_mem, host_args=args, arg_infos=arg_infos)
        data = DataStruct(*device_mem)

        hipModuleLaunchKernel(
            kernel,
            *launch_parameters,    # [ grid[x-z], block[x-z] ]
            0,    # dynamic shared memory
            0,    # stream
            data,    # data
        )
        hipDeviceSynchronize()

        transfer_mem_rocm_to_host(device_args=device_mem, host_args=args, arg_infos=arg_infos)

    return f
