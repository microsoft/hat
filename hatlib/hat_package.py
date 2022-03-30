#!/usr/bin/env python3

# Utility to parse and validate a HAT package

import ctypes
from typing import List, Union
from collections import OrderedDict
from functools import partial

from .hat_file import HATFile, Function, Parameter
from .arg_info import ArgInfo, verify_args

import os


class HATPackage:

    def __init__(self, hat_file_path):
        """A HAT Package is defined to be a HAT file and corresponding binary file, located in the same directory.
        The binary file is specified in the HAT file's link_target attribute.
        The same binary file can be referenced by many HAT files.
        Many HAT packages can exist in the same directory.
        An instance of HATPackage is created by giving HATPackage the file path to the .hat file."""
        self.name = os.path.basename(hat_file_path)
        self.hat_file_path = hat_file_path
        self.hat_file = HATFile.Deserialize(hat_file_path)

        self.link_target = self.hat_file.dependencies.link_target
        self.link_target_path = os.path.join(
            os.path.split(self.hat_file_path)[0], self.hat_file.dependencies.link_target
        )

        self.functions = self.hat_file.functions

    def get_functions(self):
        return self.hat_file.functions

    def get_functions_for_target(self, os: str, arch: str, required_extensions: list = []):
        all_functions = self.get_functions()

        def matches_target(hat_function):
            hat_file = hat_function.hat_file
            if hat_file.target.required.os != os or hat_file.target.required.cpu.architecture != arch:
                return False
            for required_ext in required_extensions:
                if required_ext not in hat_file.target.required.cpu.extensions:
                    return False
            return True

        return list(filter(matches_target, all_functions))


class AttributeDict(OrderedDict):
    """ Dictionary that allows entries to be accessed like attributes
    """
    __getattr__ = OrderedDict.__getitem__

    @property
    def names(self):
        return list(self.keys())

    def __getitem__(self, key):
        for k, v in self.items():
            if k.startswith(key):
                return v
        return OrderedDict.__getitem__(key)


def _make_cpu_func(shared_lib: ctypes.CDLL, function_name: str, arg_infos: List[Parameter]):
    arg_infos = [ArgInfo(d) for d in arg_infos]
    fn = shared_lib[function_name]

    def f(*args):
        # verify that the (numpy) input args match the description in
        # the hat file
        verify_args(args, arg_infos, function_name)

        # prepare the args to the hat package
        hat_args = [arg.ctypes.data_as(arg_info.ctypes_pointer_type) for arg, arg_info in zip(args, arg_infos)]

        # call the function in the hat package
        fn(*hat_args)

    return f


def _make_device_func(func_runtime: str, hat_dir_path: str, func: Function):
    if func_runtime == "CUDA":
        from . import cuda_loader
        return cuda_loader.create_loader_for_device_function(func, hat_dir_path)
    elif func_runtime == "ROCM":
        from . import rocm_loader
        return rocm_loader.create_loader_for_device_function(func, hat_dir_path)


def _load_pkg_binary_module(hat_pkg: HATPackage):
    shared_lib = None
    if os.path.isfile(hat_pkg.link_target_path):

        supported_extensions = [".dll", ".so"]
        _, extension = os.path.splitext(hat_pkg.link_target_path)

        if extension and extension not in supported_extensions:
            # TODO: Should this be an error? Maybe just move on to the
            # device function section?
            raise RuntimeError(f"Unsupported HAT library extension: {extension}")

        hat_binary_path = os.path.abspath(hat_pkg.link_target_path)

        # load the hat_library:
        hat_library = ctypes.cdll.LoadLibrary(hat_binary_path) if extension else None
        shared_lib = hat_library

    return shared_lib


def hat_package_to_func_dict(hat_pkg: HATPackage) -> AttributeDict:

    try:
        try:
            from . import cuda_loader
        except ModuleNotFoundError:
            import cuda_loader
    except:
        CUDA_AVAILABLE = False
    else:
        CUDA_AVAILABLE = True

    try:
        try:
            from . import rocm_loader
        except ModuleNotFoundError:
            import rocm_loader
    except:
        ROCM_AVAILABLE = False
    else:
        ROCM_AVAILABLE = True

    NOTIFY_ABOUT_CUDA = not CUDA_AVAILABLE
    NOTIFY_ABOUT_ROCM = not ROCM_AVAILABLE

    # check that the HAT library has a supported file extension
    func_dict = AttributeDict()
    shared_lib = _load_pkg_binary_module(hat_pkg)
    hat_dir_path, _ = os.path.split(hat_pkg.hat_file_path)

    for func_name, func_desc in hat_pkg.hat_file.function_map.items():

        launches = func_desc.launches
        if not launches and shared_lib:

            func_dict[func_name] = _make_cpu_func(shared_lib, func_desc.name, func_desc.arguments)
        else:
            device_func = hat_pkg.hat_file.device_function_map.get(launches)

            func_runtime = func_desc.runtime
            if not device_func:
                raise RuntimeError(f"Couldn't find device function for loader: " + launches)
            if not func_runtime:
                raise RuntimeError(f"Couldn't find runtime for loader: " + launches)

            # TODO: Generalize this concept to work so it's not CUDA/ROCM specific
            if func_runtime == "CUDA" and not CUDA_AVAILABLE:

                # TODO: printing to stdout only makes sense in tool mode
                if NOTIFY_ABOUT_CUDA:
                    print(
                        "CUDA functionality not available on this machine. Please install the cuda and pvnrtc python modules"
                    )
                    NOTIFY_ABOUT_CUDA = False

                continue

            elif func_runtime == "ROCM" and not ROCM_AVAILABLE:

                # TODO: printing to stdout only makes sense in tool mode
                if NOTIFY_ABOUT_ROCM:
                    print("ROCm functionality not available on this machine. Please install ROCm 4.2 or higher")
                    NOTIFY_ABOUT_ROCM = False

                continue

            func_dict[func_name] = _make_device_func(
                func_runtime=func_runtime, hat_dir_path=hat_dir_path, func=device_func
            )

    return func_dict
