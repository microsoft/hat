"""Loads a dynamically-linked HAT package in Python

Call 'load' to load a HAT package in Python. After loading, call the HAT
functions using numpy arrays as arguments. The shape, element type, and order
of each numpy array should exactly match the requirements of the HAT function.

For example:
    import numpy as np
    import hatlib as hat

    # load the package
    package = hat.load("my_package.hat")

    # print the function names
    for name in package.names:
        print(name)

    # create numpy arguments with the correct shape, dtype, and order
    A = np.ones([256,32], dtype=np.float32, order="C")
    B = np.ones([32,256], dtype=np.float32, order="C")
    D = np.ones([256,32], dtype=np.float32, order="C")
    E = np.ones([256,32], dtype=np.float32, order="C")

    # call a package function named 'my_func_698b5e5c'
    package.my_func_698b5e5c(A, B, D, E)
"""

import ctypes
import os
import sys
from typing import Tuple, Union
from collections import OrderedDict
from functools import partial

try:
    from . import hat_package
    from . import hat_file
    from .arg_info import ArgInfo, verify_args, generate_input_sets
except:
    import hat_file
    from arg_info import ArgInfo, verify_args, generate_input_sets




def generate_input_sets_for_hat_file(hat_path):
    hat_path = pathlib.Path(hat_path).absolute()
    t: hat_file.HATFile = toml.load(hat_path)
    return {
        func_name:
        generate_input_sets(list(map(ArgInfo, func_desc["arguments"])))
        for func_name, func_desc in t["functions"].items()
    }


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


def hat_description_to_python_function(hat_description: hat_file.HATFile,
                                       hat_details: AttributeDict):
    """ Creates a callable function based on a function description in a HAT
    package
    """

    for func_name, func_desc in hat_description["functions"].items():

        func_desc: hat_file.Function
        func_name: str

        launches = func_desc.get("launches")
        if not launches:
            hat_library: ctypes.CDLL = hat_details.shared_lib

            def f(function_name, hat_arg_descriptions, *args):
                # verify that the (numpy) input args match the description in
                # the hat file
                arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
                verify_args(args, arg_infos, function_name)

                # prepare the args to the hat package
                hat_args = [
                    arg.ctypes.data_as(arg_info.ctypes_pointer_type)
                    for arg, arg_info in zip(args, arg_infos)
                ]

                # call the function in the hat package
                hat_library[function_name](*hat_args)

            yield func_name, partial(f, func_desc["name"],
                                     func_desc["arguments"])

        else:
            device_func = hat_description.get("device_functions",
                                              {}).get(launches)

            func_runtime = func_desc.get("runtime")
            if not device_func:
                raise RuntimeError(
                    f"Couldn't find device function for loader: " + launches)
            if not func_runtime:
                raise RuntimeError(f"Couldn't find runtime for loader: " +
                                   launches)
            if func_runtime == "CUDA":
                global NOTIFY_ABOUT_CUDA
                if CUDA_AVAILABLE:
                    yield (func_name,
                           cuda_loader.create_loader_for_device_function(
                               device_func, hat_details))
                elif NOTIFY_ABOUT_CUDA:
                    print("CUDA functionality not available on this machine."
                          " Please install the cuda and pvnrtc python modules")
                    NOTIFY_ABOUT_CUDA = False
            elif func_runtime == "ROCM":
                global NOTIFY_ABOUT_ROCM
                if ROCM_AVAILABLE:
                    yield (func_name,
                           rocm_loader.create_loader_for_device_function(
                               device_func, hat_details))
                elif NOTIFY_ABOUT_ROCM:
                    print("ROCm functionality not available on this machine."
                          " Please install the ROCm 4.2 or higher")
                    NOTIFY_ABOUT_ROCM = False


def load(hat_path):
    """ Creates a class with static functions based on the function
    descriptions in a HAT package
    """
    # load the function decscriptions from the hat file
    hat_path = pathlib.Path(hat_path).absolute()
    t: hat_file.HATFile = toml.load(hat_path)
    hat_details = AttributeDict({"path": hat_path})

    # function_descriptions = t["functions"]
    hat_binary_filename = t["dependencies"]["link_target"]
    hat_binary_path = hat_path.parent / hat_binary_filename

    # check that the HAT library has a supported file extension
    supported_extensions = [".dll", ".so"]
    extension = hat_binary_path.suffix
    if extension and extension not in supported_extensions:
        sys.exit(f"Unsupported HAT library extension: {extension}")

    # load the hat_library:
    hat_library = ctypes.cdll.LoadLibrary(
        str(hat_binary_path)) if extension else None
    hat_details["shared_lib"] = hat_library

    # create dictionary of functions defined in the hat file
    function_dict = AttributeDict(
        dict(hat_description_to_python_function(t, hat_details)))
    return function_dict


def load2(
    hat_path,
    try_dynamic_load=True
) -> Tuple[hat_package.HATPackage, Union[AttributeDict, None]]:
    """
    Returns a HATPackage object loaded from the path provided. If
    `try_dynamic_load` is True, a non-empty dictionary object that can be used
    to invoke the functions in the HATPackage on the current system is the
    second returned object, `None` otherwise.
    """

    pkg = hat_package.HATPackage(hat_file_path=hat_path)

    # TODO: Add heuristics to determine whether loading is possible on this system
    function_dict = None

    if try_dynamic_load:
        try:
            # TODO: Figure out how to communicate failure better
            function_dict = hat_package.hat_package_to_func_dict(pkg)
        except:
            pass

    return pkg, function_dict
