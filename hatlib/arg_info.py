import ctypes
import numpy as np
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

from . import hat_file


@dataclass
class ArgInfo:
    """Extracts necessary information from the description of a function argument in a hat file"""
    hat_declared_type: str
    numpy_shape: Tuple[int]
    numpy_strides: Tuple[int]
    numpy_dtype: type
    element_num_bytes: int
    ctypes_pointer_type: Any
    usage: str = ""

    def __init__(self, param_description: hat_file.Parameter):
        self.hat_declared_type = param_description.declared_type
        self.numpy_shape = tuple(param_description.shape)
        self.usage = param_description.usage
        if self.hat_declared_type == "float16_t*":
            self.numpy_dtype = np.float16
            self.element_num_bytes = 2
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_uint16)    # same bitwidth as float16
        elif self.hat_declared_type == "float*":
            self.numpy_dtype = np.float32
            self.element_num_bytes = 4
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_float)
        elif self.hat_declared_type == "double*":
            self.numpy_dtype = np.float64
            self.element_num_bytes = 8
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_double)
        elif self.hat_declared_type == "int64_t*":
            self.numpy_dtype = np.int64
            self.element_num_bytes = 8
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_int64)
        elif self.hat_declared_type == "int32_t*":
            self.numpy_dtype = np.int32
            self.element_num_bytes = 4
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_int32)
        elif self.hat_declared_type == "int16_t*":
            self.numpy_dtype = np.int16
            self.element_num_bytes = 2
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_int16)
        elif self.hat_declared_type == "int8_t*":
            self.numpy_dtype = np.int8
            self.element_num_bytes = 1
            self.ctypes_pointer_type = ctypes.POINTER(ctypes.c_int8)

        else:
            raise NotImplementedError(f"Unsupported declared_type {self.hat_declared_type} in hat file")

        self.numpy_strides = tuple([self.element_num_bytes * x for x in param_description.affine_map])


# TODO: Update this to take a HATFunction instead, instead of arg_infos and function_name
def verify_args(args: List, arg_infos: List[ArgInfo], function_name: str):
    """ Verifies that a list of arguments matches a list of argument descriptions in a HAT file
    """
    # check number of args
    if len(args) != len(arg_infos):
        sys.exit(f"Error calling {function_name}(...): expected {len(arg_infos)} arguments but received {len(args)}")

    # for each arg
    for i in range(len(args)):
        arg = args[i]
        arg_info = arg_infos[i]

        # confirm that the arg is a numpy ndarray
        if not isinstance(arg, np.ndarray):
            sys.exit(
                "Error calling {function_name}(...): expected argument {i} to be <class 'numpy.ndarray'> but received {type(arg)}"
            )

        # confirm that the arg dtype matches the dexcription in the hat package
        if arg_info.numpy_dtype != arg.dtype:
            sys.exit(
                f"Error calling {function_name}(...): expected argument {i} to have dtype={arg_info.numpy_dtype} but received dtype={arg.dtype}"
            )

        # confirm that the arg shape is correct
        if arg_info.numpy_shape != arg.shape:
            sys.exit(
                f"Error calling {function_name}(...): expected argument {i} to have shape={arg_info.numpy_shape} but received shape={arg.shape}"
            )

        # confirm that the arg strides are correct
        if arg_info.numpy_strides != arg.strides:
            sys.exit(
                f"Error calling {function_name}(...): expected argument {i} to have strides={arg_info.numpy_strides} but received strides={arg.strides}"
            )
