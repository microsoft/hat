import ctypes
import numpy as np
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

from . import hat_file

# hat_declared_type : [ ctype, dtype_str ]
ARG_TYPES = {
    "int8_t*" : [ ctypes.c_int8, "int8" ],
    "int16_t*" : [ ctypes.c_int16, "int16" ],
    "int32_t*" : [ ctypes.c_int32, "int32" ],
    "int64_t*" : [ ctypes.c_int64, "int64" ],
    "uint8_t*" : [ ctypes.c_uint8, "uint8" ],
    "uint16_t*" : [ ctypes.c_uint16, "uint16" ],
    "uint32_t*" : [ ctypes.c_uint32, "uint32" ],
    "uint64_t*" : [ ctypes.c_uint64, "uint64" ],
    "float16_t*" : [ ctypes.c_uint16, "float16" ], # same bitwidth as uint16
    "float*" : [ ctypes.c_float, "float32" ],
    "double*" : [ ctypes.c_double, "float64" ],
}
CTYPE_ENTRY = 0
DTYPE_ENTRY = 1

@dataclass
class ArgInfo:
    """Extracts necessary information from the description of a function argument in a hat file"""
    hat_declared_type: str
    numpy_shape: Tuple[int]
    numpy_strides: Tuple[int]
    numpy_dtype: type
    element_num_bytes: int
    ctypes_pointer_type: Any
    usage: hat_file.UsageType = None

    def __init__(self, param_description: hat_file.Parameter):
        self.hat_declared_type = param_description.declared_type
        self.numpy_shape = tuple(param_description.shape)
        self.usage = param_description.usage

        if not self.hat_declared_type in ARG_TYPES:
            raise NotImplementedError(f"Unsupported declared_type {self.hat_declared_type} in hat file")

        self.ctypes_pointer_type = ctypes.POINTER(ARG_TYPES[self.hat_declared_type][CTYPE_ENTRY])
        self.numpy_dtype = np.dtype(ARG_TYPES[self.hat_declared_type][DTYPE_ENTRY])
        self.element_num_bytes = self.numpy_dtype.itemsize
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
