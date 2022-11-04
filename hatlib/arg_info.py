import ctypes
import numpy as np
import re
from dataclasses import dataclass
from typing import Any, Tuple, Union

from . import hat_file

# element_type : [ ctype, dtype_str ]
ARG_TYPES = {
    "int8_t": [ctypes.c_int8, "int8"],
    "int16_t": [ctypes.c_int16, "int16"],
    "int32_t": [ctypes.c_int32, "int32"],
    "int64_t": [ctypes.c_int64, "int64"],
    "uint8_t": [ctypes.c_uint8, "uint8"],
    "uint16_t": [ctypes.c_uint16, "uint16"],
    "uint32_t": [ctypes.c_uint32, "uint32"],
    "uint64_t": [ctypes.c_uint64, "uint64"],
    "float16_t": [ctypes.c_uint16, "float16"],    # same bitwidth as uint16
    "bfloat16_t": [ctypes.c_uint16, "bfloat16"],
    "float": [ctypes.c_float, "float32"],
    "double": [ctypes.c_double, "float64"],
}
CTYPE_ENTRY = 0
DTYPE_ENTRY = 1


@dataclass
class ArgInfo:
    """Extracts necessary information from the description of a function argument in a hat file"""
    name: str
    hat_declared_type: str
    shape: Tuple[Union[int, str], ...]    # int for affine_arrays, str symbols for runtime_arrays
    numpy_strides: Tuple[int, ...]
    numpy_dtype: type
    element_num_bytes: int
    element_strides: Tuple[int, ...]
    total_element_count: Union[int, str]
    total_byte_size: Union[int, str]
    ctypes_type: Any
    pointer_level: int
    usage: hat_file.UsageType = None

    def _get_type(self, type_str):
        if type_str == "bfloat16":
            from bfloat16 import bfloat16
            return bfloat16

        return np.dtype(type_str)

    def _get_pointer_level(self, declared_type: str):
        pos = declared_type.find("*")
        if pos == -1:
            return 0
        return declared_type[pos:].count("*")

    def __init__(self, param_description: hat_file.Parameter):
        self.name = param_description.name
        self.hat_declared_type = param_description.declared_type
        self.shape = tuple(param_description.shape)
        self.usage = param_description.usage
        self.pointer_level = self._get_pointer_level(self.hat_declared_type)
        element_type = self.hat_declared_type[:(-1 * self.pointer_level)] \
            if self.pointer_level else self.hat_declared_type

        if not element_type in ARG_TYPES:
            raise NotImplementedError(f"Unsupported element_type {element_type} in hat file")

        ctypes_type = ARG_TYPES[element_type][CTYPE_ENTRY]
        dtype_entry = ARG_TYPES[element_type][DTYPE_ENTRY]
        self.numpy_dtype = self._get_type(dtype_entry)
        self.element_num_bytes = 2 if dtype_entry == "bfloat16" else self.numpy_dtype.itemsize

        if param_description.logical_type == hat_file.ParameterType.AffineArray:
            self.ctypes_type = ctypes.POINTER(ctypes_type)
            if self.shape:
                self.element_strides = param_description.affine_map
                self.numpy_strides = tuple([self.element_num_bytes * x for x in self.element_strides])

                major_dim = self.element_strides.index(max(self.element_strides))
                self.total_element_count = self.shape[major_dim] * self.element_strides[major_dim]

            else:
                self.shape = [1]
                self.element_strides = self.numpy_strides = [self.element_num_bytes]
                self.total_element_count = 1
            self.total_byte_size = self.element_num_bytes * self.total_element_count

        elif param_description.logical_type == hat_file.ParameterType.RuntimeArray:
            self.ctypes_type = ctypes.POINTER(ctypes_type)
            self.total_byte_size = f"{self.element_num_bytes} * {param_description.size}"
            self.total_element_count = param_description.size
            # assume the sizes are in shape order
            self.shape = re.split(r"\s?\*\s?", param_description.size)

        elif param_description.logical_type == hat_file.ParameterType.Element:
            if param_description.usage == hat_file.UsageType.Input or (element_type == 'int64_t' and param_description.usage == hat_file.UsageType.InputOutput):
                self.ctypes_type = ctypes_type
            else:
                self.ctypes_type = ctypes.POINTER(ctypes_type)
            self.shape = [1]
            self.element_strides = self.numpy_strides = [self.element_num_bytes]
            self.total_element_count = 1
            self.total_byte_size = self.element_num_bytes * self.total_element_count

        else:
            raise ValueError(f"Unknown logical type {param_description.logical_type} in hat file")

    @property
    def is_constant_shaped(self):
        return all(integer_like(s) for s in self.shape)


def integer_like(s: Any):
    # handle types that have an int conversion, such as tomlkit.items.Integer or str
    try:
        _ = int(s)
        return True
    except:
        return False
