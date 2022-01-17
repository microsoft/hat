#!/usr/bin/env python3

"""Loads a dynamically-linked HAT package in Python

Call 'load' to load a HAT package in Python. After loading, call the HAT functions using numpy
arrays as arguments. The shape, element type, and order of each numpy array should exactly match
the requirements of the HAT function. 

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

import sys
import toml
import ctypes
import os
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Tuple

@dataclass
class ArgInfo:
    """Extracts necessary information from the description of a function argument in a hat file"""
    hat_declared_type: str
    numpy_shape: Tuple[int]
    numpy_strides: Tuple[int]
    numpy_dtype: type
    element_num_bytes: int
    ctypes_pointer_type: Any 

    def __init__(self, param_description):
        self.hat_declared_type = param_description["declared_type"]
        self.numpy_shape = tuple(param_description["shape"])
        if self.hat_declared_type == "float*":
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

        self.numpy_strides = tuple([self.element_num_bytes * x for x in param_description["affine_map"]])


def verify_args(args, arg_infos, function_name):
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
            sys.exit("Error calling {function_name}(...): expected argument {i} to be <class 'numpy.ndarray'> but received {type(arg)}")

        # confirm that the arg dtype matches the dexcription in the hat package
        if arg_info.numpy_dtype != arg.dtype:
            sys.exit(f"Error calling {function_name}(...): expected argument {i} to have dtype={arg_info.numpy_dtype} but received dtype={arg.dtype}")

        # confirm that the arg shape is correct
        if arg_info.numpy_shape != arg.shape:
            sys.exit(f"Error calling {function_name}(...): expected argument {i} to have shape={arg_info.numpy_shape} but received shape={arg.shape}")

        # confirm that the arg strides are correct
        if arg_info.numpy_strides != arg.strides:
            sys.exit(f"Error calling {function_name}(...): expected argument {i} to have strides={arg_info.numpy_strides} but received strides={arg.strides}")


def hat_description_to_python_function(hat_description, hat_library):
    """ Creates a callable function based on a function description in a HAT package
    """    
    hat_arg_descriptions = hat_description["arguments"]
    function_name = hat_description["name"]

    def f(*args):
        # verify that the (numpy) input args match the description in the hat file 
        arg_infos = [ArgInfo(d) for d in hat_arg_descriptions]
        verify_args(args, arg_infos, function_name)
        
        # prepare the args to the hat package
        hat_args = [arg.ctypes.data_as(arg_info.ctypes_pointer_type) for arg, arg_info in zip   (args, arg_infos)]
        
        # call the function in the hat package
        hat_library[function_name](*hat_args)

    return f


class AttributeDict(OrderedDict):
    """ Dictionary that allows entries to be accessed like attributes
    """
    __getattr__ = OrderedDict.__getitem__

    @property
    def names(self):
        return list(self.keys())


def load(hat_path):
    """ Creates a class with static functions based on the function descriptions in a HAT package
    """
    # load the function decscriptions from the hat file
    hat_path = os.path.abspath(hat_path)
    t = toml.load(hat_path)

    function_descriptions = t["functions"]
    hat_binary_filename = t["dependencies"]["link_target"] 
    hat_binary_path = os.path.join(os.path.dirname(hat_path), hat_binary_filename)

    # check that the HAT library has a supported file extension
    supported_extensions = [".dll", ".so"]
    _, extension = os.path.splitext(hat_binary_path)
    if extension not in supported_extensions:
        sys.exit(f"Unsupported HAT library extension: {extension}")

    # load the hat_library: 
    hat_library = ctypes.cdll.LoadLibrary(hat_binary_path)

    # create dictionary of functions defined in the hat file
    function_dict = AttributeDict({key : hat_description_to_python_function(val, hat_library) for key,val in function_descriptions.items()})
    return function_dict
