from typing import Any, List, Mapping
from ctypes import byref
import numpy as np
import random

from .arg_info import ArgInfo, integer_like
from . import hat_file


class ArgValue:
    """An argument containing a scalar, ndarray, or pointer value.
    Used for calling HAT functions from ctypes"""

    def __init__(self, arg_info: ArgInfo, value: Any = None):
        # TODO: set the free and alloc function symbols here?
        self.arg_info = arg_info
        self.pointer_level = arg_info.pointer_level
        self.ctypes_type = arg_info.ctypes_type

        if self.pointer_level > 2:    # punt until we really need this
            raise NotImplementedError("Pointer levels > 2 are not supported")

        self.value = value
        if self.value is None:
            if not self.pointer_level:
                raise ValueError("A value is required for non-pointers")
            else:
                # no value provided, allocate the pointer
                self.allocate()
        elif type(self.value) in [int, float]:
            self.value = self.arg_info.numpy_dtype.type(self.value)
        self.dim_values = None

    def allocate(self):
        if not self.pointer_level:
            return    # nothing to do
        if self.value:
            return    # value already assigned, nothing to do

        if self.pointer_level == 1:
            # allocate an ndarray with random input values
            self.value = np.lib.stride_tricks.as_strided(
                np.random.rand(self.arg_info.total_element_count).astype(self.arg_info.numpy_dtype),
                shape=self.arg_info.shape,
                strides=self.arg_info.numpy_strides
            )
        elif self.pointer_level == 2:
            # allocate a pointer. HAT function will perform the actual allocation.
            self.value = self.ctypes_type()

    def as_carg(self):
        "Return the C interface for this argument"
        if self.pointer_level:
            if isinstance(self.value, np.ndarray):
                return self.value.ctypes.data_as(self.ctypes_type)
            else:
                return byref(self.value)
        else:
            return self.ctypes_type(self.value)

    def verify(self, desc: ArgInfo):
        "Verifies that this argument matches an argument description"
        if desc.pointer_level == 1:
            if not isinstance(self.value, np.ndarray):
                raise ValueError(f"expected argument to be <class 'numpy.ndarray'> but received {type(self.value)}")

            if desc.numpy_dtype != self.value.dtype:
                raise ValueError(
                    f"expected argument to have dtype={desc.numpy_dtype} but received dtype={self.value.dtype}"
                )

            if desc.is_constant_shaped:
                if self.value.size > 1:
                    # confirm that the arg shape is correct (numpy represents shapes as tuples)
                    desc_shape = tuple(map(int, desc.shape))
                    if desc_shape != self.value.shape:
                        raise ValueError(
                            f"expected argument to have shape={desc_shape} but received shape={self.value.shape}"
                        )

                    # confirm that the arg strides are correct (numpy represents strides as tuples)
                    desc_numpy_strides = tuple(desc.numpy_strides) if hasattr(desc, 'numpy_strides') else tuple(
                        map(lambda x: x * desc.element_num_bytes, desc_shape[1:] + (1, ))
                    )
                    if desc_numpy_strides != self.value.strides:
                        raise ValueError(
                            f"expected argument to have strides={desc_numpy_strides} but received strides={self.value.strides}"
                        )
                else:
                    # Will raise ValueError if total_element_count can't be converted to int
                    desc.total_element_count = int(desc.total_element_count)

                    # special casing for size=1 arrays
                    if self.value.size != desc.total_element_count:
                        raise ValueError(
                            f"expected argument to have size={desc.total_element_count} but received shape={self.value.size}"
                        )
        else:
            pass    # TODO - support other pointer levels

    def __repr__(self):
        if self.pointer_level:
            if isinstance(self.value, np.ndarray):
                return ",".join(map(str, self.value.ravel()[:32]))
            else:
                try:
                    if self.dim_values:
                        # cross-reference the dimension output values to pretty print the output
                        shape = [d.value[0] for d in self.dim_values]    # stored as single-element ndarrays
                        s = repr(np.ctypeslib.as_array(self.value, shape))
                    else:
                        s = repr(self.value.contents)
                except Exception as e:
                    if e.args[0].startswith("NULL pointer"):
                        s = f"{repr(self.value)} nullptr"
                    else:
                        raise (e)
                return s
        else:
            return repr(self.value)

    def __del__(self):
        if self.pointer_level == 2:
            pass    # TODO - free the pointer, presumably calling a symbol passed into this ArgValue


def get_dimension_arg_indices(array_arg: ArgInfo, all_arguments: List[ArgInfo]) -> List[int]:
    # Returns the dimension argument indices in shape order for an array argument
    indices = []
    for sym_name in filter(lambda x: x and not integer_like(x), array_arg.shape):
        for i, info in enumerate(all_arguments):
            if info.name == sym_name:    # limitation: only string shapes are supported
                indices.append(i)
                break
        else:
            # not found
            raise RuntimeError(f"{sym_name} is not an argument to the function")    # likely an invalid HAT file
    return indices


def _gen_random_data(dtype, shape):
    dtype = np.uint16 if dtype == "bfloat16" else dtype
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if isinstance(dtype, type) and issubclass(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        min_num = iinfo.min
        max_num = iinfo.max
        data = np.random.randint(low=min_num, high=max_num, size=tuple(shape), dtype=dtype)
    else:
        data = np.random.random(tuple(shape)).astype(dtype)

    return data


def generate_arg_values(arguments: List[ArgInfo], dim_names_to_values={}) -> List[ArgValue]:
    """Generate argument values from argument descriptions
    Input and input/output affine_arrays: initialized with random inputs
    Input and input/output runtime_arrays: initialized with arbitrary dimensions and random inputs
    Output elements and runtime_arrays: pointers are allocated
    """

    def generate_dim_value():
        return random.choice([2, 3, 4])    # example dimension values

    values = []

    for arg in arguments:
        if arg.usage != hat_file.UsageType.Output and not arg.is_constant_shaped:
            # input runtime arrays
            dim_args: Mapping[str, ArgInfo] = {
                arguments[i].name: arguments[i]
                for i in get_dimension_arg_indices(arg, arguments)
            }

            # assign shape values to the corresponding dimension arguments
            shape = []
            if len(arg.shape) == 1 and arg.shape[0] == '':    # takes cares of shapes of type ['']
                shape = [1]
            else:
                for d in arg.shape:
                    if integer_like(d):
                        shape.append(int(d))
                    else:
                        assert d in dim_args
                        if d not in dim_names_to_values:
                            shape.append(generate_dim_value())
                            dim_names_to_values[d] = ArgValue(dim_args[d], shape[-1])
                        else:
                            v = dim_names_to_values[d].value
                            shape.append(v if isinstance(v, np.integer) or type(v) == int else v[0])

            # materialize an array input using the generated shape
            runtime_array_inputs = _gen_random_data(arg.numpy_dtype, shape)
            values.append(ArgValue(arg, runtime_array_inputs))

        elif arg.name in dim_names_to_values:
            # input element that is a dimension value (populated when its input runtime array is created)
            # BUGBUG / TODO: this assumes dimension values are ordered *after* their arrays
            values.append(dim_names_to_values[arg.name])
        else:
            # everything else is known size or a pointer
            if arg.is_constant_shaped:
                arg.total_element_count = int(arg.total_element_count)
                arg.shape = list(map(int, arg.shape))
                if not hasattr(arg, 'numpy_strides'):
                    arg.numpy_strides = list(map(lambda x: x * arg.element_num_bytes, arg.shape[1:] + [1]))

            if arg.usage != hat_file.UsageType.Output:
                arg_data = _gen_random_data(arg.numpy_dtype, arg.shape)
                values.append(ArgValue(arg, arg_data))
            else:
                values.append(ArgValue(arg))

    # collect the dimension ArgValues for each output runtime_array ArgValue
    for value in values:
        if value.arg_info.usage == hat_file.UsageType.Output and not value.arg_info.is_constant_shaped:
            dim_values = [values[i] for i in get_dimension_arg_indices(value.arg_info, arguments)]
            value.dim_values = dim_values

    return values