from typing import Any, List
from ctypes import byref
import numpy as np

from .arg_info import ArgInfo


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
                # confirm that the arg shape is correct (numpy represents shapes as tuples)
                if tuple(desc.shape) != self.value.shape:
                    raise ValueError(
                        f"expected argument to have shape={desc.shape} but received shape={self.value.shape}"
                    )

                # confirm that the arg strides are correct (numpy represents strides as tuples)
                if tuple(desc.numpy_strides) != self.value.strides:
                    raise ValueError(
                        f"expected argument to have strides={desc.numpy_strides} but received strides={self.value.strides}"
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
                        shape = [d.value[0] for d in self.dim_values]  # stored as single-element ndarrays
                        s = repr(np.ctypeslib.as_array(self.value, shape))
                    else:
                        s = repr(self.value.contents)
                except Exception as e:
                    if e.args[0].startswith("NULL pointer"):
                        s = f"{repr(self.value)} nullptr"
                    else:
                        raise(e)
                return s
        else:
            return repr(self.value)

    def __del__(self):
        if self.pointer_level == 2:
            pass    # TODO - free the pointer, presumably calling a symbol passed into this ArgValue
