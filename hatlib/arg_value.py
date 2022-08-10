import numpy as np
from ctypes import byref

from .arg_info import ArgInfo

class ArgValue:
    "An argument instance, usable for calling a C function"
    def __init__(self, desc: ArgInfo):
        self.pointer_level = desc.pointer_level
        if self.pointer_level:
            self.ctypes_type = desc.ctypes_pointer_type
            if self.pointer_level == 1:
                # materialize an ndarray with random input values
                self.value = np.lib.stride_tricks.as_strided(
                    np.random.rand(desc.total_element_count).astype(desc.numpy_dtype),
                    shape=desc.numpy_shape,
                    strides=desc.numpy_strides)
            elif self.pointer_level  == 2:
                # materialize a pointer type
                self.value = self.ctypes_type()
            else:
                raise NotImplementedError("Pointer levels > 2 are not supported")
        else:
            raise NotImplementedError("Non-pointer types are not supported") # TODO

    def as_carg(self):
        "Return the C interface for this argument"
        if isinstance(self.value, np.ndarray):
            return self.value.ctypes.data_as(self.ctypes_type)
        else:
            return byref(self.value)

    def verify(self, desc: ArgInfo):
        "Verifies that this argument matches an argument description"
        if desc.pointer_level == 1:
            if not isinstance(self.value, np.ndarray):
                raise ValueError(f"Expected argument to be <class 'numpy.ndarray'> but received {type(self.value)}")

            if desc.numpy_dtype != self.value.dtype:
                raise ValueError(f"Expected argument to have dtype={desc.numpy_dtype} but received dtype={self.value.dtype}")

            # confirm that the arg shape is correct
            if desc.numpy_shape != self.value.shape:
                raise ValueError(f"Expected argument to have shape={desc.numpy_shape} but received shape={self.value.shape}")

            # confirm that the arg strides are correct
            if desc.numpy_strides != self.value.strides:
                raise ValueError(f"Expected argument to have strides={desc.numpy_strides} but received strides={self.value.strides}")
        else:
            pass # TODO

    def cleanup(self):
        if self.pointer_level == 2:
            pass # TODO - freel the pointer
