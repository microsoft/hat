from ctypes import byref
import numpy as np

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
            elif self.pointer_level == 2:
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

    def verify(self, desc):
        "Verifies that this argument matches an argument description"
        if desc.pointer_level == 1:
            if not isinstance(self.value, np.ndarray):
                raise ValueError(f"expected argument to be <class 'numpy.ndarray'> but received {type(self.value)}")

            if desc.numpy_dtype != self.value.dtype:
                raise ValueError(f"expected argument to have dtype={desc.numpy_dtype} but received dtype={self.value.dtype}")

            # confirm that the arg shape is correct (numpy represents shapes as tuples)
            if tuple(desc.numpy_shape) != self.value.shape:
                raise ValueError(f"expected argument to have shape={desc.numpy_shape} but received shape={self.value.shape}")

            # confirm that the arg strides are correct (numpy represents strides as tuples)
            if tuple(desc.numpy_strides) != self.value.strides:
                raise ValueError(f"expected argument to have strides={desc.numpy_strides} but received strides={self.value.strides}")
        else:
            pass # TODO

    def __repr__(self):
        if isinstance(self.value, np.ndarray):
            return ",".join(map(str, self.value.ravel()[:32]))
        else:
            try:
                s = repr(self.value.contents)
            except: # NULL pointer
                s = repr(self.value)
            finally:
                return s

    def __del__(self):
        if self.pointer_level == 2:
            pass # TODO - free the pointer
