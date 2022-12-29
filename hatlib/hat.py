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
import numpy as np

from typing import Callable, List, Tuple, Union
from functools import reduce

from . import hat_file
from . import hat_package
from .arg_value import generate_arg_values, ArgValue
from .arg_info import integer_like
from .function_info import FunctionInfo

PLACEHOLDER_SIZE = 128    # arbitrary, to be replaced with a better way to estimate size for runtime arrays


def generate_arg_sets_for_func(
    func: hat_file.Function,
    input_sets_minimum_size_MB: int = 0,
    num_additional: int = 0,
    dyn_func_shape_fn: Callable[[FunctionInfo], List[List[int]]] = None
):

    def default_dyn_func_shape_fn(func_info: FunctionInfo) -> List[List[int]]:
        return [[int(d) if integer_like(d) else PLACEHOLDER_SIZE for d in p.shape] for p in func_info.arguments
                if p.pointer_level == 1 and p.usage != hat_file.UsageType.Output]

    func_info = FunctionInfo(func)
    parameters = func_info.arguments

    # plug in values for non-constant dimensions in an attempt to estimate the minimum set size
    dim_names_to_values = {}
    if any(map(lambda p: not p.is_constant_shaped, func_info.arguments)):
        if dyn_func_shape_fn is None:
            dyn_func_shape_fn = default_dyn_func_shape_fn
        numerical_shapes = dyn_func_shape_fn(func_info)
        # TODO: We really need to be able to distinguish between args that are dimensions vs. just scalars
        shape_idx = 0
        param_idx = 0
        while param_idx < len(parameters):
            p = parameters[param_idx]
            if p.is_constant_shaped:
                if p.pointer_level and p.usage != hat_file.UsageType.Output:
                    shape_idx += 1

                param_idx += 1
                continue

            if p.usage == hat_file.UsageType.Output:
                param_idx += 1
                continue

            numerical_shape = numerical_shapes[shape_idx]

            # parameter is NOT constant shaped
            dyn_dimensions = filter(lambda idx_d: not integer_like(idx_d[1]), enumerate(p.shape))
            for dyn_dim_idx, dyn_dim in dyn_dimensions:
                if dyn_dim not in dim_names_to_values:
                    param_idx += 1
                    dim_names_to_values[dyn_dim] = ArgValue(parameters[param_idx], numerical_shape[dyn_dim_idx])

            param_idx += 1
            shape_idx += 1

    else:
        numerical_shapes = [p.shape if p.shape else ([int(p.size)] if p.size else [1]) for p in func.arguments]

    shapes_to_sizes = [reduce(lambda x, y: x * y, shape, 1) for shape in numerical_shapes]
    set_size = reduce(lambda x, y: x + y, map(lambda size, p: size * p.element_num_bytes, shapes_to_sizes, parameters))
    num_input_sets = (input_sets_minimum_size_MB * 1024 * 1024 // set_size) + 1 + num_additional

    arg_sets = [generate_arg_values(parameters, dim_names_to_values) for _ in range(num_input_sets)]

    return arg_sets[0] if len(arg_sets) == 1 else arg_sets


def generate_arg_sets_for_hat_file(hat_path):
    t = hat_file.HATFile.Deserialize(hat_path)
    return {func_name: generate_arg_sets_for_func(func_desc)
            for func_name, func_desc in t.function_map.items()}


def load(hat_path, try_dynamic_load=True) -> Tuple[hat_package.HATPackage, Union[hat_package.AttributeDict, None]]:
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
            function_dict = hat_package.hat_package_to_func_dict(pkg)
        except Exception as e:
            # TODO: Figure out how to communicate failure better
            print(e)

    return pkg, function_dict
