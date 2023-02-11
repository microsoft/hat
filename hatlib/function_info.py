import numpy as np
import sys
from dataclasses import dataclass, field
from typing import Any, List

from .arg_info import ArgInfo, integer_like
from .arg_value import ArgValue
from . import hat_file


@dataclass
class FunctionInfo:
    "Information about a HAT function"
    desc: hat_file.Function
    arguments: List[ArgInfo] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.name = self.desc.name
        self.arguments = list(map(ArgInfo, self.desc.arguments))

    def preprocess(self, args: List[Any]) -> List[ArgValue]:
        if len(args) >= len(self.arguments):
            return args  # pass-through

        if any(not isinstance(value, np.ndarray) for value in args):
            return args  # pass-through

        # len(values) < len(self.arguments) and we have numpy arrays
        # do a best effort parameter expansion based on hat metadata
        i_value = 0
        expanded_args = [None] * len(self.arguments)

        # determine if the caller is passing in all arrays as arguments (including outputs)
        # (useful for automation scenarios)
        num_array_args = len(
            list(
                filter(
                    lambda x: x.logical_type == hat_file.ParameterType.RuntimeArray
                    or x.logical_type == hat_file.ParameterType.AffineArray,
                    self.desc.arguments,
                )
            )
        )
        has_full_array_args = num_array_args == len(args)

        # maps argument names to indices
        names_to_indices = {info.name: i for i, info in enumerate(self.arguments)}
        for i, (hat_desc, info) in enumerate(zip(self.desc.arguments, self.arguments)):
            if hat_desc.logical_type == hat_file.ParameterType.RuntimeArray:
                if hat_desc.usage == hat_file.UsageType.Output:
                    # insert an output pointer for the C function
                    expanded_args[i] = ArgValue(info)
                    expanded_args[i].dim_values = []

                    if len(info.shape) and info.shape[0] == '':
                        info.shape = []

                    array_shape = info.shape
                    if has_full_array_args:  # skip over the output arg
                        i_value = i_value + 1
                else:
                    array = args[i_value]
                    if hat_desc.usage == hat_file.UsageType.InputOutput:
                        # TODO: support the first pass of the two-pass-alloc pattern where the caller
                        # passes NULL for the dynamic InputOutput arrays to determine the shapes
                        # to allocate. Currently we assume that the caller knows the shape through
                        # some out-of-band means (such as model shape inference)
                        assert array is not None, "two-pass-alloc NULL arrays are not yet supported"

                    expanded_args[i] = array
                    array_shape = array.shape
                    i_value = i_value + 1

                # expand the dimension args
                for dim_name, dim_val in zip(info.shape, array_shape):
                    if integer_like(dim_name):
                        assert int(dim_name) == int(dim_val)
                        if hat_desc.usage == hat_file.UsageType.Output:
                            # add the constant dimension to the array dim_values
                            expanded_args[i].dim_values.append(int(dim_val))
                        continue  # constant dimension

                    # dynamic dimension
                    # initialize a dimension ArgValue at its index (with value if is input)
                    i_dim = names_to_indices[dim_name]
                    dim_arg_info = self.arguments[i_dim]
                    dim_hat_desc = self.desc.arguments[i_dim]
                    assert dim_hat_desc.logical_type == hat_file.ParameterType.Element

                    # The two-pass alloc calling pattern:
                    # 1. call the function with NULL arrays (i.e. 1st pass) to compute the shape of the runtime array
                    # 2. allocate the runtime array with the computed shape
                    # 3. call the function again (i.e. 2nd pass) with the allocated runtime array
                    # The runtime array is therefore Input_Output, with Output dimensions
                    two_pass_alloc = hat_desc.usage == hat_file.UsageType.InputOutput \
                        and dim_hat_desc.usage == hat_file.UsageType.Output

                    if hat_desc.usage == hat_file.UsageType.Output:
                        assert dim_hat_desc.usage == hat_file.UsageType.Output
                        if expanded_args[i_dim] is None:  # arg not yet initialized
                            expanded_args[i_dim] = ArgValue(dim_arg_info)
                        # add a cross reference so that we can resolve shapes for the output array
                        # after the function is called
                        expanded_args[i].dim_values.append(expanded_args[i_dim])
                    elif two_pass_alloc:
                        if expanded_args[i_dim] is None:  # arg not yet initialized
                            expanded_args[i_dim] = ArgValue(dim_arg_info)
                        # a cross reference is not needed because we know the shapes in the 2nd pass
                    else:
                        if expanded_args[i_dim] is None:  # arg not yet initialized
                            expanded_args[i_dim] = ArgValue(dim_arg_info, dim_val)
            elif hat_desc.logical_type == hat_file.ParameterType.AffineArray:
                expanded_args[i] = args[i_value]
                i_value = i_value + 1
            # else hat_file.ParameterType.Element handled above

        if any(a is None for a in expanded_args):
            raise RuntimeError(
                f"Could not resolve some arguments for {self.name} (see arguments marked 'None'): {expanded_args}"
            )

        return expanded_args

    def postprocess(self, expanded_args: List[Any], caller_args: List[Any]) -> None:
        if len(expanded_args) == len(caller_args):
            return  # pass-through

        if any(not isinstance(value, np.ndarray) for value in caller_args):
            return  # pass-through

        results = []

        # extract output arrays from the expanded args and override caller args
        for hat_desc, expanded_arg in zip(self.desc.arguments, expanded_args):
            if (
                hat_desc.logical_type == hat_file.ParameterType.RuntimeArray
                and hat_desc.usage == hat_file.UsageType.Output
            ):
                # resolve shape using the output dimensions
                shape = [
                    d.value[0] if isinstance(d, ArgValue) else d
                    for d in expanded_arg.dim_values
                ]
                # override the output array argument for the caller
                results.append(np.ctypeslib.as_array(expanded_arg.value, shape))

        return results[0] if len(results) == 1 else results

    def verify(self, args: List[Any]):
        "Verifies that a list of argument values matches the function description"
        if len(args) != len(self.arguments):
            sys.exit(
                f"Error calling {self.name}(...): expected {len(self.arguments)} arguments but received {len(args)}"
            )

        for i, (info, value) in enumerate(zip(self.arguments, args)):
            try:
                if isinstance(value, np.ndarray) or issubclass(type(value), np.integer) or issubclass(type(value), np.floating):
                    value = ArgValue(info, value)

                value.verify(info)
            except ValueError as v:
                sys.exit(
                    f"Error calling {self.name}(...): argument {i} failed verification: {v}"
                )

    def as_cargs(self, args: List[Any]):
        "Converts arguments to their C interfaces"
        arg_values = [
            ArgValue(info, value)
            if isinstance(value, np.ndarray)
            or issubclass(type(value), np.integer)
            or issubclass(type(value), np.floating)
            else value
            for info, value in zip(self.arguments, args)
        ]

        return [value.as_carg() for value in arg_values]

    def as_arg_type_decl(self):
        return ", ".join([f"{arg.hat_declared_type} arg_{i}" for i, arg in enumerate(self.arguments)])

    def as_arg_names(self):
        return ", ".join([f"arg_{i}" for i in range(len(self.arguments))])
