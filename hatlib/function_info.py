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
            return args # pass-through

        if any(not isinstance(value, np.ndarray) for value in args):
            return args # pass-through

        # len(values) < len(self.arguments) and we have numpy arrays
        # do a best effort parameter expansion based on hat metadata
        i_value = 0
        expanded_args = [None] * len(self.arguments)
        # maps argument names to indices
        names_to_indices = {info.name: i for i, info in enumerate(self.arguments)}
        for i, (hat_desc, info) in enumerate(zip(self.desc.arguments, self.arguments)):
            array = args[i_value]
            if hat_desc.logical_type == hat_file.ParameterType.RuntimeArray:
                if isinstance(array, np.ndarray):
                    if hat_desc.usage == hat_file.UsageType.Output:
                        # replace the ndarray with a pointer
                        expanded_args[i] = ArgValue(info)
                        expanded_args[i].dim_values = []
                    else:
                        expanded_args[i] = array

                    # get the shape of the ndarray and use the values to expand the dimension args
                    for dim_name, dim_val in zip(info.shape, array.shape):
                        if integer_like(dim_name):
                            assert int(dim_name) == int(dim_val)
                            continue  # constant dimension

                        # dynamic dimension, initialize a dimension ArgValue at its index (and value if input)
                        i_dim = names_to_indices[dim_name]
                        dim_arg_info = self.arguments[i_dim]
                        dim_hat_desc = self.desc.arguments[i_dim]
                        assert dim_hat_desc.logical_type == hat_file.ParameterType.Element

                        if hat_desc.usage == hat_file.UsageType.Output:
                            assert dim_hat_desc.usage == hat_file.UsageType.Output
                            expanded_args[i_dim] = ArgValue(dim_arg_info)
                            # add a cross reference so that we can resolve shapes for the output array
                            expanded_args[i].dim_values.append(expanded_args[i_dim])
                        else:
                            assert dim_hat_desc.usage == hat_file.UsageType.Input
                            expanded_args[i_dim] = ArgValue(dim_arg_info, dim_val)
                i_value = i_value + 1
            elif hat_desc.logical_type == hat_file.ParameterType.AffineArray:
                expanded_args[i] = array
                i_value = i_value + 1
            # else hat_file.ParameterType.Element handled above

            if i_value == len(args):
                break  #  pre-processing complete

        return expanded_args

    def postprocess(self, expanded_args: List[Any], caller_args: List[Any]) -> None:
        if len(expanded_args) == len(caller_args):
            return # pass-through

        if any(not isinstance(value, np.ndarray) for value in caller_args):
            return # pass-through

        results = []

        # extract output arrays from the expanded args and override caller args
        for hat_desc, expanded_arg in zip(self.desc.arguments, expanded_args):
            if hat_desc.logical_type == hat_file.ParameterType.RuntimeArray and hat_desc.usage == hat_file.UsageType.Output:
                # resolve shape using the output dimensions
                shape = [d.value[0] for d in expanded_arg.dim_values]
                # override the output array argument for the caller
                results.append(np.ctypeslib.as_array(expanded_arg.value, shape))

        return results

    def verify(self, args: List[Any]):
        "Verifies that a list of argument values matches the function description"
        if len(args) != len(self.arguments):
            sys.exit(
                f"Error calling {self.name}(...): expected {len(self.arguments)} arguments but received {len(args)}"
            )

        for i, (info, value) in enumerate(zip(self.arguments, args)):
            try:
                if isinstance(value, np.ndarray) or isinstance(value, np.int64):
                    value = ArgValue(info, value)

                value.verify(info)
            except ValueError as v:
                sys.exit(f"Error calling {self.name}(...): argument {i} failed verification: {v}")

    def as_cargs(self, args: List[Any]):
        "Converts arguments to their C interfaces"
        arg_values = [
            ArgValue(info, value) if isinstance(value, np.ndarray) or isinstance(value, np.int64) else value
            for info, value in zip(self.arguments, args)
        ]

        return [value.as_carg() for value in arg_values]
