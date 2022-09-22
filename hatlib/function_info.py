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
            return args # bail

        if any(not isinstance(value, np.ndarray) for value in args):
            return args # bail

        # len(values) < len(self.arguments) and we have numpy arrays
        # do a best effort parameter expansion based on hat metadata
        value_idx = 0
        expanded_args = [None] * len(self.arguments)
        # maps argument names to indices
        names_to_indices = {info.name: i for i, info in enumerate(self.arguments)}
        for i, (hat_desc, info) in enumerate(zip(self.desc.arguments, self.arguments)):
            array = args[value_idx]
            if hat_desc.logical_type == hat_file.ParameterType.RuntimeArray:
                if isinstance(array, np.ndarray):
                    expanded_args[i] = array
                    for dim_name, dim_val in zip(info.shape, array.shape):
                        if integer_like(dim_name):
                            assert int(dim_name) == int(dim_val)
                            continue  # constant dimension

                        # dynamic dimension, initialize a dimension ArgValue at its index
                        dim_index = names_to_indices[dim_name]
                        dim_arg_info = self.arguments[dim_index]
                        dim_hat_desc = self.desc.arguments[dim_index]
                        assert dim_hat_desc.logical_type == hat_file.ParameterType.Element

                        if hat_desc.usage == hat_file.UsageType.Output:
                            assert dim_hat_desc.usage == hat_file.UsageType.Output
                            expanded_args[dim_index] = ArgValue(dim_arg_info)
                        else:
                            assert dim_hat_desc.usage == hat_file.UsageType.Input
                            expanded_args[dim_index] = ArgValue(dim_arg_info, dim_val)
                value_idx = value_idx + 1
            elif hat_desc.logical_type == hat_file.ParameterType.AffineArray:
                expanded_args[i] = array
                value_idx = value_idx + 1
            # else hat_file.ParameterType.Element handled above

            if value_idx == len(args):
                break  #  pre-processing complete

        return expanded_args

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
