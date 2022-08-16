import numpy as np
import random
import sys
from dataclasses import dataclass, field
from typing import Any, List

from .arg_info import ArgInfo
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

    def verify(self, args: List[Any]):
        "Verifies that a list of argument values matches the function description"
        if len(args) != len(self.arguments):
            sys.exit(
                f"Error calling {self.name}(...): expected {len(self.arguments)} arguments but received {len(args)}"
            )

        for i, (info, value) in enumerate(zip(self.arguments, args)):
            try:
                if isinstance(value, np.ndarray):
                    value = ArgValue(info, value)

                value.verify(info)
            except ValueError as v:
                sys.exit(f"Error calling {self.name}(...): argument {i} failed verification: {v}")

    def as_cargs(self, args: List[Any]):
        "Converts arguments to their C interfaces"
        arg_values = [
            ArgValue(info, value) if isinstance(value, np.ndarray) else value
            for info, value in zip(self.arguments, args)
        ]

        return [value.as_carg() for value in arg_values]

    def _get_dimension_arg_indices(self, array_arg: ArgInfo) -> List[int]:
        # Returns the dimension argument indices in shape order for an array argument
        indices = []
        for sym_name in array_arg.shape:
            for i, info in enumerate(self.arguments):
                if info.name == sym_name:    # limitation: only string shapes are supported
                    indices.append(i)
                    break
            else:
                # not found
                raise RuntimeError(f"{sym_name} is not an argument to the function")    # likely an invalid HAT file
        return indices

    def generate_arg_values(self):
        "Generate argument values from argument descriptions"

        def generate_dim_value():
            return random.choice([128, 256, 1234])

        dim_names_to_values = {}
        values = []

        for arg in self.arguments:
            if arg.usage == hat_file.UsageType.Input and not arg.constant_sized:
                # runtime_array: input
                dim_args = [self.arguments[i] for i in self._get_dimension_arg_indices(arg)]

                # assign generated shape values to the corresponding dimension arguments
                shape = []
                for d in dim_args:
                    if d.name not in dim_names_to_values:
                        shape.append(generate_dim_value())
                        dim_names_to_values[d.name] = ArgValue(d, shape[-1])
                    else:
                        shape.append(dim_names_to_values[d.name].value)

                # generate array inputs using the generated shape
                runtime_array_inputs = np.random.random(tuple(shape)).astype(arg.numpy_dtype)
                values.append(ArgValue(arg, runtime_array_inputs))

            elif arg.name in dim_names_to_values:
                # element: input (already has a value generated as a dimension)
                values.append(dim_names_to_values[arg.name])
            else:
                # affine_arrays and input elements not used as a dimension
                values.append(ArgValue(arg))

        for value in values:
            if value.arg_info.usage == hat_file.UsageType.Output and not value.arg_info.constant_sized:
                # runtime_array: output
                # find the corresponding output elements for its dimension
                dim_values = [values[i] for i in self._get_dimension_arg_indices(value.arg_info)]
                assert dim_values, f"Runtime array {value.arg_info.name} has no dimensions"
                value.dim_values = dim_values

        return values
