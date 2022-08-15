import numpy as np
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

    def generate_arg_values(self):
        "Generate argument values from argument descriptions"
        values = [ArgValue(a) for a in self.arguments]

        # resolve the reference values for dimensions by looking up the shape symbol names
        for value in values:
            if type(value.arg_info.total_byte_size) == str:
                dim_values = []
                for sym_name in value.arg_info.shape:
                    # Note: only support all str shapes, not a mixture of str/int
                    dim = list(filter(lambda v: v.arg_info.name == sym_name, values))
                    if not dim:
                        raise RuntimeError(
                            f"{sym_name} is not an argument, cannot resolve shape for {value.arg_info.name}"
                        )    # likely an invalid HAT file
                    dim_values.append(dim[0])
                if dim_values:
                    value.set_dimension_values(dim_values)
        
        return values
