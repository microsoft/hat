import sys
from dataclasses import dataclass, field
from typing import List

from .arg_info import ArgInfo
from .arg_value import ArgValue
from . import hat_file


@dataclass
class FunctionInfo:
    "Extracts necessary information from the description of a function in a hat file"
    desc: hat_file.Function
    args: List[ArgInfo] = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.name = self.desc.name
        self.args = list(map(ArgInfo, self.desc.arguments))

    def verify_args(self, values: List[ArgValue]):
        "Verifies that a list of argument values matches the function description"
        # check number of args
        if len(values) != len(self.args):
            sys.exit(f"Error calling {self.name}(...): expected {len(self.args)} arguments but received {len(args)}")

        # for each arg
        for i, (arg, value) in enumerate(zip(self.args, values)):
            try:
                value.verify(arg)
            except ValueError as v:
                sys.exit(f"Error calling {self.name}(...): argument {i} failed verification: {v}")
