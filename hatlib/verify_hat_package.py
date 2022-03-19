#!/usr/bin/env python3

import argparse
from ast import arg
import sys

if __package__:
    from .hat import load, generate_input_sets_for_hat_file
else:
    from hat import load, generate_input_sets_for_hat_file


def verify_hat_package(hat_path):
    funcs = load(hat_path)
    inputs = generate_input_sets_for_hat_file(hat_path)
    for name, fn in funcs.items():
        fn(inputs[name])


def main():
    arg_parser = argparse.ArgumentParser(
        description="Executes every available function in the hat package \
            with randomized inputs. Meant for quick verification.\n"
        "Example:\n"
        "    hatlib.verify_hat_package <hat_path>\n")

    arg_parser.add_argument("hat_path",
                            help="Path to the HAT file",
                            default=None)

    args = vars(arg_parser.parse_args())
    verify_hat_package(args["hat_path"])


if __name__ == "__main__":
    main()
