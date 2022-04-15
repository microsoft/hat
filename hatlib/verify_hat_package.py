#!/usr/bin/env python3

import argparse

from . import hat


def verify_hat_package(hat_path):
    _, funcs = hat.load(hat_path)
    inputs = hat.generate_input_sets_for_hat_file(hat_path)
    for name, fn in funcs.items():
        print(f"\n{'*' * 10}\n")
        
        print(f"[*] Verifying function {name} --")
        func_inputs = inputs[name]

        print("[*] Inputs before function call:")
        for i, func_input in enumerate(func_inputs):
            print(f"[*]\tInput {i}: {','.join(map(str, func_input.ravel()[:32]))}")

        try:
        
            time = fn(*inputs[name])

        except RuntimeError as e:
            print(f"[!] Error while running {name}: {e}")
            continue
        
        print("Inputs after function call:")
        for i, func_input in enumerate(func_inputs):
            print(f"[*]\tInput {i}: {','.join(map(str, func_input.ravel()[:32]))}")

        if time:
            print(f"[*] Function execution time: {time:4f}ms")
        
        del inputs[name]

    else:
        print(f"\n{'*' * 10}\n")


def main():
    arg_parser = argparse.ArgumentParser(
        description=(
            "Executes every available function in the hat package with randomized inputs. Meant for quick verification.\n"
            "Example:\n"
            "    hatlib.verify_hat_package <hat_path>\n"
        )
    )

    arg_parser.add_argument("hat_path", help="Path to the HAT file", default=None)

    args = vars(arg_parser.parse_args())
    verify_hat_package(args["hat_path"])


if __name__ == "__main__":
    main()
