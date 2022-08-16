#!/usr/bin/env python3

import argparse

from . import hat


def verify_hat_package(hat_path):
    _, funcs = hat.load(hat_path)
    args = hat.generate_arg_sets_for_hat_file(hat_path)
    for name, fn in funcs.items():
        print(f"\n{'*' * 10}\n")

        print(f"[*] Verifying function {name} --")
        func_args = args[name]

        print("[*] Args before function call:")
        for i, func_arg in enumerate(func_args):
            print(f"[*]\tArg {i}: {func_arg}")

        try:
            time = fn(*args[name])

        except RuntimeError as e:
            print(f"[!] Error while running {name}: {e}")
            continue

        print("Args after function call:")
        for i, func_arg in enumerate(func_args):
            print(f"[*]\tArg {i}: {func_arg}")

        if time:
            print(f"[*] Function execution time: {time:4f}ms")

        del args[name]

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
