#!/usr/bin/env python3
"""Converts a statically-linked HAT package into a Dynamically-linked HAT package

HAT packages come in two varieties: statically-linked and dynamically-linked. A statically-linked
HAT package contains a binary file with the extension '.o', '.obj', '.a', or `.lib`. A dynamically-linked
HAT package contains a binary file with the extension '.dll' or '.so'. This tool converts a
statically-linked HAT package into a dynamically-linked HAT package.

To use the tool, point it to the '.hat' file associated with the statically-linked package (the
'.hat' file knows where to find the associated binary file), and provide a filename for the new
'.hat' file.

Dependencies on Windows:
* the cl.exe command-line compiler, available with Microsoft Visual Studio

Dependencies on Linux / macOS:
* the gcc command-line compiler
"""

import sys
import os
import argparse
import shutil
from secrets import token_hex

from .hat_file import HATFile, OperatingSystem
from .hat_package import HATPackage
from .platform_utilities import get_platform, ensure_compiler_in_path, run_command


def linux_create_dynamic_package(input_hat_path,
                                 input_hat_binary_path,
                                 output_hat_path,
                                 hat_file,
                                 quiet=True):
    """Creates a dynamic HAT (.so) from a static HAT (.o/.a) on a Linux/macOS platform"""
    # Confirm that this is a static hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension not in [".o", ".a"]:
        sys.exit(
            f"ERROR: Expected input library to have extension .o or .a, but received {input_hat_binary_path} instead"
        )

    # Create a C source file to resolve inline functions defined in the static HAT package
    include_path = os.path.dirname(input_hat_binary_path)
    inline_c_path = os.path.join(include_path, "inline.c")
    inline_obj_path = os.path.join(include_path, "inline.o")
    with open(inline_c_path, "w") as f:
        f.write(f"#include <{os.path.basename(input_hat_path)}>")
    # compile it separately so that we can suppress the warnings about the missing terminating ' character
    run_command(
        f'gcc -c -w -fPIC -o "{inline_obj_path}" -I"{include_path}" "{inline_c_path}"',
        quiet=quiet)

    # create new HAT binary
    prefix, _ = os.path.splitext(output_hat_path)
    # always create a new so file (avoids cases where so file is already loaded)
    suffix = token_hex(4)
    output_hat_binary_path = f"{prefix}_{suffix}.so"
    libraries = " ".join(
        [d.target_file for d in hat_file.dependencies.dynamic])
    run_command(
        f'gcc -shared -fPIC -o "{output_hat_binary_path}" "{inline_obj_path}" "{input_hat_binary_path}" {libraries}',
        quiet=quiet)

    # create new HAT file
    # previous dependencies are now part of the binary
    hat_file.dependencies.dynamic = []
    hat_file.dependencies.link_target = os.path.basename(
        output_hat_binary_path)
    hat_file.Serialize(output_hat_path)

    return HATPackage(output_hat_path)


def windows_create_dynamic_package(input_hat_path,
                                   input_hat_binary_path,
                                   output_hat_path,
                                   hat_file,
                                   quiet=True):
    """Creates a Windows dynamic HAT package (.dll) from a static HAT package (.obj/.lib)"""

    # Confirm that this is a static hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension not in [".obj", ".lib"]:
        sys.exit(
            f"ERROR: Expected input library to have extension .obj or .lib, but received {input_hat_binary_path} instead"
        )

    # Create all file in a directory named build
    if not os.path.exists("build"):
        os.mkdir("build")

    cwd = os.getcwd()
    try:
        os.chdir("build")

        # Create a C source file for the DLL entry point and compile in into an obj
        with open("dllmain.cpp", "w") as f:
            f.write("#include <windows.h>\n")
            # Resolve inline functions defined in the static HAT package
            f.write("#include <{}>\n".format(os.path.basename(input_hat_path)))
            f.write(
                "BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }\n"
            )
        run_command(
            f'cl.exe /nologo /I"{os.path.dirname(input_hat_path)}" /Fodllmain.obj /c dllmain.cpp',
            quiet=quiet)

        # create the new HAT binary dll
        # always create a new dll (avoids case where dll is already loaded)
        suffix = token_hex(4)
        prefix, _ = os.path.splitext(output_hat_path)
        output_hat_binary_path = f"{prefix}_{suffix}.dll"

        function_descriptions = hat_file.functions
        function_names = [f.name for f in function_descriptions]
        exports = " /EXPORT:".join(function_names)

        libraries = " ".join(
            [d.target_file for d in hat_file.dependencies.dynamic])
        linker_command_line = f'link.exe /NOLOGO /DLL /FORCE:MULTIPLE /EXPORT:{exports} /OUT:out.dll dllmain.obj "{input_hat_binary_path}" {libraries}'
        run_command(linker_command_line, quiet=quiet)
        shutil.copyfile("out.dll", output_hat_binary_path)

        # create new HAT file
        # previous dependencies are now part of the binary
        hat_file.dependencies.dynamic = []
        hat_file.dependencies.link_target = os.path.basename(
            output_hat_binary_path)
        hat_file.Serialize("out.hat")
        shutil.copyfile("out.hat", output_hat_path)
    finally:
        os.chdir(cwd)  # restore the current working directory

    return HATPackage(output_hat_path)


def parse_args():
    """Parses and checks the command line arguments"""
    parser = argparse.ArgumentParser(
        description=
        "Creates a dynamically-linked HAT package from a statically-linked HAT package.\n"
        "Example:\n"
        "    hatlib.hat_to_dynamic input.hat output.hat\n")

    parser.add_argument(
        "input_hat_path",
        type=str,
        help=
        "Path to the existing HAT file, which represents a statically-linked HAT package"
    )
    parser.add_argument(
        "output_hat_path",
        type=str,
        help=
        "Path to the new HAT file, which will represent a dynamically-linked HAT package"
    )
    parser.add_argument('-v',
                        "--verbose",
                        action='store_true',
                        help="Enable verbose output")
    args = parser.parse_args()

    # check args
    if not os.path.exists(args.input_hat_path):
        sys.exit(f"ERROR: File {args.input_hat_path} not found")

    if os.path.abspath(args.input_hat_path) == os.path.abspath(
            args.output_hat_path):
        sys.exit("ERROR: Output file must be different from input file")

    _, extension = os.path.splitext(args.input_hat_path)
    if extension != ".hat":
        sys.exit(
            f"ERROR: Expected input file to have extension .hat, but received {extension} instead"
        )

    _, extension = os.path.splitext(args.output_hat_path)
    if extension != ".hat":
        sys.exit(
            f"ERROR: Expected output file to have extension .hat, but received {extension} instead"
        )

    return args


def create_dynamic_package(input_hat_path, output_hat_path, quiet=True):
    platform = get_platform()
    ensure_compiler_in_path()

    # load the function decscriptions and the library path from the hat file
    input_hat_path = os.path.abspath(input_hat_path)
    hat_file = HATFile.Deserialize(input_hat_path)

    # get the absolute path to the input binary
    input_hat_binary_filename = hat_file.dependencies.link_target
    input_hat_binary_path = os.path.join(os.path.dirname(input_hat_path),
                                         input_hat_binary_filename)

    # create the dynamic package
    output_hat_path = os.path.abspath(output_hat_path)
    if platform == OperatingSystem.Windows:
        windows_create_dynamic_package(input_hat_path,
                                       input_hat_binary_path,
                                       output_hat_path,
                                       hat_file,
                                       quiet=quiet)
    elif platform in [OperatingSystem.Linux, OperatingSystem.MacOS]:
        linux_create_dynamic_package(input_hat_path,
                                     input_hat_binary_path,
                                     output_hat_path,
                                     hat_file,
                                     quiet=quiet)


def main():
    args = parse_args()
    create_dynamic_package(args.input_hat_path,
                           args.output_hat_path,
                           quiet=not args.verbose)


if __name__ == "__main__":
    main()
