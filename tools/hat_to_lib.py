#!/usr/bin/env python3

"""Converts a HAT package with .obj/.o into a HAT package with a .lib/.a

A statically-linked HAT package contains a binary file with the extension '.o', '.obj', '.a', or `.lib`. 

This tool converts a statically-linked HAT package with an '.o' or '.obj' binary file into another
statically-linked HAT package with a '.a' or '.lib' binary file.

To use the tool, point it to the '.hat' file associated with the statically-linked package (the 
'.hat' file knows where to find the associated binary file), and provide a filename for the new 
'.hat' file.

Dependencies on Windows:
* the cl.exe command-line compiler, available with Microsoft Visual Studio  

Dependencies on Linux / macOS:
* the gcc command-line compiler
"""

from lib2to3.pgen2.token import OP
import sys
import os
import argparse
import shutil

if __package__:
    from .hat_file import HATFile, OperatingSystem
    from .platform_utilities import get_platform, ensure_compiler_in_path
else:
    from hat_file import HATFile, OperatingSystem
    from platform_utilities import get_platform, ensure_compiler_in_path


def linux_create_static_package(input_hat_path, input_hat_binary_path, output_hat_path, hat_file):
    """Creates a static HAT (.a) from a static HAT (.o) on a Linux/macOS platform"""
    # Confirm that this is a static .o hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension != ".o":
        sys.exit(f"ERROR: Expected input library to have extension .o, but received {input_hat_binary_path} instead")

    # create new HAT binary
    prefix, _ = os.path.splitext(output_hat_path)
    output_hat_binary_path = f"{prefix}.a"
    libraries = " ".join([d.target_file for d in hat_file.dependencies.dynamic])
    os.system(f'ar rcs "{output_hat_binary_path}" "{input_hat_binary_path}" {libraries}')

    # create new HAT file
    hat_file.dependencies.dynamic = [] # previous dependencies are now part of the binary
    hat_file.dependencies.link_target = os.path.basename(output_hat_binary_path)
    hat_file.Serialize(output_hat_path)


def windows_create_static_package(input_hat_path, input_hat_binary_path, output_hat_path, hat_file):
    """Creates a Windows static .lib HAT package from a static .obj HAT package"""

    # Confirm that this is a static hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension != ".obj":
        sys.exit(f"ERROR: Expected input library to have extension .obj, but received {input_hat_binary_path} instead")

    # Create all file in a directory named build
    if not os.path.exists("build"):
        os.mkdir("build")

    cwd = os.getcwd()
    try:
        os.chdir("build")
        
        # create the new HAT binary lib
        prefix, _ = os.path.splitext(output_hat_path)
        output_hat_binary_path = f"{prefix}.lib"

        # presume /DEF is not needed because the exports will be part of the HAT file
        libraries = " ".join([d.target_file for d in hat_file.dependencies.dynamic])
        archiver_command_line = f'lib.exe /NOLOGO /OUT:out.lib "{input_hat_binary_path}" {libraries}'
        os.system(archiver_command_line)
        shutil.copyfile("out.lib", output_hat_binary_path)

        # create new HAT file
        hat_file.dependencies.dynamic = [] # previous dependencies are now part of the binary
        hat_file.dependencies.link_target = os.path.basename(output_hat_binary_path)
        hat_file.Serialize("out.hat")
        shutil.copyfile("out.hat", output_hat_path)
    finally:
        os.chdir(cwd) # restore the current working directory

def parse_args():
    """Parses and checks the command line arguments"""
    parser = argparse.ArgumentParser(description="Creates a statically-linked HAT package with a .lib/.a from a statically-linked HAT package with an .obj/.o.\n"
        "Example:\n"
        "    python hatlib.hat_to_lib input.hat output.hat")

    parser.add_argument("input_hat_path", type=str, help="Path to the existing HAT file, which represents a statically-linked HAT package with a .obj or .o binary file")
    parser.add_argument("output_hat_path", type=str, help="Path to the new HAT file, which will represent a statically-linked HAT package with a .lib or .a binary file")
    args = parser.parse_args()

    # check args
    if not os.path.exists(args.input_hat_path):
        sys.exit(f"ERROR: File {args.input_hat_path} not found")

    if os.path.abspath(args.input_hat_path) == os.path.abspath(args.output_hat_path):
        sys.exit("ERROR: Output file must be different from input file")

    _, extension = os.path.splitext(args.input_hat_path)
    if extension != ".hat":
        sys.exit(f"ERROR: Expected input file to have extension .hat, but received {extension} instead")

    _, extension = os.path.splitext(args.output_hat_path)
    if extension != ".hat":
        sys.exit(f"ERROR: Expected output file to have extension .hat, but received {extension} instead")

    return args


def create_static_package(input_hat_path, output_hat_path):
    platform = get_platform()
    ensure_compiler_in_path()

    # load the function decscriptions and the library path from the hat file
    input_hat_path = os.path.abspath(input_hat_path)
    hat_file = HATFile.Deserialize(input_hat_path)

    # get the absolute path to the input binary
    input_hat_binary_filename = hat_file.dependencies.link_target
    input_hat_binary_path = os.path.join(os.path.dirname(input_hat_path), input_hat_binary_filename)

    # create the static library package
    # TODO: prefer lld when available and support cross-compilation
    output_hat_path = os.path.abspath(output_hat_path)
    if platform == OperatingSystem.Windows:
        windows_create_static_package(input_hat_path, input_hat_binary_path, output_hat_path, hat_file)
    elif platform in [OperatingSystem.Linux, OperatingSystem.MacOS]:
        linux_create_static_package(input_hat_path, input_hat_binary_path, output_hat_path, hat_file)


def main():
    args = parse_args()
    create_static_package(args.input_hat_path, args.output_hat_path)


if __name__ == "__main__":
    main()
