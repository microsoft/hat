#!/usr/bin/env python3

"""Converts a statically-linked HAT package into a Dynamically-linked HAT package

HAT packages come in two varieties: statically-linked and dynamically-linked. A statically-linked
HAT package contains a binary file with the extension '.obj', '.a', or `.lib`. A dynamically-linked
HAT package contains a binary file with the extension '.dll' or '.so'. This tool converts a
statically-linked HAT package into a dynamically-linked HAT package.

To use the tool, point it to the '.hat' file associated with the statically-linked package (the 
'.hat' file knows where to find the associated binary file), and provide a filename for the new 
'.hat' file.

Dependencies on Windows (add these to your console by running 'vcvarsall.bat x64', avaliable with
Microsoft Visual Studio):
* the cl.exe command-line compiler, available with Microsoft Visual Studio  
* the link.exe linker, available with Microsoft Visual Studio

Dependencies on Linux:
* the g++ command-line compiler
"""

import sys
import os
import argparse
import toml
import shutil

def get_platform():
    """Returns the current platform: Linux, Windows, or OS X"""
    
    if sys.platform.startswith("linux"):
        return "Linux"
    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "darwin":
        return "OS X"

    sys.exit(f"ERROR: Unsupported operating system: {sys.platform}")


def linux_create_dynamic_package(input_hat_binary_path, output_hat_path, hat_description):
    """Creates a dynamic HAT (.so) from a static HAT (.obj) on a Linux platform"""
    # Confirm that this is a static hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension not in [".o", ".a"]:
        sys.exit(f"ERROR: Expected input library to have extension .o or .a, but received {input_hat_binary_path} instead")

    # create new HAT binary
    prefix, _ = os.path.splitext(output_hat_path)
    output_hat_binary_path = prefix + ".so"
    os.system(f"g++ -shared -fPIC -o {output_hat_binary_path} {input_hat_binary_path}")

    # create new HAT file
    hat_description["dependencies"]["link_target"] = os.path.basename(output_hat_binary_path)
    with open(output_hat_path, "w") as f:
        toml.dump(hat_description, f)


def windows_create_dynamic_package(input_hat_binary_path, output_hat_path, hat_description):
    """Creates a Windows dynamic HAT package (.dll) from a static HAT package (.obj/.lib)"""
    # Confirm that this is a static hat library
    _, extension = os.path.splitext(input_hat_binary_path)
    if extension not in [".obj", ".lib"]:
        sys.exit(f"ERROR: Expected input library to have extension .obj or .lib, but received {input_hat_binary_path} instead")

    # Create all file in a directory named build
    if not os.path.exists("build"):
        os.mkdir("build")

    cwd = os.getcwd()
    try:
        os.chdir("build")
        
        # Create a C source file for the DLL entry point and compile in into an obj
        if not os.path.exists("dllmain.cpp"):
            with open("dllmain.cpp", "w") as f:
                f.write("#include <windows.h>\n")
                f.write("BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }\n")
        os.system("cl /Fodllmain.obj /c dllmain.cpp")

        # create the new HAT binary dll
        prefix, _ = os.path.splitext(output_hat_path)
        output_hat_binary_path = prefix + ".dll"

        function_descriptions = hat_description["functions"]
        function_names = list(function_descriptions.keys())
        linker_command_line = "link -dll -FORCE:MULTIPLE -EXPORT:{} -out:out.dll dllmain.obj {}".format(" -EXPORT:".join(function_names), input_hat_binary_path)
        os.system(linker_command_line)
        shutil.copyfile("out.dll", output_hat_binary_path)

        # create new HAT file
        hat_description["dependencies"]["link_target"] = os.path.basename(output_hat_binary_path)
        with open("out.hat", "w") as f:
            toml.dump(hat_description, f)
        shutil.copyfile("out.hat", output_hat_path)
    finally:
        os.chdir(cwd) # restore the current working directory

def parse_args():
    """Parses and checks the command line arguments"""
    parser = argparse.ArgumentParser(description="Creates a dynamically-linked HAT package from a statically-linked HAT package. Example:"
        "    python hat_to_dynamic.py input.hat output.hat")

    parser.add_argument("input_hat_path", type=str, help="Path to the existing HAT file, which represents a statically-linked HAT package")
    parser.add_argument("output_hat_path", type=str, help="Path to the new HAT file, which will represent a dynamically-linked HAT package")
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


def create_dynamic_package(input_hat_path, output_hat_path):
    platform = get_platform()

    # load the function decscriptions and the library path from the hat file
    input_hat_path = os.path.abspath(input_hat_path)
    hat_description = toml.load(input_hat_path)

    # get the absolute path to the input binary
    input_hat_binary_filename = hat_description["dependencies"]["link_target"]
    input_hat_binary_path = os.path.join(os.path.dirname(input_hat_path), input_hat_binary_filename)

    # create the dynamic package
    output_hat_path = os.path.abspath(output_hat_path)
    if platform == "Windows":
        windows_create_dynamic_package(input_hat_binary_path, output_hat_path, hat_description)
    elif platform in ["Linux", "OS X"]:
        linux_create_dynamic_package(input_hat_binary_path, output_hat_path, hat_description)


def main():
    args = parse_args()
    create_dynamic_package(args.input_hat_path, args.output_hat_path)


if __name__ == "__main__":
    main()
