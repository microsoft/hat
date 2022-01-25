#!/usr/bin/env python3

import os
import sys

if __package__:
    from .hat_file import OperatingSystem
else:
    from hat_file import OperatingSystem

def get_platform():
    """Returns the current platform: Linux, Windows, or OS X"""
    
    if sys.platform.startswith("linux"):
        return OperatingSystem.Linux
    if sys.platform == "win32":
        return OperatingSystem.Windows
    if sys.platform == "darwin":
        return OperatingSystem.MacOS

    sys.exit(f"ERROR: Unsupported operating system: {sys.platform}")


def linux_ensure_compiler_in_path():
    """Ensures that PATH contains the gcc compiler.
    Prompts the user if not found."""
    if os.system("gcc --version"): # returns 0 if found, !0 otherwise
        sys.exit(f'ERROR: Could not find gcc, please install gcc before continuing')


def windows_ensure_compiler_in_path():
    """Ensures that PATH contains the cl.exe compiler from Microsoft Visual Studio.
    Prompts the user if not found."""
    import vswhere
    vs_path = vswhere.get_latest_path()
    if not vs_path:
        sys.exit("ERROR: Could not find Visual Studio, please ensure that you have Visual Studio installed")

    # Check if cl.exe is in PATH
    if os.system("cl.exe"): # returns 0 if found, !0 otherwise
        vcvars_script_path = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvars64.bat")
        sys.exit(f'ERROR: Could not find cl.exe, please run "{vcvars_script_path}" (including quotes) to setup your command prompt')


def ensure_compiler_in_path():
    platform = get_platform()
    if platform == OperatingSystem.Windows:
        windows_ensure_compiler_in_path()
    else:
        linux_ensure_compiler_in_path()