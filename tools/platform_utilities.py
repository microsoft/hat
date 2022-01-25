#!/usr/bin/env python3

import sys

def get_platform():
    """Returns the current platform: Linux, Windows, or OS X"""
    
    if sys.platform.startswith("linux"):
        return "Linux"
    if sys.platform == "win32":
        return "Windows"
    if sys.platform == "darwin":
        return "OS X"

    sys.exit(f"ERROR: Unsupported operating system: {sys.platform}")

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
