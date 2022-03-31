#!/usr/bin/env python3

import io
import os
import platform
import shlex
import shutil
import subprocess
import sys

from .hat_file import OperatingSystem


def _preprocess_command(command_to_run, shell):
    if platform.system() == "Windows":
        return command_to_run
    elif type(command_to_run) == str and not shell:
        return shlex.split(command_to_run)
    elif type(command_to_run) == list and shell:
        return subprocess.list2cmdline(command_to_run)
    else:
        return command_to_run


def _dump_file_contents(iostream):
    if isinstance(iostream, io.TextIOBase):
        with open(iostream.name, "r") as f:
            print(f.name)
            print(f.read())


def run_command(
    command_to_run, working_directory=None, stdout=None, stderr=None, shell=False, pretend=False, quiet=True
):
    if not working_directory:
        working_directory = os.getcwd()

    if not quiet:
        print(f"\ncd {working_directory}")
        print(f"{command_to_run}\n")

    command_to_run = _preprocess_command(command_to_run, shell)

    if not pretend:
        with subprocess.Popen(command_to_run, close_fds=(platform.system() != "Windows"), shell=shell, stdout=stdout,
                              stderr=stderr, cwd=working_directory) as proc:

            proc.wait()
            if proc.returncode:
                _dump_file_contents(stderr)
                _dump_file_contents(stdout)
                raise subprocess.CalledProcessError(proc.returncode, command_to_run)


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
    """Ensures that PATH contains the $CXX/$CC (or gcc) compiler.
    Prompts the user if not found."""
    compiler = os.environ.get("CXX") or (os.environ.get("CC") or "gcc")
    if not shutil.which(compiler):
        sys.exit('ERROR: Could not find any valid C or C++ compiler, please install gcc before continuing')


def windows_ensure_compiler_in_path():
    """Ensures that PATH contains the cl.exe compiler from Microsoft Visual Studio.
    Prompts the user if not found."""
    import vswhere
    vs_path = vswhere.get_latest_path()
    if not vs_path:
        sys.exit("ERROR: Could not find Visual Studio, please ensure that you have Visual Studio installed")

    # Check if cl.exe is in PATH
    if not shutil.which("cl"):    # returns 0 if found, !0 otherwise
        vcvars_script_path = os.path.join(vs_path, r"VC\Auxiliary\Build\vcvars64.bat")
        sys.exit(
            f'ERROR: Could not find cl.exe, please run "{vcvars_script_path}" (including quotes) to setup your command prompt'
        )


def ensure_compiler_in_path():
    platform = get_platform()
    if platform == OperatingSystem.Windows:
        windows_ensure_compiler_in_path()
    else:
        linux_ensure_compiler_in_path()
