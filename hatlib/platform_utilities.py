#!/usr/bin/env python3

import io
import os
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Mapping
import enum

from .hat_file import OperatingSystem


class BUILD_TARGET(enum.Enum):
    STATIC_LIB = "_s"
    DYNAMIC_LIB = "_dyn"


class BUILD_TYPE(enum.Enum):
    DEBUG = "Debug"
    RELEASE = "Release"
    REL_WITH_DEB_INFO = "RelWithDebInfo"
    MIN_SIZE_REL = "MinSizeRel"


def _preprocess_command(command_to_run, shell):
    if get_platform() == OperatingSystem.Windows:
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
        with subprocess.Popen(command_to_run, close_fds=(get_platform() != OperatingSystem.Windows), shell=shell, stdout=stdout,
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


def get_lib_prefix():
    if get_platform() == OperatingSystem.Windows:
        return ""
    else:
        return "lib"


def get_lib_extension(shared=False):
    if get_platform() == OperatingSystem.Windows:
        return ".dll" if shared else ".lib"
    elif get_platform() == OperatingSystem.MacOS:
        return ".dylib" if shared else ".a"
    else:
        return ".so" if shared else ".a"


def normalize_path(path):
    return os.path.abspath(path).replace("\\", "/")


def get_file_extension(file_name):
    return os.path.splitext(file_name)[-1]


# NOTE: here we assume that all source files are already in "src_dir"
#       We might need to change that in the future.
def generate_and_run_cmake_file(
    target_name,
    src_dir,
    build_targets=[BUILD_TARGET.DYNAMIC_LIB],
    build_type="RelWithDebInfo",
    additional_include_filepaths=[],
    additional_link_filepaths=[],
    profile=False
) -> Mapping[BUILD_TARGET, str]:

    template_cmake_filename = os.path.join(src_dir, 'CMakeLists.txt.in')
    generated_cmake_filename = 'CMakeLists.txt'

    # Verify that the template cmake file exists
    if not os.path.exists(template_cmake_filename):
        raise FileNotFoundError(
            f"Template CMake file not found! Please add {template_cmake_filename} template file to current project directory and re-run."
        )

    src_files = []
    lib_files = [normalize_path(path) for path in additional_link_filepaths]
    include_paths = [normalize_path(path) for path in additional_include_filepaths]

    for src_file in os.listdir(src_dir):
        extension = get_file_extension(src_file)
        if extension in [".h", ".hat", ".c", ".cpp"]:
            src_files.append("${CMAKE_CURRENT_SOURCE_DIR}/" + src_file)
        elif extension in [".lib", ".a", ".obj", ".o"]:
            lib_files.append("${CMAKE_CURRENT_SOURCE_DIR}/" + src_file)

    with open(template_cmake_filename) as f:
        template_lines = f.readlines()

    with open(os.path.join(src_dir, generated_cmake_filename), mode="wt") as f:
        delimiter = "@"
        mappings = {
            "GEN_EXECUTABLE_NAME": target_name,
            "GEN_STATIC_LIB": f"{target_name}{BUILD_TARGET.STATIC_LIB.value}",
            "GEN_DYNAMIC_LIB": f"{target_name}{BUILD_TARGET.DYNAMIC_LIB.value}",
            "GENERATED_SOURCE_FILES": " ".join(src_files),
            "GEN_LINK_LIBS": " ".join(lib_files),
            "GEN_INCLUDE_PATHS": " ".join(include_paths)
        }
        for line in template_lines:
            for template in mappings:
                line = line.replace(f"{delimiter}{template}{delimiter}", mappings[template])
            f.write(line)

    build_dir = os.path.join(src_dir, f"build_{target_name}")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    if get_platform() == OperatingSystem.Windows:
        import vswhere
        vs_path = vswhere.get_latest_path()
        if not vs_path:
            raise RuntimeError("Could not find Visual Studio, please ensure that you have Visual Studio installed")

        run_command(
            f'cmake -G "Visual Studio {vswhere.get_latest_major_version()}" -Ax64 ..', working_directory=build_dir
        )
    else:
        profiling_flags = "-DPROFILING_MODE=ON" if profile else ""
        run_command(f'cmake -G Ninja -DCMAKE_BUILD_TYPE={build_type} {profiling_flags} ..', working_directory=build_dir)

    targets = {t: f"{target_name}{t.value}"
               for t in build_targets}

    #logging.info(f"running cmake --build . --config {build_type} --target {' '.join(targets.values())}")
    run_command(
        f"cmake --build . --config {build_type} --target {' '.join(targets.values())}", working_directory=build_dir
    )

    # Post-build cleaning
    config_build_dir = os.path.join(build_dir, build_type) if get_platform() == OperatingSystem.Windows else build_dir
    src_files = list(
        filter(
            lambda x: target_name in x and os.path.isfile(os.path.join(config_build_dir, x)),
            os.listdir(config_build_dir)
        )
    )

    # copy source files to source directory
    for file in src_files:
        shutil.copy(os.path.join(config_build_dir, file), src_dir)

    # clean the build directory except when debugging
    if build_type != BUILD_TYPE.DEBUG.value:
        shutil.rmtree(build_dir)

    target_filenames = {}
    for type, target in targets.items():
        if type == BUILD_TARGET.DYNAMIC_LIB:
            target_filenames[type] = get_lib_prefix() + target + get_lib_extension(shared=True)
        elif type == BUILD_TARGET.STATIC_LIB:
            target_filenames[type] = get_lib_prefix() + target + get_lib_extension(shared=False)
        else:
            target_filenames[type] = target

    return target_filenames