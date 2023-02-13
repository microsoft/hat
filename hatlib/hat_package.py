# Utility to parse and validate a HAT package

import ctypes
from collections import OrderedDict
import os
from typing import List

from .hat_file import HATFile, Function
from .function_info import FunctionInfo


class HATPackage:

    def __init__(self, hat_file_path):
        """A HAT Package is defined to be a HAT file and corresponding binary file, located in the same directory.
        The binary file is specified in the HAT file's link_target attribute.
        The same binary file can be referenced by many HAT files.
        Many HAT packages can exist in the same directory.
        An instance of HATPackage is created by giving HATPackage the file path to the .hat file."""
        self.name = os.path.basename(hat_file_path)
        self.hat_file_path = hat_file_path
        self.hat_file = HATFile.Deserialize(hat_file_path)

        self.link_target = self.hat_file.dependencies.link_target
        self.link_target_path = os.path.join(
            os.path.split(self.hat_file_path)[0], self.hat_file.dependencies.link_target
        )

        self.functions = self.hat_file.functions

    def __iter__(self):
        return iter(self.hat_file.functions)

    def get_functions(self) -> List[Function]:
        return self.hat_file.functions

    def get_functions_for_target(self, os: str, arch: str, required_extensions: list = []) -> List[Function]:
        all_functions = self.get_functions()

        def matches_target(hat_function):
            hat_file = hat_function.hat_file
            if hat_file.target.required.os != os or hat_file.target.required.cpu.architecture != arch:
                return False
            for required_ext in required_extensions:
                if required_ext not in hat_file.target.required.cpu.extensions:
                    return False
            return True

        return list(filter(matches_target, all_functions))

    def get_function(self, name: str) -> Function:
        for f in self.get_functions():
            if f.name == name:
                return f
        raise ModuleNotFoundError(f"Error: Cannot find {name} in {self.name}")

    def benchmark(
        self,
        functions: List[Function] = None,
        store_in_hat=False,
        batch_size=10,
        min_time_in_sec=10,
        input_sets_minimum_size_MB=50,
        device_id: int = 0,
        verbose: bool = False
    ) -> List["hatlib.Result"]:
        "Benchmarks the selected functions in the HAT package. If none are selected, all functions in the package are benchmarked."

        functions = functions if functions is not None else self.functions

        from benchmark_hat_package import run_benchmark

        return run_benchmark(
            self.hat_file_path,
            store_in_hat=store_in_hat,
            batch_size=batch_size,
            min_time_in_sec=min_time_in_sec,
            input_sets_minimum_size_MB=input_sets_minimum_size_MB,
            device_id=device_id,
            verbose=verbose,
            functions=[f.name for f in functions]
        )


class AttributeDict(OrderedDict):
    """ Dictionary that allows entries to be accessed like attributes
    """
    __getattr__ = OrderedDict.__getitem__

    @property
    def names(self):
        return list(self.keys())

    def __getitem__(self, key):
        for k, v in self.items():
            if k.startswith(key):
                return v
        return OrderedDict.__getitem__(key)


def _make_unprofiled_cpu_func(shared_lib: ctypes.CDLL, func: Function):
    func_info = FunctionInfo(func)
    fn = shared_lib[func_info.name]

    def f(*args):
        args_ = func_info.preprocess(args)

        # verify that the args match the description in the hat file
        func_info.verify(args_)

        # prepare the args to the hat package
        hat_args = func_info.as_cargs(args_)

        # call the function in the hat package
        fn(*hat_args)

        # get any results after post-processing
        return func_info.postprocess(args_, args)

    return f


def _make_callable_func(func_runtime: str, hat_dir_path: str, func: Function):
    if func_runtime == "CUDA":
        from . import cuda_loader
        return cuda_loader.create_loader_for_device_function(func, hat_dir_path)
    elif func_runtime == "ROCM":
        from . import rocm_loader
        return rocm_loader.create_loader_for_device_function(func, hat_dir_path)
    else:
        from . import host_loader
        return host_loader.create_loader_for_host_function(func, hat_dir_path)


def _load_pkg_binary_module(hat_pkg: HATPackage):
    shared_lib = None
    if os.path.isfile(hat_pkg.link_target_path):
        supported_extensions = [".dll", ".so", ".dylib"]
        _, extension = os.path.splitext(hat_pkg.link_target_path)

        if extension and extension in supported_extensions:
            hat_binary_path = os.path.abspath(hat_pkg.link_target_path)

            # load the hat_library:
            hat_library = ctypes.cdll.LoadLibrary(hat_binary_path) if extension else None
            shared_lib = hat_library

    return shared_lib


def hat_package_to_func_dict(hat_pkg: HATPackage, enable_native_profiling: bool) -> AttributeDict:
    try:
        try:
            from . import cuda_loader
        except ModuleNotFoundError:
            import cuda_loader
    except:
        CUDA_AVAILABLE = False
    else:
        CUDA_AVAILABLE = True

    try:
        try:
            from . import rocm_loader
        except ModuleNotFoundError:
            import rocm_loader
    except:
        ROCM_AVAILABLE = False
    else:
        ROCM_AVAILABLE = True

    NOTIFY_ABOUT_CUDA = not CUDA_AVAILABLE
    NOTIFY_ABOUT_ROCM = not ROCM_AVAILABLE

    # check that the HAT library has a supported file extension
    func_dict = AttributeDict()
    shared_lib = _load_pkg_binary_module(hat_pkg)
    hat_dir_path, _ = os.path.split(hat_pkg.hat_file_path)

    for func_name, func_desc in hat_pkg.hat_file.function_map.items():
        launches = func_desc.launches
        if not enable_native_profiling and not launches and shared_lib:
            func_dict[func_name] = _make_unprofiled_cpu_func(shared_lib, func_desc)
        else:
            device_func = hat_pkg.hat_file.device_function_map.get(launches)
            func_runtime = func_desc.runtime

            if device_func:
                if not func_runtime:
                    raise RuntimeError(f"Couldn't find runtime for loader: " + launches)

                # TODO: Generalize this concept to work so it's not CUDA/ROCM specific
                if func_runtime == "CUDA" and not CUDA_AVAILABLE:

                    # TODO: printing to stdout only makes sense in tool mode
                    if NOTIFY_ABOUT_CUDA:
                        print("CUDA functionality not available on this machine. Please install the cuda python modules")
                        NOTIFY_ABOUT_CUDA = False

                    continue

                elif func_runtime == "ROCM" and not ROCM_AVAILABLE:

                    # TODO: printing to stdout only makes sense in tool mode
                    if NOTIFY_ABOUT_ROCM:
                        print("ROCm functionality not available on this machine. Please install ROCm 4.2 or higher")
                        NOTIFY_ABOUT_ROCM = False

                    continue
            else:
                device_func = hat_pkg.hat_file.function_map.get(func_name)

            func_dict[func_name] = _make_callable_func(
                func_runtime=func_runtime, hat_dir_path=hat_dir_path, func=device_func
            )

    return func_dict
