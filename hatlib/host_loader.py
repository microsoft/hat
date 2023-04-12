import os
import numpy as np
from .callable_func import CallableFunc
from .hat_file import Function, HATFile, Declaration, Dependencies, CallingConventionType, Parameter, ParameterType, OperatingSystem, UsageType
from .hat import load
from .function_info import FunctionInfo
from .arg_info import ArgInfo
from .arg_value import generate_arg_values
from .platform_utilities import generate_and_run_cmake_file, get_platform

profiler_code = """
#undef TOML
#include "{src_include}"
#include <chrono>
#include <cstdio>

#ifdef _MSC_VER
#define DLL_EXPORT extern "C" __declspec( dllexport )
#else
#define DLL_EXPORT extern "C"
#endif

DLL_EXPORT void timer({intput_args_decl}, double* timing)
{
    auto start = std::chrono::high_resolution_clock::now();
    {func_to_profile}
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ms = end - start;
    *timing += elapsed_ms.count(); 
}
"""

class HostCallableFunc(CallableFunc):
    def __init__(self, func: Function, host_src_path: str) -> None:
        super().__init__()
        self.host_src_path = os.path.abspath(host_src_path)
        self.hat_func = func
        self.func_info = FunctionInfo(func)

    def init_runtime(self, benchmark: bool, device_id: int, working_dir: str):
        # create the timer code
        src_dir = os.path.dirname(__file__)
        dest_dir = working_dir or os.getcwd()
        native_profiler_name = self.func_info.name + "_timer"
        native_profiler_prefix = os.path.join(dest_dir, native_profiler_name)
        self.native_profiler_srcfile = native_profiler_prefix + ".cpp"
        self.native_profiler_hatfile = native_profiler_prefix + ".hat"
        with open(self.native_profiler_srcfile, "w") as timer_file:
            timer_file.write(profiler_code
                .replace("{src_include}", self.hat_func.hat_file.name + ".hat")
                .replace("{intput_args_decl}", self.func_info.as_arg_type_decl())
                .replace("{func_to_profile}", f"{self.func_info.name}({self.func_info.as_arg_names()});"))

        # Build the timer code
        static_lib = self.hat_func.hat_file.dependencies.auxiliary["static"]

        target_binaries = generate_and_run_cmake_file(
            target_name=native_profiler_name,
            src_dir=src_dir,
            dest_dir=dest_dir,
            #build_type="Debug",
            additional_include_filepaths=[self.host_src_path],
            additional_link_filepaths=[os.path.join(os.path.dirname(self.hat_func.hat_file.path), static_lib)],
            additional_src_filepaths=[self.native_profiler_srcfile],
            profile=benchmark
        )

        assert len(target_binaries.items()) == 1
        _, self.target = list(target_binaries.items())[0]
        timer_header_code = f"void timer({self.func_info.as_arg_type_decl()}, double*);"
        timing_param = Parameter(
                                logical_type=ParameterType.Element,
                                declared_type="double*",
                                element_type="double",
                                usage=UsageType.InputOutput
                                )
        timing_arg_info = ArgInfo(timing_param)
        self.timing_arg_val = generate_arg_values([timing_arg_info])[0]
        timer_hat_file = HATFile(name=native_profiler_prefix,
                                    functions=[Function(
                                        arguments=self.hat_func.arguments + [timing_param],
                                        calling_convention=CallingConventionType.StdCall,
                                        description=f"Native profiler",
                                        name="timer",
                                        return_info=Parameter.void()
                                    )],
                                    path=self.native_profiler_hatfile,
                                    declaration=Declaration(code=timer_header_code),
                                    dependencies=Dependencies(link_target=self.target))
        timer_hat_file.Serialize()
        self.timer_hat_pkg, timer_func_dict = load(self.native_profiler_hatfile)
        self.timer_func = timer_func_dict["timer"]


    def cleanup_runtime(self, benchmark: bool, working_dir: str):
        working_dir = working_dir or os.getcwd()
        if self.native_profiler_srcfile and os.path.exists(self.native_profiler_srcfile):
            os.remove(self.native_profiler_srcfile)

        if self.native_profiler_hatfile and os.path.exists(self.native_profiler_hatfile):
            os.remove(self.native_profiler_hatfile)

        cmake_file = os.path.join(working_dir, 'CMakeLists.txt')
        if os.path.exists(cmake_file):
            os.remove(cmake_file)

        if self.target and get_platform() != OperatingSystem.Windows: # Windows won't let you remove the dll until the process is dead
            target_file = os.path.join(working_dir, self.target)
            if os.path.exists(target_file):
                os.remove(target_file)


    def init_batch(self, benchmark: bool, warmup_iters=0, device_id: int = 0, args=[]):
        self.func_info.verify(args[0] if benchmark else args)

        for _ in range(warmup_iters):
            if benchmark:
                for arg in args:
                    self.timer_func(*arg, self.timing_arg_val)
            else:
                self.timer_func(*args, self.timing_arg_val)

    def run_batch(self, benchmark: bool, iters, args=[]) -> float:
        i_max = len(args) if benchmark else 1
        self.timing_arg_val.value = np.zeros((1,))

        for iter in range(iters):
            func_args = args[iter % i_max] if benchmark else args
            self.timer_func(*func_args, self.timing_arg_val)

        return float(self.timing_arg_val.value)

    def cleanup_batch(self, benchmark: bool, args=[]):
        pass

    def should_flush_cache(self) -> bool:
        return True


def create_loader_for_host_function(host_func: Function, hat_dir_path: str):
    return HostCallableFunc(host_func, hat_dir_path)