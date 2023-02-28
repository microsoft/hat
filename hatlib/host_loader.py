import os
import numpy as np
from typing import List
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
    std::chrono::duration<double> elapsed_seconds = end - start;
    *timing += elapsed_seconds.count() * 1000.0; // Time in milliseconds
}
"""

class HostCallableFunc(CallableFunc):
    def __init__(self, func: Function, host_src_path: str) -> None:
        super().__init__()
        self.host_src_path = os.path.abspath(host_src_path)
        self.hat_func = func
        self.func_info = FunctionInfo(func)

    def init_runtime(self, benchmark: bool, device_id: int):
        # create the timer code
        src_dir = os.path.dirname(__file__)
        native_profiler_name = self.func_info.name + "_timer"
        native_profiler_prefix = os.path.join(src_dir, native_profiler_name)
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
            #build_type="Debug",
            additional_include_filepaths=[self.host_src_path],
            additional_link_filepaths=[os.path.join(os.path.dirname(self.hat_func.hat_file.path), static_lib)],
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


    def cleanup_runtime(self, benchmark: bool):
        if self.native_profiler_srcfile and os.path.exists(self.native_profiler_srcfile):
            os.remove(self.native_profiler_srcfile)

        if self.native_profiler_hatfile and os.path.exists(self.native_profiler_hatfile):
            os.remove(self.native_profiler_hatfile)

        cmake_file = os.path.join(os.path.dirname(__file__), 'CMakeLists.txt')
        if os.path.exists(cmake_file):
            os.remove(cmake_file)

        if self.target and get_platform() != OperatingSystem.Windows: # Windows won't let you remove the dll until the process is dead
            target_file = os.path.join(os.path.dirname(__file__), self.target)
            if os.path.exists(target_file):
                os.remove(target_file)


    def init_main(self, benchmark: bool, warmup_iters=0, device_id: int = 0, args=[]):
        self.func_info.verify(args)

        for _ in range(warmup_iters):
            self.timer_func(*args, self.timing_arg_val)

    def main(self, benchmark: bool, iters=1, batch_size=1, min_time_in_sec=0, args=[]) -> float:
        batch_timings: List[float] = []
        while True:
            for _ in range(batch_size):
                self.timing_arg_val.value = np.zeros((1,))

                for _ in range(iters):
                    self.timer_func(*args, self.timing_arg_val)

                batch_time = self.timing_arg_val.value
                batch_timings.append(batch_time)

            if sum(batch_timings) >= (min_time_in_sec * 1000):
                break

        return batch_timings

    def cleanup_main(self, benchmark: bool, args=[]):
        pass


def create_loader_for_host_function(host_func: Function, hat_dir_path: str):
    return HostCallableFunc(host_func, hat_dir_path)