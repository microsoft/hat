#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import os
from typing import Callable, List
import numpy as np
import pandas as pd
import sys
import time
import traceback

from .callable_func import CallableFunc
from .hat_file import Function, HATFile
from .function_info import FunctionInfo
from .hat import load, generate_arg_sets_for_func


def print_verbose(verbose: bool, message: str):
    if verbose:
        print(message)


@dataclass
class Result:
    function_name: str
    mean: float
    median_of_means: float
    mean_of_small_means: float
    robust_mean: float
    min_of_means: float


class Benchmark:
    """A basic python-based benchmark.
    Useful for comparison only, due to overhead in the Python layer.

    Requirements:
        A compilation toolchain in your PATH: cl.exe & link.exe (Windows), gcc (Linux), or clang (macOS)
    """

    def __init__(self, hat_path: str, native_profiling: bool, working_dir: str):
        self.hat_path = hat_path
        self.hat_package, self.func_dict = load(self.hat_path, enable_native_profiling=native_profiling)
        self.hat_functions = self.func_dict.names
        self.working_dir = working_dir

        # create dictionary of function descriptions defined in the hat file
        self.function_descriptions = self.hat_package.hat_file.function_map

    def run(
        self,
        function_name: str,
        warmup_iterations: int = 10,
        min_timing_iterations: int = 100,
        batch_size: int = 10,
        min_time_in_sec: int = 10,
        input_sets_minimum_size_MB=50,
        device_id: int = 0,
        verbose: bool = False,
        time_in_ms: bool = False,
        dyn_func_shape_fn: Callable[[FunctionInfo], List[List[int]]] = None,
        input_data_process_fn: Callable[[List], List] = None
    ) -> float:
        """Runs benchmarking for a function.
           Multiple inputs are run through the function until both minimum time and minimum iterations have been reached.
           The mean duration is then calculated as mean_duration = total_time_elapsed / total_iterations_performed.
        Args:
            function_name: name of the function
            warmup_iterations: number of warmup iterations
            min_timing_iterations: minimum number of timing iterations
            min_time_in_sec: minimum amount of time to run the benchmark
            input_sets_minimum_size_MB: generate enough input sets to exceed this size to avoid cache hits
            device_id: the ID of the device on which to run the benchmark
            dyn_func_shape_fn: A callback function that's called for a function with dynamic arguments and returns the shape of arguments
        Returns:
            Mean duration in seconds,
            Vector of timings in seconds/milliseconds for each batch that was run
        """
        if function_name not in self.hat_functions:
            raise ValueError(f"{function_name} is not found")

        # TODO: support packing and unpacking functions

        mean_elapsed_time, batch_timings = self._profile(
            function_name, warmup_iterations, min_timing_iterations, batch_size, min_time_in_sec,
            input_sets_minimum_size_MB, device_id, verbose, time_in_ms, dyn_func_shape_fn, input_data_process_fn
        )

        time_unit = "ms" if time_in_ms else "s"
        print_verbose(verbose, f"[Benchmarking] Mean duration per iteration: {mean_elapsed_time} {time_unit}")

        return mean_elapsed_time, batch_timings

    def _profile(
        self,
        function_name,
        warmup_iterations,
        min_timing_iterations,
        batch_size,
        min_time_in_sec,
        input_sets_minimum_size_MB,
        device_id: int,
        verbose: bool,
        time_in_ms: bool,
        dyn_func_shape_fn: Callable[[FunctionInfo], List[List[int]]] = None,
        input_data_process_fn: Callable[[List], List] = None
    ):

        def get_perf_counter():
            if hasattr(time, 'perf_counter_ns'):
                _perf_counter = time.perf_counter_ns
                perf_counter_scale = 1000000000
            else:
                _perf_counter = time.perf_counter
                perf_counter_scale = 1

            def perf_counter():
                return _perf_counter() / perf_counter_scale

            return perf_counter

        func = self.function_descriptions[function_name]

        benchmark_func = self.func_dict[function_name]
        if not isinstance(benchmark_func, CallableFunc):
            # generate sufficient input sets to overflow the L3 cache, since we don't know the size of the model
            # we'll make a guess based on the minimum input set size
            input_sets = generate_arg_sets_for_func(
                func, input_sets_minimum_size_MB, num_additional=10, dyn_func_shape_fn=dyn_func_shape_fn
            )

            if input_data_process_fn:
                input_sets = input_data_process_fn(input_sets)

            set_size = 0
            for i in input_sets[0]:
                if not i.dim_values:
                    set_size += i.value.size * i.value.dtype.itemsize

            print_verbose(verbose, f"[Benchmarking] Using {len(input_sets)} input sets, each {set_size} bytes")

            perf_counter = get_perf_counter()
            print_verbose(verbose, f"[Benchmarking] Warming up for {warmup_iterations} iterations...")

            for _ in range(warmup_iterations):
                for calling_args in input_sets:
                    benchmark_func(*calling_args)

            print_verbose(verbose, f"[Benchmarking] Timing for at least {min_time_in_sec}s and at least {min_timing_iterations} iterations...")

            i = 0
            i_max = len(input_sets)
            iterations = 1
            batch_timings = []
            end_time_secs = perf_counter()
            start_time_secs = perf_counter()
            while True:
                batch_start_time_secs = perf_counter()
                for _ in range(min_timing_iterations):
                    benchmark_func(*input_sets[i])
                    i = iterations % i_max
                    iterations += 1
                end_time_secs = perf_counter()
                batch_timings.append((end_time_secs - batch_start_time_secs))

                if ((end_time_secs - start_time_secs)) >= min_time_in_sec or len(batch_timings) >= batch_size:
                    break

            elapsed_time_secs = ((end_time_secs - start_time_secs))
            elapsed_time = elapsed_time_secs * (1000.0 if time_in_ms else 1.0)
            mean_elapsed_time = elapsed_time / iterations
            batch_timings = list(map(lambda t: t * 1000, batch_timings)) if time_in_ms else batch_timings
            return mean_elapsed_time, batch_timings
        else:
            print_verbose(
                verbose, f"[Benchmarking] Benchmarking device function on device {device_id}. {batch_size} batches of warming up for {warmup_iterations} and then measuring with {min_timing_iterations} iterations."
            )

            if benchmark_func.should_flush_cache():
                input_sets = generate_arg_sets_for_func(
                    func, input_sets_minimum_size_MB, num_additional=10, dyn_func_shape_fn=dyn_func_shape_fn
                )
            else:
                print_verbose(verbose, "[Benchmarking] Benchmarking device that does not need cache flushing, skipping generation of multiple datasets")

                input_sets = [generate_arg_sets_for_func(func, dyn_func_shape_fn=dyn_func_shape_fn)]

            if input_data_process_fn:
                input_sets = input_data_process_fn(input_sets)

            set_size = 0
            for i in input_sets[0]:
                if not i.dim_values:
                    set_size += i.value.size * i.value.dtype.itemsize

            print_verbose(verbose, f"[Benchmarking] Using input of {set_size} bytes")

            mean_elapsed_time_ms, batch_timings_ms = benchmark_func.benchmark(
                warmup_iters=warmup_iterations,
                iters=min_timing_iterations,
                batch_size=batch_size,
                min_time_in_sec=min_time_in_sec,
                args=input_sets,
                device_id=device_id,
                working_dir=self.working_dir
            )
            batch_timings = batch_timings_ms if time_in_ms else list(map(lambda t: t / 1000, batch_timings_ms))
            mean_timings = mean_elapsed_time_ms if time_in_ms else mean_elapsed_time_ms / 1000
            return mean_timings, batch_timings


def write_runtime_to_hat_file(hat_path, function_name, mean_time_secs):
    """Writes the mean time in seconds to a HAT file
    """
    # Write back the runtime to the HAT file
    hat_file = HATFile.Deserialize(hat_path)
    hat_func = hat_file.function_map.get(function_name)
    hat_func.auxiliary["mean_duration_in_sec"] = mean_time_secs

    hat_file.Serialize(hat_path)

    # Workaround to remove extra empty lines
    with open(hat_path, "r") as f:
        lines = f.readlines()
        lines = [
            lines[i] for i in range(len(lines))
            if not (lines[i] == "\n" and i < len(lines) - 1 and lines[i + 1] == "\n")
        ]
    with open(hat_path, "w") as f:
        f.writelines(lines)


def run_benchmark(
    hat_path,
    store_in_hat=False,
    batch_size=10,
    min_time_in_sec=10,
    min_timing_iterations=100,
    input_sets_minimum_size_MB=50,
    device_id: int = 0,
    verbose: bool = False,
    native_profiling: bool = False,
    time_in_ms: bool = False,
    functions: List[str] = None,
    working_dir: str = None
) -> List[Result]:
    results = []

    benchmark = Benchmark(hat_path, native_profiling, working_dir)
    functions = functions if functions is not None else benchmark.hat_functions
    for function_name in functions:
        if "Initialize" in function_name or "_debug_check_allclose" in function_name:    # Skip init and debug functions
            print(f"\nSkipping function: {function_name}")
            continue

        print_verbose(verbose, f"\nBenchmarking function: {function_name}")

        try:
            _, batch_timings = benchmark.run(
                function_name,
                warmup_iterations=batch_size,
                min_timing_iterations=min_timing_iterations,
                batch_size=batch_size,
                min_time_in_sec=min_time_in_sec,
                input_sets_minimum_size_MB=input_sets_minimum_size_MB,
                device_id=device_id,
                time_in_ms=time_in_ms,
                verbose=verbose
            )

            num_batches = len(batch_timings)
            sorted_batch_means = np.array(sorted(batch_timings)) / min_timing_iterations

            mean_of_means = sorted_batch_means.mean()
            median_of_means = sorted_batch_means[num_batches // 2]
            mean_of_small_means = sorted_batch_means[0:max(1, num_batches // 2)].mean()
            robust_means = sorted_batch_means[(num_batches // 5):(-num_batches // 5)]
            robust_mean_of_means = robust_means.mean() if len(robust_means) > 0 else -1
            min_of_means = sorted_batch_means[0]

            if store_in_hat:
                write_runtime_to_hat_file(hat_path, function_name, mean_of_means)
            results.append(
                Result(
                    **{
                        "function_name": function_name,
                        "mean": mean_of_means,
                        "median_of_means": median_of_means,
                        "mean_of_small_means": mean_of_small_means,
                        "robust_mean": robust_mean_of_means,
                        "min_of_means": min_of_means,
                    }
                )
            )
        except Exception as e:
            if verbose:
                exc_type, exc_val, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)
                print("\nException message: ", e)
                print(f"WARNING: Failed to run function {function_name}, skipping this benchmark.")

            raise e
    return results


def main(argv):
    arg_parser = argparse.ArgumentParser(
        description="Benchmarks each function in a HAT package and estimates its duration.\n"
        "Example:\n"
        "    hatlib.benchmark_hat_package <hat_path>\n"
    )

    arg_parser.add_argument("hat_path", help="Path to the HAT file", default=None)
    arg_parser.add_argument(
        "--store_in_hat",
        help="If set, will write the duration as meta-data back into the hat file",
        action='store_true'
    )
    arg_parser.add_argument("--results_file", help="Full path where the results will be written", default="results.csv")
    arg_parser.add_argument(
        "--batch_size",
        help="The number of function calls in each batch (at least one full batch is executed)",
        default=10
    )
    arg_parser.add_argument("--min_time_in_sec", help="Minimum number of seconds to run the benchmark for", default=30)
    arg_parser.add_argument("--min_timing_iterations", help="Minimum number of iterations per batch", default=100)
    arg_parser.add_argument(
        "--input_sets_minimum_size_MB",
        help=
        "Minimum size in MB of the input sets. Typically this is large enough to ensure eviction of the biggest cache on the target (e.g. L3 on an desktop CPU)",
        default=50
    )
    arg_parser.add_argument("--verbose", help="Enable verbose logging", default=False)
    arg_parser.add_argument(
        "--cpp", help="If set, timings with be measured in native code instead of in python", action='store_true'
    )
    arg_parser.add_argument(
        "--time_in_ms", help="If set, timings with be measured in milliseconds instead of seconds", action='store_true'
    )
    arg_parser.add_argument("--functions", type=str, nargs="+", help="Functions to benchmark")
    arg_parser.add_argument("--working_dir", help="Path to a working directory. Defaults to the current directory", default=None)

    args = vars(arg_parser.parse_args(argv))

    results = run_benchmark(
        args["hat_path"],
        args["store_in_hat"],
        batch_size=int(args["batch_size"]),
        min_time_in_sec=int(args["min_time_in_sec"]),
        min_timing_iterations=int(args["min_timing_iterations"]),
        input_sets_minimum_size_MB=int(args["input_sets_minimum_size_MB"]),
        verbose=bool(args["verbose"]),
        native_profiling=args["cpp"],
        time_in_ms=args["time_in_ms"],
        functions=args["functions"],
        working_dir=args["working_dir"] or os.getcwd()
    )
    df = pd.DataFrame(results)
    df.to_csv(args["results_file"], index=False, mode='a', header=not os.path.exists(args["results_file"]))
    pd.options.display.float_format = '{:8.8f}'.format
    print(df)

    print(f"Results saved to {args['results_file']}")


def main_command():
    main(sys.argv[1:])    # drop the first argument (program name)


if __name__ == "__main__":
    main_command()
