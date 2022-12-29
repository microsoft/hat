#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
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
    def __init__(self, hat_path: str):
        self.hat_path = hat_path
        self.hat_package, self.func_dict = load(self.hat_path)
        self.hat_functions = self.func_dict.names

        # create dictionary of function descriptions defined in the hat file
        self.function_descriptions = self.hat_package.hat_file.function_map

    def run(self,
            function_name: str,
            warmup_iterations: int = 10,
            min_timing_iterations: int = 100,
            batch_size: int = 10,
            min_time_in_sec: int = 10,
            input_sets_minimum_size_MB=50,
            gpu_id: int=0,
            verbose: bool=False,
            dyn_func_shape_fn: Callable[[FunctionInfo], List[List[int]]]=None,
            input_data_process_fn: Callable[[List], List]=None) -> float:
        """Runs benchmarking for a function.
           Multiple inputs are run through the function until both minimum time and minimum iterations have been reached.
           The mean duration is then calculated as mean_duration = total_time_elapsed / total_iterations_performed.
        Args:
            function_name: name of the function
            warmup_iterations: number of warmup iterations
            min_timing_iterations: minimum number of timing iterations
            min_time_in_sec: minimum amount of time to run the benchmark
            input_sets_minimum_size_MB: generate enough input sets to exceed this size to avoid cache hits
            gpu_id: the ID of the gpu device on which to run the benchmark
            dyn_func_shape_fn: A callback function that's called for a function with dynamic arguments and returns the shape of arguments
        Returns:
            Mean duration in seconds,
            Vector of timings in seconds for each batch that was run
        """
        if function_name not in self.hat_functions:
            raise ValueError(f"{function_name} is not found")

        # TODO: support packing and unpacking functions

        mean_elapsed_time, batch_timings = self._profile(
            function_name, warmup_iterations, min_timing_iterations, batch_size,
            min_time_in_sec, input_sets_minimum_size_MB, gpu_id, verbose, dyn_func_shape_fn, input_data_process_fn)

        if verbose:
            print(f"[Benchmarking] Mean duration per iteration: {mean_elapsed_time:.8f}s")

        return mean_elapsed_time, batch_timings

    def _profile(self, function_name, warmup_iterations, min_timing_iterations, batch_size,
                 min_time_in_sec, input_sets_minimum_size_MB, gpu_id: int, verbose: bool, dyn_func_shape_fn: Callable[[FunctionInfo], List[List[int]]]=None, input_data_process_fn: Callable[[List], List]=None):
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
            input_sets = generate_arg_sets_for_func(func,
                                                    input_sets_minimum_size_MB,
                                                    num_additional=10,
                                                    dyn_func_shape_fn=dyn_func_shape_fn)

            if input_data_process_fn:
                input_sets = input_data_process_fn(input_sets)

            set_size = 0
            for i in input_sets[0]:
                if not i.dim_values:
                    set_size += i.value.size * i.value.dtype.itemsize

            if verbose:
                print(f"[Benchmarking] Using {len(input_sets)} input sets, each {set_size} bytes")

            perf_counter = get_perf_counter()
            if verbose:
                print(f"[Benchmarking] Warming up for {warmup_iterations} iterations...")

            for _ in range(warmup_iterations):
                for calling_args in input_sets:
                    self.func_dict[function_name](*calling_args)

            if verbose:
                print(f"[Benchmarking] Timing for at least {min_time_in_sec}s and at least {min_timing_iterations} iterations...")

            start_time_secs = perf_counter()
            end_time_secs = perf_counter()

            i = 0
            i_max = len(input_sets)
            iterations = 1
            batch_timings = []
            while ((end_time_secs - start_time_secs)) < min_time_in_sec and len(batch_timings) < batch_size:
                batch_start_time_secs = perf_counter()
                for _ in range(min_timing_iterations):
                    self.func_dict[function_name](*input_sets[i])
                    i = iterations % i_max
                    iterations += 1
                end_time_secs = perf_counter()
                batch_timings.append(
                    (end_time_secs - batch_start_time_secs))

            elapsed_time_secs = ((end_time_secs - start_time_secs))
            mean_elapsed_time_secs = elapsed_time_secs / iterations
            return mean_elapsed_time_secs, batch_timings
        else:
            if verbose:
                print(f"[Benchmarking] Benchmarking device function on gpu {gpu_id}. {batch_size} batches of warming up for {warmup_iterations} and then measuring with {min_timing_iterations} iterations.")
            input_sets = generate_arg_sets_for_func(func, dyn_func_shape_fn=dyn_func_shape_fn)

            if input_data_process_fn:
                input_sets = input_data_process_fn(input_sets)

            set_size = 0
            for i in input_sets:
                set_size += i.value.size * i.value.dtype.itemsize

            if verbose:
                print(f"[Benchmarking] Using input of {set_size} bytes")

            batch_timings_ms = benchmark_func.benchmark(warmup_iters=warmup_iterations, iters=min_timing_iterations, batch_size=batch_size, args=input_sets, gpu_id=gpu_id)
            batch_timings_secs = list(map(lambda t: t / 1000, batch_timings_ms))
            mean_timings = sum(batch_timings_secs) / (min_timing_iterations * batch_size)
            return mean_timings, batch_timings_secs


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
            if not (lines[i] == "\n" and i < len(lines) -
                    1 and lines[i + 1] == "\n")
        ]
    with open(hat_path, "w") as f:
        f.writelines(lines)


def run_benchmark(hat_path,
                  store_in_hat=False,
                  batch_size=10,
                  min_time_in_sec=10,
                  input_sets_minimum_size_MB=50,
                  gpu_id: int=0,
                  verbose: bool=False,
                  functions:List[str]=None) -> List[Result]:
    results = []

    benchmark = Benchmark(hat_path)
    functions = functions if functions is not None else benchmark.hat_functions
    for function_name in functions:
        if verbose:
            print(f"\nBenchmarking function: {function_name}")
        if "Initialize" in function_name or "_debug_check_allclose" in function_name:  # Skip init and debug functions
            continue

        try:
            _, batch_timings = benchmark.run(
                function_name,
                warmup_iterations=batch_size,
                min_timing_iterations=batch_size,
                batch_size=batch_size,
                min_time_in_sec=min_time_in_sec,
                input_sets_minimum_size_MB=input_sets_minimum_size_MB,
                gpu_id=gpu_id,
                verbose=verbose)

            sorted_batch_means = np.array(sorted(batch_timings)) / batch_size
            num_batches = len(batch_timings)

            mean_of_means = sorted_batch_means.mean()
            median_of_means = sorted_batch_means[num_batches // 2]
            mean_of_small_means = sorted_batch_means[0:max(1, num_batches // 2)].mean()
            robust_means = sorted_batch_means[(num_batches // 5):(-num_batches // 5)]
            robust_mean_of_means = robust_means.mean() if len(robust_means) > 0 else -1
            min_of_means = sorted_batch_means[0]

            if store_in_hat:
                write_runtime_to_hat_file(hat_path, function_name,
                                          mean_of_means)
            results.append(Result(**{
                "function_name": function_name,
                "mean": mean_of_means,
                "median_of_means": median_of_means,
                "mean_of_small_means": mean_of_small_means,
                "robust_mean": robust_mean_of_means,
                "min_of_means": min_of_means,
            }))
        except Exception as e:
            if verbose:
                exc_type, exc_val, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type,
                                        exc_val,
                                        exc_tb,
                                        file=sys.stderr)
                print("\nException message: ", e)
                print(f"WARNING: Failed to run function {function_name}, skipping this benchmark.")

            raise e
    return results


def main(argv):
    arg_parser = argparse.ArgumentParser(
        description=
        "Benchmarks each function in a HAT package and estimates its duration.\n"
        "Example:\n"
        "    hatlib.benchmark_hat_package <hat_path>\n")

    arg_parser.add_argument("hat_path",
                            help="Path to the HAT file",
                            default=None)
    arg_parser.add_argument(
        "--store_in_hat",
        help=
        "If set, will write the duration as meta-data back into the hat file",
        action='store_true')
    arg_parser.add_argument("--results_file",
                            help="Full path where the results will be written",
                            default="results.csv")
    arg_parser.add_argument(
        "--batch_size",
        help=
        "The number of function calls in each batch (at least one full batch is executed)",
        default=10)
    arg_parser.add_argument(
        "--min_time_in_sec",
        help="Minimum number of seconds to run the benchmark for",
        default=30)
    arg_parser.add_argument(
        "--input_sets_minimum_size_MB",
        help=
        "Minimum size in MB of the input sets. Typically this is large enough to ensure eviction of the biggest cache on the target (e.g. L3 on an desktop CPU)",
        default=50)
    arg_parser.add_argument(
        "--verbose",
        help="Enable verbose logging",
        default=False)

    args = vars(arg_parser.parse_args(argv))

    results = run_benchmark(args["hat_path"],
                            args["store_in_hat"],
                            batch_size=int(args["batch_size"]),
                            min_time_in_sec=int(args["min_time_in_sec"]),
                            input_sets_minimum_size_MB=int(args["input_sets_minimum_size_MB"]),
                            verbose=bool(args["verbose"]))
    df = pd.DataFrame(results)
    df.to_csv(args["results_file"], index=False)
    pd.options.display.float_format = '{:8.8f}'.format
    print(df)

    print(f"Results saved to {args['results_file']}")


def main_command():
    main(sys.argv[1:])  # drop the first argument (program name)


if __name__ == "__main__":
    main_command()
