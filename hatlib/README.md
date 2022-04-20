# HAT Package Tools

## hatlib.load
Loads a dynamically-linked HAT package in Python

Usage:

```python
    import numpy as np
    import hatlib as hat

    # load the package
    _, package = hat.load("my_package.hat")

    # print the function names
    for name in package.names():
        print(name)

    # create numpy arguments with the correct shape, dtype, and order
    A = np.ones([256,32], dtype=np.float32, order="C")
    B = np.ones([32,256], dtype=np.float32, order="C")
    D = np.ones([256,32], dtype=np.float32, order="C")
    E = np.ones([256,32], dtype=np.float32, order="C")

    # call a package function named 'my_func_698b5e5c'
    package.my_func_698b5e5c(A, B, D, E)
```


## hatlib.hat_to_dynamic
A tool that converts a statically-linked HAT package into a dynamically-linked HAT package

Usage:

```shell
> hatlib.hat_to_dynamic --help

usage: hatlib.hat_to_dynamic [-h] [-v] input_hat_path output_hat_path

Creates a dynamically-linked HAT package from a statically-linked HAT package. Example: hatlib.hat_to_dynamic input.hat output.hat

positional arguments:
  input_hat_path   Path to the existing HAT file, which represents a statically-linked HAT package
  output_hat_path  Path to the new HAT file, which will represent a dynamically-linked HAT package

optional arguments:
  -h, --help       show this help message and exit
  -v, --verbose    Enable verbose output
```

## hatlib.hat_to_lib
A tool that converts a HAT package with .obj/.o into a HAT package with a .lib/.a

Usage:

```shell
> hatlib.hat_to_lib --help
usage: hatlib.hat_to_dynamic [-h] [-v] input_hat_path output_hat_path

Creates a statically-linked HAT package with a .lib/.a from a statically-linked HAT package with an .obj/.o. Example: hatlib.hat_to_lib input.hat output.hat

positional arguments:
  input_hat_path   Path to the existing HAT file, which represents a statically-linked HAT package with an .obj/.o
  output_hat_path  Path to the new HAT file, which will represent a statically-linked HAT package with a .lib/.a

optional arguments:
  -h, --help       show this help message and exit
  -v, --verbose    Enable verbose output
```

## hatlib.benchmark_hat
Tool used to benchmark functions in a HAT package.

It is common to produce a HAT package with Accera that includes multiple functions that have the same logic but have different schedules. This tool can be used to find the best performing function on a given target.

### Description
This tool will take a given HAT package and perform the following actions:

- Introspect the function data to find input and output arguments
- Pre-allocate a set of input and output buffers. The set will be large enough to ensure that data is not kept in any caches (e.g. L1, L2 or L3 of a CPU)
- Generate random input data
- Call the function in a loop running through input sets until a minimum amount of time and minimum number of iterations has passed
- Calculate the mean duration for the function
- Store the results, either in a __.csv__ file or in the HAT package as the function metadata

NOTE: The results should only be used to compare relative performance of functions measured using this tool. It is not accurate to compare duration measurents from this tool with duration measured from another tool.

### Usage

```shell
> hatlib.benchmark_hat --help
usage: benchmark_hat_package.py [-h] [--store_in_hat]
                                [--results_file RESULTS_FILE]
                                [--min_iterations MIN_ITERATIONS]
                                [--min_time_in_sec MIN_TIME_IN_SEC]
                                [--input_sets_minimum_size_MB INPUT_SETS_MINIMUM_SIZE_MB]
                                path_to_hat_package

Benchmarks each function in a HAT package and estimates its duration. Example: hatlib.benchmark_hat <hat_path>

positional arguments:
  hat_path   Path to the HAT file

optional arguments:
  -h, --help            show this help message and exit
  --store_in_hat        If set, will write the duration as meta-data back into
                        the hat file
  --results_file RESULTS_FILE
                        Full path where the results will be written
  --min_iterations MIN_ITERATIONS
                        Minimum number of iterations to run
  --min_time_in_sec MIN_TIME_IN_SEC
                        Minimum number of seconds to run the benchmark for
  --input_sets_minimum_size_MB INPUT_SETS_MINIMUM_SIZE_MB
                        Minimum size in MB of the input sets. Typically this
                        is large enough to ensure eviction of the biggest
                        cache on the target (e.g. L3 on an desktop CPU)
```

For example:
```shell
hatlib.benchmark_hat C:\myProject\my_package.hat --min_time_in_sec=15
```

#### --store_in_hat

When using `--store_in_hat` flag, the HAT package will be updated with an `auxiliary` data section like:

```
[functions.myfunction_py_c3723b5f.auxiliary]
mean_duration_in_sec = 1.5953456437541567e-06
```


## Unit tests

This repository contains unit tests, authored with the Python `unittest` library. To setup and run all tests:

```shell
pip install -r test/requirements.txt
python -m unittest discover test
```

To run a test case:

```shell
python -m unittest discover -k "test_file_basic_serialize" test
```

Note that some tests will require a C++ compiler (e.g. MSVC for windows, gcc for linux) in `PATH`.
