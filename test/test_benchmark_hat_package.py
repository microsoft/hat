#!/usr/bin/env python3
import unittest
import accera as acc
import numpy as np
from hatlib import run_benchmark


class BenchmarkHATPackage_test(unittest.TestCase):

    def test_benchmark(self):
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 256))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 256))
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 256))

        nest = acc.Nest(shape=(256, 256, 256))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        package = acc.Package()
        package.add(nest, args=(A, B, C), base_name="test_function")
        package.build(
            name="BenchmarkHATPackage_test_benchmark", output_dir="test_acccgen", format=acc.Package.Format.HAT_DYNAMIC
        )

        results = run_benchmark(
            "test_acccgen/BenchmarkHATPackage_test_benchmark.hat",
            store_in_hat=False,
            batch_size=2,
            min_time_in_sec=1,
            input_sets_minimum_size_MB=1
        )
        self.assertIn("test_function", results[0].function_name)
        self.assertEqual(type(results[0].mean), np.float64)

    def test_benchmark_multiple_functions(self):
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 256))
        B = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 256))
        C = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(256, 256))
        D = acc.Array(role=acc.Array.Role.INPUT, shape=(256, 256))    # dummy argument

        nest = acc.Nest(shape=(256, 256, 256))
        i, j, k = nest.get_indices()

        @nest.iteration_logic
        def _():
            C[i, j] += A[i, k] * B[k, j]

        package = acc.Package()
        package.add(nest, args=(A, B, C), base_name="test_function")

        # add another function with a dummy argument - run_benchmark should be able to call it
        # with the correct signature
        package.add(nest, args=(A, B, C, D), base_name="test_function_dummy")

        package.build(
            name="BenchmarkHATPackage_test_benchmark", output_dir="test_acccgen", format=acc.Package.Format.HAT_DYNAMIC
        )

        results = run_benchmark(
            "test_acccgen/BenchmarkHATPackage_test_benchmark.hat",
            store_in_hat=False,
            batch_size=2,
            min_time_in_sec=1,
            input_sets_minimum_size_MB=1
        )

        def drop_hash_suffix(name: str) -> str:
            return name[:name.rfind("_")]

        # BUGBUG: shouldn't the order be based on package.add?
        func_names = [drop_hash_suffix(r.function_name) for r in results]
        self.assertIn("test_function", func_names)
        self.assertIn("test_function_dummy", func_names)
        for r in results:
            self.assertEqual(type(r.mean), np.float64)


if __name__ == '__main__':
    unittest.main()
