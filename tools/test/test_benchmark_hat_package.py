
#!/usr/bin/env python3
import unittest
import sys, os
import accera as acc

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark_hat_package import run_benchmark
from hat_to_dynamic import get_platform

class BenchmarkHATPackage_test(unittest.TestCase):
    @unittest.skipIf(get_platform().lower() == "macos", "macOS is not supported by test yet")
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
        package.add_function(nest, args=(A, B, C), base_name="test_function")
        package.build(name="BenchmarkHATPackage_test_benchmark", output_dir="test_acccgen")

        run_benchmark("test_acccgen/BenchmarkHATPackage_test_benchmark.hat", store_in_hat=False, batch_size=2, min_time_in_sec=1, input_sets_minimum_size_MB=1)

if __name__ == '__main__':
    unittest.main()