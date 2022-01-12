
#!/usr/bin/env python3
import unittest
import sys, os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from benchmark_hat_package import run_benchmark
from hat_to_dynamic import get_platform

class BenchmarkHATPackage_test(unittest.TestCase):
    def setUp(self):
        self.hatfile_path = Path(os.path.dirname(__file__)) / "data" / get_platform().lower() / "optimized_matmul.hat"

    @unittest.skipUnless(get_platform().lower() == "macos", "macOS is not supported by test")
    def test_benchmark(self):
        run_benchmark(self.hatfile_path, store_in_hat=False, batch_size=2, min_time_in_sec=1, input_sets_minimum_size_MB=1)

if __name__ == '__main__':
    unittest.main()