#!/usr/bin/env python3
import unittest
import sys, os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hat import load
from hat_to_dynamic import get_platform, create_dynamic_package

class HAT_test(unittest.TestCase):
    def setUp(self):
        self.hatfile_path = Path(os.path.dirname(__file__)) / "data" / get_platform().lower() / "optimized_matmul.hat"
        self.dyn_hatfile_path = Path(os.path.dirname(__file__)) / "data" / get_platform().lower() / "optimized_matmul.HAT_test.hat"

    @unittest.skipUnless(get_platform().lower() == "windows")
    def test_load(self):
        import numpy as np

        create_dynamic_package(self.hatfile_path, self.dyn_hatfile_path)
        package = load(self.dyn_hatfile_path)

        for name in package.names:
            print(name)

        # create numpy arguments with the correct shape and dtype
        A = np.random.rand(784, 128).astype(np.float32) 
        B = np.random.rand(128, 512).astype(np.float32)
        C = np.random.rand(784, 512).astype(np.float32)
        C_ref = C + A @ B

        # call the function
        name = package.names[0]
        optimized_matmul = package[name]
        optimized_matmul(A, B, C)

        # check for correctness
        np.testing.assert_allclose(C, C_ref)