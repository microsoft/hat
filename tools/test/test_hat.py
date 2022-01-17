#!/usr/bin/env python3
import unittest
import sys, os
import accera as acc
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hat import load
from hat_to_dynamic import create_dynamic_package

class HAT_test(unittest.TestCase):

    def test_load(self):

        # Generate a HAT package
        A = acc.Array(role=acc.Array.Role.INPUT, shape=(16, 16))
        B = acc.Array(role=acc.Array.Role.INPUT_OUTPUT, shape=(16, 16))

        nest = acc.Nest(shape=(16, 16))
        i, j = nest.get_indices()

        @nest.iteration_logic
        def _():
            B[i, j] += A[i, j]

        package = acc.Package()
        package.add(nest, args=(A, B), base_name="test_function")

        for mode in [acc.Package.Mode.RELEASE, acc.Package.Mode.DEBUG]:
            print(mode)
            package.build(name="HAT_test_load", output_dir="test_acccgen", mode=mode)

            create_dynamic_package("test_acccgen/HAT_test_load.hat", "test_acccgen/HAT_test_load.dyn.hat")
            hat_package = load("test_acccgen/HAT_test_load.dyn.hat")

            for name in hat_package.names:
                print(name)

            # create numpy arguments with the correct shape and dtype
            A = np.random.rand(16, 16).astype(np.float32) 
            B = np.random.rand(16, 16).astype(np.float32)
            B_ref = B + A

            name = hat_package.names[0]
            test_function = hat_package[name]
            test_function(A, B)

            # check for correctness
            np.testing.assert_allclose(B, B_ref)