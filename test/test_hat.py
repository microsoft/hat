#!/usr/bin/env python3
import accera as acc
import numpy as np
import os
import sys
import unittest
from hatlib import load, create_dynamic_package, create_static_package


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
        function = package.add(nest, args=(A, B), base_name="test_function")

        for mode in [acc.Package.Mode.RELEASE, acc.Package.Mode.DEBUG]:
            package_name = f"HAT_test_load_{mode.value}"
            package.build(name=package_name, output_dir="test_acccgen", format=acc.Package.Format.HAT_STATIC, mode=mode)

            create_dynamic_package(f"test_acccgen/{package_name}.hat", f"test_acccgen/{package_name}.dyn.hat")

            _, func_dict = load(f"test_acccgen/{package_name}.dyn.hat")

            for name in func_dict.names:
                print(name)

            # create numpy arguments with the correct shape and dtype
            A = np.random.rand(16, 16).astype(np.float32)
            B = np.random.rand(16, 16).astype(np.float32)
            B_ref = B + A

            # find the function by basename
            test_function = func_dict["test_function"]
            test_function(A, B)

            # check for correctness
            np.testing.assert_allclose(B, B_ref)

            # find the function by actual name
            B_ref = B + A
            test_function1 = func_dict[function.name]
            test_function1(A, B)

            # check for correctness
            np.testing.assert_allclose(B, B_ref)
