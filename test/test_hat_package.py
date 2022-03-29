#!/usr/bin/env python3

import sys
import os
import unittest
from pathlib import Path
from hatlib import (
    CallingConventionType, CompiledWith, Declaration, Dependencies, Description, Function, FunctionTable, HATFile,
    OperatingSystem, Parameter, ParameterType, Target, UsageType
)


class HATFile_test(unittest.TestCase):

    def test_file_basic_serialize(self):
        # Construct a HAT file from scratch
        # Start with a function definition
        my_function = Function(
            name="my_function",
            description="Some description",
            calling_convention=CallingConventionType.StdCall,
            return_info=Parameter(
                logical_type=ParameterType.RuntimeArray,
                declared_type="float*",
                element_type="float",
                usage=UsageType.Input,
                shape="[16, 16]",
                affine_map=[16, 1],
                size="16 * 16 * sizeof(float)"
            )
        )
        # Create the function table
        functions = FunctionTable({"my_function": my_function})
        # Create the HATFile object
        hat_file1 = HATFile(
            name="test_file",
            description=Description(
                version="0.0.1", author="me", license_url="https://www.apache.org/licenses/LICENSE-2.0.html"
            ),
            _function_table=functions,
            target=Target(
                required=Target.Required(
                    os=OperatingSystem.Windows,
                    cpu=Target.Required.CPU(architecture="Haswell", extensions=["AVX2"]),
                    gpu=None
                ),
                optimized_for=Target.OptimizedFor()
            ),
            dependencies=Dependencies(link_target="my_lib.lib"),
            compiled_with=CompiledWith(compiler="VC++"),
            declaration=Declaration(),
            path=Path(".").resolve()
        )
        # Serialize it to disk
        test_file_name = "test_file_serialize.hat"

        try:
            hat_file1.Serialize(test_file_name)
            # Deserialize it and verify it has what we expect
            hat_file2 = HATFile.Deserialize(test_file_name)
        finally:
            # Remove the file
            os.remove(test_file_name)

        # Do basic verification that the deserialized HatFile contains what we
        # specified when we created the HATFile directly
        self.assertEqual(hat_file1.description, hat_file2.description)
        self.assertEqual(hat_file1.dependencies, hat_file2.dependencies)
        self.assertEqual(hat_file1.compiled_with.to_table(), hat_file2.compiled_with.to_table())
        self.assertTrue("my_function" in hat_file2.function_map)

    def test_file_basic_deserialize(self):
        # Load a HAT file from the samples directory
        hat_file1 = HATFile.Deserialize(
            os.path.join(os.path.dirname(__file__), "..", "samples", "sample_gemm_library.hat")
        )
        description = {
            "author": "John Doe",
            "version": "1.2.3.5",
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        }

        # Do basic verification of known values in the file
        # Verify the description has entries we expect
        self.assertLessEqual(description.items(), hat_file1.description.to_table().items())
        # Verify the list of functions
        self.assertTrue(len(hat_file1.function_map) == 2)
        self.assertTrue("GEMM_B94D27B9934D3E08" in hat_file1.function_map)
        self.assertTrue("blas_sgemm_row_major" in hat_file1.function_map)


if __name__ == '__main__':
    unittest.main()
