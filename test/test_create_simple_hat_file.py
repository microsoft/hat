#!/usr/bin/env python3
import unittest
import os

import hatlib as hat

SAMPLE_MATMUL_DECL_CODE = '''
#endif // TOML

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)
//
// Functions
//

void MatMul(const float* A, const float* B, float* C);

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''


class CreateSimpleHatFile_test(unittest.TestCase):

    def test_create_simple_hat_file(self):
        a_shape = (1024, 512)
        a_strides = (a_shape[1], 1) # "first major" / "row major"

        b_shape = (512, 256)
        b_strides = (1, b_shape[0]) # "last major" / "column major"

        c_shape = (1024, 256)
        c_strides = (c_shape[1], 1) # "first major" / "row major"

        param_A = hat.Parameter(
            name="A",
            description="the A input matrix argument",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,

            # Affine array parameter keys
            shape=a_shape,
            affine_map=a_strides,
            affine_offset=0
        )

        param_B = hat.Parameter(
            name="B",
            description="the B input matrix argument",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,

            # Affine array parameter keys
            shape=b_shape,
            affine_map=b_strides,
            affine_offset=0
        )

        param_C = hat.Parameter(
            name="C",
            description="the C input matrix argument",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.InputOutput,

            # Affine array parameter keys
            shape=c_shape,
            affine_map=c_strides,
            affine_offset=0
        )

        arguments = [param_A, param_B, param_C]
        return_arg = hat.Parameter.void()

        func_name = "MatMul"
        hat_function = hat.Function(
            arguments=arguments,
            calling_convention=hat.CallingConventionType.StdCall,
            description="Sample matmul hat declaration",
            name=func_name,
            return_info=return_arg
        )
        auxiliary_key_name = "test_auxiliary_key"
        hat_function.auxiliary[auxiliary_key_name] = { "name": "matmul" }

        workdir = "./test_output"
        os.makedirs(workdir, exist_ok=True)

        link_target_path = "./fake_link_target.lib"
        hat_file_path = f"{workdir}/test_simple_hat_path.hat"
        new_hat_file = hat.HATFile(
            name="simple_hat_file",
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=link_target_path),
            declaration=hat.Declaration(code=SAMPLE_MATMUL_DECL_CODE),
            path=hat_file_path
        )

        if os.path.exists(hat_file_path):
            os.remove(hat_file_path)

        new_hat_file.Serialize(hat_file_path)

        self.assertTrue(os.path.exists(hat_file_path))

        parsed_hat_file = hat.HATFile.Deserialize(hat_file_path)

        self.assertTrue(func_name in parsed_hat_file.function_map)
        self.assertEqual(parsed_hat_file.dependencies.link_target, link_target_path)
        self.assertEqual(len(parsed_hat_file.function_map[func_name].arguments), 3)
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[0].name, param_A.name)
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[1].name, param_B.name)
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[2].name, param_C.name)
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[0].shape, list(param_A.shape))
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[1].shape, list(param_B.shape))
        self.assertEqual(parsed_hat_file.function_map[func_name].arguments[2].shape, list(param_C.shape))

        # Check that the code strings are equal. Serialization/deserialization doesn't always preserve leading/trailing whitespace so use strip() to normalize
        self.assertEqual(parsed_hat_file.declaration.code.strip(), SAMPLE_MATMUL_DECL_CODE.strip())


if __name__ == '__main__':
    unittest.main()
