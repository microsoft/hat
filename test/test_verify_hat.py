#!/usr/bin/env python3
import os
import unittest
import hatlib as hat
import shutil


class VerifyHat_test(unittest.TestCase):

    def test_basic(self):
        # Generate a HAT package using C and call verify_hat
        impl_code = '''#include <math.h>
#include <stdint.h>

#ifdef _MSC_VER
#define DLL_EXPORT  __declspec( dllexport )
#else
#define DLL_EXPORT
#endif

DLL_EXPORT void Softmax(const float input[2][2], float output[2][2])
{
    /* Softmax 13 (TF, pytorch style)
       axis = 0
     */
    for (uint32_t i1 = 0; i1 < 2; ++i1) {
        float max = -INFINITY;
        for (uint32_t i0 = 0; i0 < 2; ++i0) {
            max = max > input[i0][i1] ? max : input[i0][i1];
        }
        float sum = 0.0;
        for (uint32_t i0 = 0; i0 < 2; ++i0) {
            sum += expf(input[i0][i1] - max);
        }
        for (uint32_t i0 = 0; i0 < 2; ++i0) {
            output[i0][i1] = expf(input[i0][i1] - max) / sum;
        }
    }
}
'''
        decl_code = '''#endif // TOML
#pragma once

#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)

void Softmax(const float input[2][2], float output[2][2] );

#ifndef __Softmax_DEFINED__
#define __Softmax_DEFINED__
void (*Softmax)(float*, float*) = Softmax;
#endif

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        platform = hat.get_platform()
        if platform == hat.OperatingSystem.Windows:
            return    # TODO

        workdir = "test_output/verify_hat_test_basic"
        hat_path = f"{workdir}/softmax.hat"
        source_path = f"{workdir}/softmax.c"
        lib_path = f"{workdir}/softmax.so"

        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        with open(source_path, "w") as f:
            print(impl_code, file=f)

        if os.path.exists(lib_path):
            os.remove(lib_path)
        hat.run_command(f'gcc -shared -fPIC -o "{lib_path}" "{source_path}"', quiet=True)
        self.assertTrue(os.path.isfile(lib_path))

        # create the hat file
        shape = (2, 2)
        strides = (shape[1], 1)    # first major
        param_input = hat.Parameter(
            name="input",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,
            shape=shape,
            affine_map=strides
        )
        param_output = hat.Parameter(
            name="output",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.InputOutput,
            shape=shape,
            affine_map=strides
        )
        hat_function = hat.Function(
            arguments=[param_input, param_output],
            calling_convention=hat.CallingConventionType.StdCall,
            name="Softmax",
            return_info=hat.Parameter.void()
        )
        new_hat_file = hat.HATFile(
            name="softmax",
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )

        if os.path.exists(hat_path):
            os.remove(hat_path)
        new_hat_file.Serialize(hat_path)
        self.assertTrue(os.path.exists(hat_path))

        # verify
        hat.verify_hat_package(hat_path)


if __name__ == '__main__':
    unittest.main()