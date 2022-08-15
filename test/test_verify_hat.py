#!/usr/bin/env python3
from email.mime import base
import os
from posixpath import basename
import unittest
import hatlib as hat
import shutil


class VerifyHat_test(unittest.TestCase):

    def build(self, impl_code: str, workdir: str, name: str) -> str:
        source_path = f"{workdir}/{name}.c"
        lib_path = f"{workdir}/{name}.so"

        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        with open(source_path, "w") as f:
            print(impl_code, file=f)

        if os.path.exists(lib_path):
            os.remove(lib_path)

        if hat.get_platform() == hat.OperatingSystem.Windows:
            raise NotImplementedError("Windows support not yet added")

        hat.run_command(f'gcc -shared -fPIC -o "{lib_path}" "{source_path}"', quiet=True)
        self.assertTrue(os.path.isfile(lib_path))
        return lib_path

    def create_hat_file(self, hat_input: hat.HATFile, hat_path: str, name: str) -> str:
        if os.path.exists(hat_path):
            os.remove(hat_path)
        hat_input.Serialize(hat_path)
        self.assertTrue(os.path.exists(hat_path))

    def test_basic(self):
        # Generate a HAT package with a C implementation and call verify_hat
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
        if hat.get_platform() == hat.OperatingSystem.Windows:
            return    # TODO

        workdir = "./test_output/verify_hat_test_basic"
        name = "softmax"
        lib_path = self.build(impl_code, workdir, name)
        hat_path = f"{workdir}/{name}.hat"

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
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )

        self.create_hat_file(hat_input, hat_path, name)
        hat.verify_hat_package(hat_path)

    def test_output_runtime_array(self):
        # Generate a HAT package using C and call verify_hat
        impl_code = '''#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef ALLOC
#define ALLOC(size) ( malloc(size) )
#endif
#ifndef DEALLOC
#define DEALLOC(X) ( free(X) )
#endif

#ifdef _MSC_VER
#define DLL_EXPORT  __declspec( dllexport )
#else
#define DLL_EXPORT
#endif

DLL_EXPORT void Range(const int32_t start[1], const int32_t limit[1], const int32_t delta[1], int32_t** output, uint32_t* output_dim)
{
    /* Range */
    /* Ensure we don't crash with random inputs */
    int32_t start0 = start[0];
    int32_t delta0 = delta[0] == 0 ? 1 : delta[0];
    int32_t limit0 = (limit[0] <= start0) ? (start0 + delta0 * 25) : limit[0];

    *output_dim = (limit0 - start0) / delta0;
    *output = (int32_t*)ALLOC(*output_dim * sizeof(int32_t));
    printf(\"Allocated %d output elements\\n\", *output_dim);
    printf(\"start=%d, limit=%d, delta=%d\\n\", start0, limit0, delta0);

    for (uint32_t i = 0; i < *output_dim; ++i) {
        (*output)[i] = start0 + (i * delta0);
    }

    for (uint32_t i = 0; i < *output_dim; ++i) {
        (*output)[i] = start0 + (i * delta0);
    }
}
'''
        decl_code = '''#endif // TOML
#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)

void Range(const int32_t start[1], const int32_t limit[1], const int32_t delta[1], int32_t** output, uint32_t* output_dim);

#ifndef __Range_DEFINED__
#define __Range_DEFINED__
void (*Range)(int32_t*, int32_t*, int32_t*, int32_t**, uint32_t*) = Range;
#endif

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        if hat.get_platform() == hat.OperatingSystem.Windows:
            return    # TODO

        workdir = "test_output/verify_hat_test_output_runtime_array"
        name = "range"
        lib_path = self.build(impl_code, workdir, name)
        hat_path = f"{workdir}/{name}.hat"

        # create the hat file
        param_start = hat.Parameter(
            name="start",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="int32_t*",
            element_type="int32_t",
            usage=hat.UsageType.Input,
            shape=[],
        )
        param_limit = hat.Parameter(
            name="limit",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="int32_t*",
            element_type="int32_t",
            usage=hat.UsageType.Input,
            shape=[],
        )
        param_delta = hat.Parameter(
            name="delta",
            logical_type=hat.ParameterType.AffineArray,
            declared_type="int32_t*",
            element_type="int32_t",
            usage=hat.UsageType.Input,
            shape=[],
        )
        param_output = hat.Parameter(
            name="output",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="int32_t**",
            element_type="int32_t",
            usage=hat.UsageType.Output,
            size="output_dim"
        )
        param_output_dim = hat.Parameter(
            name="output_dim",
            logical_type=hat.ParameterType.Element,
            declared_type="uint32_t*",
            element_type="uint32_t",
            usage=hat.UsageType.Output,
            shape=[]
        )
        hat_function = hat.Function(
            arguments=[param_start, param_limit, param_delta, param_output, param_output_dim],
            calling_convention=hat.CallingConventionType.StdCall,
            name="Range",
            return_info=hat.Parameter.void()
        )
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )

        self.create_hat_file(hat_input, hat_path, name)

        # verify
        hat.verify_hat_package(hat_path)

    def test_input_output_runtime_arrays(self):
        impl_code = '''#include <stdint.h>
#include <stdlib.h>

#ifndef ALLOC
#define ALLOC(size) ( malloc(size) )
#endif
#ifndef DEALLOC
#define DEALLOC(X) ( free(X) )
#endif

#ifdef _MSC_VER
#define DLL_EXPORT  __declspec( dllexport )
#else
#define DLL_EXPORT
#endif

DLL_EXPORT void /* Unsqueeze_18 */ Unsqueeze( const int64_t* data, const int64_t output_dim0, int64_t** expanded, int64_t* dim0, int64_t* dim1 )
{
    /* Unsqueeze */
    *dim0 = 1;
    *dim1 = output_dim0;
    *expanded = (int64_t*)ALLOC((*dim0) * (*dim1) * sizeof(int64_t));
    int64_t* data_ = (int64_t*)data;
    int64_t* expanded_ = (int64_t*)(*expanded);
    for (uint32_t i = 0; i < output_dim0; ++i)
        expanded_[i] = data_[i];
}
'''
        decl_code = '''#endif // TOML
#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C"
{
#endif // defined(__cplusplus)

void Unsqueeze(const int64_t* data, const int64_t output_dim0, int64_t** expanded, int64_t* dim0, int64_t* dim1 );

#ifndef __Unsqueeze_DEFINED__
#define __Unsqueeze_DEFINED__
void (*Unsqueeze)(int64_t**, int64_t**, int64_t**, int64_t**, int64_t**) = Unsqueeze;
#endif

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        if hat.get_platform() == hat.OperatingSystem.Windows:
            return    # TODO

        workdir = "test_output/verify_hat_test_input_output_runtime_arrays"
        name = "unsqueeze"
        lib_path = self.build(impl_code, workdir, name)
        hat_path = f"{workdir}/{name}.hat"

        # create the hat file


if __name__ == '__main__':
    unittest.main()