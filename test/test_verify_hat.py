#!/usr/bin/env python3
import hatlib as hat
import numpy as np
import os
import shutil
import unittest


class VerifyHat_test(unittest.TestCase):

    def build(self, impl_code: str, workdir: str, name: str, func_name: str) -> str:
        hat.ensure_compiler_in_path()
        if hat.get_platform() == hat.OperatingSystem.Windows:
            return self.windows_build(impl_code, workdir, name, func_name)
        else:
            return self.linux_build(impl_code, workdir, name)

    def windows_build(self, impl_code: str, workdir: str, name: str, func_name: str) -> str:
        source_path = f"{workdir}/{name}.c"
        lib_path = f"{workdir}/{name}.dll"

        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        with open(source_path, "w") as f:
            print(impl_code, file=f)

        dllmain_path = f"{workdir}/dllmain.cpp"
        with open(dllmain_path, "w") as f:
            print("#include <windows.h>\n", file=f)
            print("BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }\n", file=f)

        if os.path.exists(lib_path):
            os.remove(lib_path)

        hat.run_command(
            f'cl.exe "{source_path}" "{dllmain_path}" /nologo /link /DLL /EXPORT:{func_name} /OUT:"{lib_path}"',
            quiet=True
        )
        self.assertTrue(os.path.isfile(lib_path))
        return lib_path

    def linux_build(self, impl_code: str, workdir: str, name: str) -> str:
        source_path = f"{workdir}/{name}.c"
        lib_path = f"{workdir}/{name}.so"

        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)
        with open(source_path, "w") as f:
            print(impl_code, file=f)

        if os.path.exists(lib_path):
            os.remove(lib_path)

        hat.run_command(f'gcc -shared -fPIC -o "{lib_path}" "{source_path}"', quiet=True)
        self.assertTrue(os.path.isfile(lib_path))
        return lib_path

    def create_hat_file(self, hat_input: hat.HATFile):
        hat_path = hat_input.path
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

void Softmax(const float input[2][2], float output[2][2]);

#ifndef __Softmax_DEFINED__
#define __Softmax_DEFINED__
void (*Softmax)(float*, float*) = Softmax;
#endif

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        workdir = "./test_output/verify_hat_basic"
        name = "softmax"
        func_name = "Softmax"
        lib_path = self.build(impl_code, workdir, name, func_name)
        hat_path = f"{workdir}/{name}.hat"

        # create the hat file
        shape = [2, 2]
        strides = [shape[1], 1]    # first major
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
            name=func_name,
            return_info=hat.Parameter.void()
        )
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )
        self.create_hat_file(hat_input)
        hat.verify_hat_package(hat_path)

    def test_runtime_array(self):
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
    int32_t delta0;
    if (limit[0] < start[0]) {
        delta0 = delta[0] <= 0 ? delta[0] : -delta[0];
        delta0 = delta0 == 0 ? -1 : delta[0];
    } else {
        delta0 = delta[0] >= 0 ? delta[0] : -delta[0];
        delta0 = delta0 == 0 ? 1 : delta[0];
    }
    int32_t start0 = start[0];
    int32_t limit0 = limit[0];

    *output_dim = (limit0 - start0) / delta0;
    *output = (int32_t*)ALLOC(*output_dim * sizeof(int32_t));
    printf(\"Allocated %u output elements\\n\", *output_dim);
    printf(\"start=%d, limit=%d, delta=%d\\n\", start0, limit0, delta0);

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
        workdir = "test_output/verify_hat_runtime_array"
        name = "range"
        func_name = "Range"
        lib_path = self.build(impl_code, workdir, name, func_name)
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
            name=func_name,
            return_info=hat.Parameter.void()
        )
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )
        self.create_hat_file(hat_input)
        hat.verify_hat_package(hat_path)

    def test_input_runtime_arrays(self):
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

DLL_EXPORT void /* Unsqueeze_18 */ Unsqueeze(const float* data, const int64_t data_dim0, float** expanded, int64_t* dim0, int64_t* dim1)
{
    /* Unsqueeze */
    *dim0 = 1;
    *dim1 = data_dim0;
    *expanded = (float*)ALLOC((*dim0) * (*dim1) * sizeof(float));
    float* data_ = (float*)data;
    float* expanded_ = (float*)(*expanded);
    for (int64_t i = 0; i < data_dim0; ++i)
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

void Unsqueeze(const float* data, const int64_t data_dim0, float** expanded, int64_t* dim0, int64_t* dim1);

#ifndef __Unsqueeze_DEFINED__
#define __Unsqueeze_DEFINED__
void (*Unsqueeze_)(float*, int64_t, float**, int64_t*, int64_t*) = Unsqueeze;
#endif

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        for id, usage in enumerate([hat.UsageType.Input, hat.UsageType.InputOutput]):
            workdir = "test_output/verify_hat_inout_runtime_arrays"
            name = f"unsqueeze_{id}"    # uniqify for Windows to avoid load conflict
            func_name = "Unsqueeze"
            lib_path = self.build(impl_code, workdir, name, func_name)
            hat_path = f"{workdir}/{name}.hat"

            # create the hat file
            param_data = hat.Parameter(
                name="data",
                logical_type=hat.ParameterType.RuntimeArray,
                declared_type="float*",
                element_type="float",
                usage=usage,
                size="data_dim"
            )
            param_data_dim = hat.Parameter(
                name="data_dim",
                logical_type=hat.ParameterType.Element,
                declared_type="int64_t",
                element_type="int64_t",
                usage=hat.UsageType.Input,
                shape=[]
            )
            param_expanded = hat.Parameter(
                name="expanded",
                logical_type=hat.ParameterType.RuntimeArray,
                declared_type="float**",
                element_type="float",
                usage=hat.UsageType.Output,
                size="dim0*dim1"
            )
            param_dim0 = hat.Parameter(
                name="dim0",
                logical_type=hat.ParameterType.Element,
                declared_type="int64_t*",
                element_type="int64_t",
                usage=hat.UsageType.Output,
                shape=[]
            )
            param_dim1 = hat.Parameter(
                name="dim1",
                logical_type=hat.ParameterType.Element,
                declared_type="int64_t*",
                element_type="int64_t",
                usage=hat.UsageType.Output,
                shape=[]
            )
            hat_function = hat.Function(
                arguments=[param_data, param_data_dim, param_expanded, param_dim0, param_dim1],
                calling_convention=hat.CallingConventionType.StdCall,
                name=func_name,
                return_info=hat.Parameter.void()
            )
            hat_input = hat.HATFile(
                name=name,
                functions=[hat_function],
                dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
                declaration=hat.Declaration(code=decl_code),
                path=hat_path
            )
            self.create_hat_file(hat_input)
            hat.verify_hat_package(hat_path)

    def test_partial_dynamic_runtime_arrays(self):
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

#define DIM1 100
#define DIM2 16

DLL_EXPORT void /* Add_155 */ Add_partial_dynamic( const float* A, uint32_t A_dim0, const float* B, float** C, uint32_t* C_dim0 )
{
    (*C_dim0) = A_dim0;
    (*C) = (float*)ALLOC((*C_dim0)*DIM1*DIM2*4);
    for (unsigned i0 = 0; i0 < (*C_dim0); ++i0) {
    for (unsigned i1 = 0; i1 < DIM1; ++i1) {
    for (unsigned i2 = 0; i2 < DIM2; ++i2) {
        *(*C + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) = *(A + (A_dim0 == 1 ? 0 : i0)*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) + *(B + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1);
        // printf(\"A[%d][%d][%d]=%f, \", i0, i1, i2, (double)(*(A + (A_dim0 == 1 ? 0 : i0)*DIM1*DIM2*1 + i1*DIM2*1 + i2*1)));
        // printf(\"B[%d][%d][%d]=%f, \", i0, i1, i2, (double)(*(B + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1)));
        // printf(\"C[%d][%d][%d]=%f\\n\", i0, i1, i2, (double)(*(*C + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1)));
    }
    }
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

void Add_partial_dynamic(const float* A, uint32_t A_dim0, const float* B, float** C, uint32_t* C_dim0, uint32_t* C_dim1, uint32_t* C_dim2 );

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        workdir = "test_output/test_partial_dynamic_runtime_arrays"
        name = f"add"
        func_name = "Add_partial_dynamic"
        lib_path = self.build(impl_code, workdir, name, func_name)
        hat_path = f"{workdir}/{name}.hat"
        DIM1 = 100
        DIM2 = 16

        # create the hat file
        param_A = hat.Parameter(
            name="A",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,
            size=f"A_dim0*{DIM1}*{DIM2}"
        )
        param_A_dim0 = hat.Parameter(
            name="A_dim0",
            logical_type=hat.ParameterType.Element,
            declared_type="uint32_t",
            element_type="uint32_t",
            usage=hat.UsageType.Input,
            shape=[]
        )
        param_B = hat.Parameter(
            name="B",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,
            size=f"A_dim0*{DIM1}*{DIM2}"
        )
        param_C = hat.Parameter(
            name="C",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float**",
            element_type="float",
            usage=hat.UsageType.Output,
            size=f"C_dim0*{DIM1}*{DIM2}"
        )
        param_C_dim0 = hat.Parameter(
            name="C_dim0",
            logical_type=hat.ParameterType.Element,
            declared_type="uint32_t*",
            element_type="uint32_t",
            usage=hat.UsageType.Output,
            shape=[]
        )
        hat_function = hat.Function(
            arguments=[param_A, param_A_dim0, param_B, param_C, param_C_dim0],
            calling_convention=hat.CallingConventionType.StdCall,
            name=func_name,
            return_info=hat.Parameter.void()
        )
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )
        self.create_hat_file(hat_input)
        hat.verify_hat_package(hat_path)

        # verify correctness
        _, func_map = hat.load(hat_path)
        A = np.random.rand(5, DIM1, DIM2).astype("float32")
        B = np.random.rand(5, DIM1, DIM2).astype("float32")
        C_ref = A + B

        C = func_map.Add_partial_dynamic(A, B)
        np.testing.assert_allclose(C, C_ref)

    def test_partial_dynamic_runtime_arrays_multi_output(self):
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

#define DIM1 100
#define DIM2 16

DLL_EXPORT void Add_Sub_partial_dynamic( const float* A, uint32_t A_dim0, const float* B, float** C, uint32_t* C_dim0, float** D )
{
    (*C_dim0) = A_dim0;
    (*C) = (float*)ALLOC((*C_dim0)*DIM1*DIM2*4);
    (*D) = (float*)ALLOC((*C_dim0)*DIM1*DIM2*4);
    for (unsigned i0 = 0; i0 < (*C_dim0); ++i0) {
    for (unsigned i1 = 0; i1 < DIM1; ++i1) {
    for (unsigned i2 = 0; i2 < DIM2; ++i2) {
        *(*C + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) = *(A + (A_dim0 == 1 ? 0 : i0)*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) + *(B + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1);
        *(*D + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) = *(A + (A_dim0 == 1 ? 0 : i0)*DIM1*DIM2*1 + i1*DIM2*1 + i2*1) - *(B + i0*DIM1*DIM2*1 + i1*DIM2*1 + i2*1);
    }
    }
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

void Add_Sub_partial_dynamic(const float* A, uint32_t A_dim0, const float* B, float** C, uint32_t* C_dim0, float** D, uint32_t* D_dim0 );

#if defined(__cplusplus)
} // extern "C"
#endif // defined(__cplusplus)

#ifdef TOML
'''
        workdir = "test_output/test_partial_dynamic_runtime_arrays_multi_output"
        name = f"add"
        func_name = "Add_Sub_partial_dynamic"
        lib_path = self.build(impl_code, workdir, name, func_name)
        hat_path = f"{workdir}/{name}.hat"
        DIM1 = 100
        DIM2 = 16

        # create the hat file
        param_A = hat.Parameter(
            name="A",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,
            size=f"A_dim0*{DIM1}*{DIM2}"
        )
        param_A_dim0 = hat.Parameter(
            name="A_dim0",
            logical_type=hat.ParameterType.Element,
            declared_type="uint32_t",
            element_type="uint32_t",
            usage=hat.UsageType.Input,
            shape=[]
        )
        param_B = hat.Parameter(
            name="B",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float*",
            element_type="float",
            usage=hat.UsageType.Input,
            size=f"A_dim0*{DIM1}*{DIM2}"
        )
        param_C = hat.Parameter(
            name="C",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float**",
            element_type="float",
            usage=hat.UsageType.Output,
            size=f"C_dim0*{DIM1}*{DIM2}"
        )
        param_C_dim0 = hat.Parameter(
            name="C_dim0",
            logical_type=hat.ParameterType.Element,
            declared_type="uint32_t*",
            element_type="uint32_t",
            usage=hat.UsageType.Output,
            shape=[]
        )
        param_D = hat.Parameter(
            name="D",
            logical_type=hat.ParameterType.RuntimeArray,
            declared_type="float**",
            element_type="float",
            usage=hat.UsageType.Output,
            size=f"C_dim0*{DIM1}*{DIM2}"
        )
        hat_function = hat.Function(
            arguments=[param_A, param_A_dim0, param_B, param_C, param_C_dim0, param_D],
            calling_convention=hat.CallingConventionType.StdCall,
            name=func_name,
            return_info=hat.Parameter.void()
        )
        hat_input = hat.HATFile(
            name=name,
            functions=[hat_function],
            dependencies=hat.Dependencies(link_target=os.path.basename(lib_path)),
            declaration=hat.Declaration(code=decl_code),
            path=hat_path
        )
        self.create_hat_file(hat_input)
        hat.verify_hat_package(hat_path)

        # verify correctness
        _, func_map = hat.load(hat_path)
        A = np.random.rand(5, DIM1, DIM2).astype("float32")
        B = np.random.rand(5, DIM1, DIM2).astype("float32")
        C_ref = A + B
        D_ref = A - B

        C, D = func_map.Add_Sub_partial_dynamic(A, B)
        np.testing.assert_allclose(C, C_ref)
        np.testing.assert_allclose(D, D_ref)


if __name__ == '__main__':
    unittest.main()