![HAT](https://upload.wikimedia.org/wikipedia/commons/8/80/Crystal_Project_wizard.png)

# HAT file format

HAT is a format for distributing compiled libraries of functions in the C/C++ programming languages. HAT stands for "H-file Annotated with Toml", and implies that we decorate standard C/C++ header files with metadata in the TOML markup language. 

A library in the HAT format is distributed as a pair of files, much like standard C libraries. The object file, with file extension `.lib`, contains the compiled machine code implementation of the library functions. The HAT file, with file extension `.hat`, combines standard C/C++ function declarations, like the ones you would expect to see in a `.h` file, with metadata in the TOML markup language. 

The metadata that accompanies each function describes how that function should be used and provides detail on how it was implemented. The metadata is intended to be human-readable, providing structured and systematic documentation for the library, as well as machine-readable, allowing downstream tools to examine the library contents. 

Additionally, a `.hat` file has the convenient property that it is simultaneously a valid H-file and a valid TOML file. Namely, a standard C/C++ compiler will ignore the TOML metadata and see the file as a standard `.h` file. On the other hand, the `.hat` file is a valid TOML file that can be parsed with any standard TOML parser. 

# What problem does the HAT format attempt to solve? 

C/C++ are among the most popular programming languages, and the low-level control offered by C/C++ make them especially attractive for performance-critical applications. However, C/C++ also come with serious shortcomings compared to other programming languages. One of these shortcomings is that C/C++ libraries are typically opaque and lack mechanisms for systematic documentation and introspection. 

This idea is best explained with an example. Say that we implement a function in C that performs a highly-optimized in-place column-wise normalization of a 10x10 matrix. Namely, the function takes a 10x10 matrix `A` and divides each column by the Euclidean norm of that column. A highly-optimized implementation of this function would take advantage of the target computer's specific hardware properties, such as the structure of its cache hierarchy, its multiple CPU cores, or perhaps even the presence of GPU cores. The declaration of this function in the H-file would look something like this:
```
void normalize(float* A);
```
The accompanying object file would contain the compiled machine code for our function. This object/header file pair is opaque, as it provides very little information on how the function should be used, its dependencies, and how it was implemented. For example:

* What does the pointer `float* A` point to? By convention, it is reasonable to assume that `A` points to the first element of an array that contains the 100 matrix elements, but this is not explicitly expressed anywhere.
* Does the function expect the matrix elements to appear in row-major order, column-major order, Z-order, or something else?
* What is the size of the array `A`? We may have auxiliary knowledge that the array is 100 elements long, but this information is not given explicitly. 
* We can see that `A` is not `const`, so we know that its elements can be changed by the function, but is it an "output-only" array (its initial values are overwritten) or is it an "input-output" array? We have auxiliary knowledge that `A` is both an input and an output, but this is not given explicitly. 
* Who created this library? It is a newer version of a previous library? Is it distributed under an open-source license?
* Is this function compiled for Windows or Linux? Does it need to be linked to a C runtime library or other library?
* For which instruction set is the function compiled? x86_64? With SSE extensions? AVX? AVX512?
* Does the function assume a fixed number of CPU cores? Does the function implementation rely on GPU hardware?  

Some of the information above is typically provided in unstructured human-readable documentation, provided in H-file comments, in a `README.txt` file, in a `man` manual page, or in a web page that describes the library. Some of the information may be implied by the library name or the function name (e.g., imagine that the function is named "singleCoreNormalize") or by common sense (e.g., if a GPU is not mentioned anywhere, the function probably doesn't use one). Some of these questions may simply remain unanswered. Moreover, none of this information is exposed to downstream programming tools. For example, imagine a downstream tool that executes each function in a library and measures its running time - how would it provide the function with reasonable arguments?

The HAT format attempts to replace this opacity with transparency, by annotating each declared function with descriptive metadata.

# How is a .hat file simultaneously an H-file and a TOML file?

A `.hat` file is simultaneously a valid H-file and a valid TOML file. It tricks the C/C++ parser into only seeing the valid C/C++ parts of the file, while maintaining the structure of a valid TOML file. This is accomplished with the following file sneaky structure:
```
#ifdef TOML

// Add TOML here

[declaration]
    code = '''
#endif // TOML

// Add C/C++ declarations here

#ifdef TOML
    '''
#endif // TOML
```

How does a C/C++ handle this file? Assuming that the `TOML` macro is not defined, the parser ignores everything that appears between `#ifdef` and `#endif`. This leaves whatever is added instead of `// Add C/C++ declarations here`. How does a TOML parser handle this file? First note that `#` is a comment escape character in TOML, so the `#ifdef` and `#endif` lines are simply ignored. Any TOML code that is places instead of `// Add TOML here` is parsed normally. Finally, a special TOML table named `[declaration]` is defined, and in it a key named `code` that contains all of the C/C++ code as a multiline string. Voila!
