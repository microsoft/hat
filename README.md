![HAT](https://upload.wikimedia.org/wikipedia/commons/8/80/Crystal_Project_wizard.png)

# HAT file format

HAT is a format for distributing compiled libraries of functions in the C programming languages. HAT stands for "C **H**eader **A**nnotated with **T**OML", and implies that we decorate standard C header files with useful metadata in the [TOML](https://toml.io/) markup language. 

A function library in the HAT format typically includes one static library file (`.a` or `.lib`) and one or more `.hat` files. The static library contains all the compiled object code that implements the functions in the library. Each of the `.hat` files contains a combination of standard C function declarations (like a typical C `.h` file) and metadata in the TOML markup language. 

The metadata that accompanies each function describes how it should be called and how it was implemented. The metadata is intended to be human-readable, providing structured and systematic documentation, as well as machine-readable, allowing downstream tools to examine the library contents. 

Each `.hat` file has the convenient property that it is simultaneously a valid C h-file and a valid TOML file. We accomplish this using a technique we call *the hat trick*, which is explained below. Namely, a standard C compiler will ignore the TOML metadata and see the file as a standard `.h` file, while a TOML parser will understand the entire file as a valid TOML file.

# What problem does the HAT format attempt to solve? 

C is among the most popular programming languages, but it also has some serious shortcomings compared to other programming languages. One of those is that C libraries are typically opaque and lack mechanisms for systematic documentation and introspection. This is best explained with an example. Say that we implement a function in C that performs a highly-optimized in-place column-wise normalization of a 10x10 matrix. Namely, the function takes a 10x10 matrix `A` and divides each column by the Euclidean norm of that column. A highly-optimized implementation of this function would take advantage of the target computer's specific hardware properties, such as the structure of its cache hierarchy, its multiple CPU cores, or perhaps even the presence of GPU cores. The declaration of this function in the h-file would look something like this:
```
void normalize(float* A);
```
The accompanying object file would contain the compiled machine code for our function. This object/header file pair is opaque, as it provides very little information on how the function should be used, its dependencies, and how it was implemented. For example:

* What does the pointer `float* A` point to? By convention, it is reasonable to assume that `A` points to the first element of an array that contains the 100 matrix elements, but this is not explicitly expressed anywhere.
* Does the function expect the matrix elements to appear in row-major order, column-major order, Z-order, or something else?
* What is the size of the array `A`? We may have auxiliary knowledge that the array is 100 elements long, but this information is not given explicitly. 
* We can see that `A` is not `const`, so we know that its elements can be changed by the function, but is it an "output-only" array (its initial values are overwritten) or is it an "input-output" array? We have auxiliary knowledge that `A` is both an input and an output, but this is not given explicitly. 
* Who created this library? Is it a newer version of a previous library? Is it distributed under an open-source license?
* Is this function compiled for Windows or Linux? Does it need to be linked to a C runtime library or other library?
* For which instruction set is the function compiled? x86_64? With SSE extensions? AVX? AVX512?
* Does the function assume a fixed number of CPU cores? Does the function implementation rely on GPU hardware?

Some of the information above is typically provided in unstructured human-readable documentation, provided in h-file comments, in a `README.txt` file, in a `man` manual page, or in a web page that describes the library. Some of the information may be implied by the library name or the function name (e.g., imagine that the function is named "singleCoreNormalize") or by common sense (e.g., if a GPU is not mentioned anywhere, the function probably doesn't use one). Some of these questions may simply remain unanswered. Moreover, none of this information is exposed to downstream programming tools. For example, imagine a downstream tool that executes each function in a library and measures its running time - how would it provide the function with reasonable arguments?

The HAT library format attempts to replace this opacity with transparency, by annotating each declared function with descriptive metadata in TOML.

# The HAT trick

A `.hat` file is simultaneously a valid h-file and a valid TOML file. It tricks the C parser into only seeing the valid C parts of the file, while maintaining the structure of a valid TOML file. This is accomplished with the following file structure:
```
#ifdef TOML

// Add TOML here

[declaration]
    code = '''
#endif // TOML

// Add C declarations here

#ifdef TOML
    '''
#endif // TOML
```

How does a C compiler handle this file? Assuming that the `TOML` macro is not defined, the parser ignores everything that appears between `#ifdef TOML` and `#endif`. This leaves whatever is added instead of `// Add C declarations here`. 

How does a TOML parser handle this file? First note that `#` is a comment escape character in TOML, so the `#ifdef` and `#endif` lines are simply ignored. Any TOML code that is places instead of `// Add TOML here` is parsed normally. Finally, a special TOML table named `[declaration]` is defined, and in it a key named `code` that contains all of the C code as a multiline string.

Why is it important for the TOML and the C declarations to live in the same file? Why not put the metadata in a separate file? The fact that C already splits the library code between object files and h-files is already a concern, because the user has to worry about distributing a `.hat` file with an incorrect `.lib` file (just like in C). We don't want to make things worse by adding yet another separate file. Keeping the metadata in the same file as the function declaration ensures that each function declaration is never separated from its metadata. 

# Function name

The HAT format does not specify how functions should be named. In particular, the HAT format replies on the function name to match a function declaration with its metadata. Since HAT files are really C based headers, function overloading is not supported. All function names must be globally unique within the final binary. If multiple functions were to share the same name, the mapping between declarations and metadata would become ambiguous. 

# How many files in a library?

As mentioned above, a library in the HAT format *usually* includes a single static library file, but support for multiple static libraries within a HAT library is available. In many situations, a single static library can contain code that targets different hardware instruction sets. For example, the static library can contain a function that uses AVX512 instructions, another function that only uses AVX instructions, and a third function that makes do with SSE instructions.

A library in the HAT format can contain multiple `.hat` files, just like multiple `.h` files can correspond to a single object file in C. However, each `.hat` file can only contain functions that are compiled for the same target. This is because the metadata that describes the hardware target is defined for an entire `.hat` file, and not per function. More generally, any metadata that is defined at the file level in a `.hat` file applies to all the functions declared in that file, which could influence how functions are split among different `.hat` files.

# .hat file schema

The TOML metadata in each `.hat` file follows a certain TOML schema, defined in XXX. TODO
