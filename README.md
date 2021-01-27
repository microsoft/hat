![HAT](https://upload.wikimedia.org/wikipedia/commons/8/80/Crystal_Project_wizard.png)

# HAT file format

HAT is a format for distributing compiled libraries in the C programming language. HAT stands for "C **H**eader **A**nnotated with **T**OML", and implies that standard C header files are decorated with useful metadata in the [TOML](https://toml.io/) markup language.

A library in the HAT format typically includes one static library file (`.a` or `.lib`) and one or more `.hat` files. The static library contains all the compiled object code that implements the library functions. Each `.hat` file contains a combination of standard C function declarations (like a typical `.h` file) and metadata in the TOML markup language. The metadata that accompanies each function declaration describes how the function should be called and how it was implemented. The metadata is intended to be both human-readable and machine-readable, providing structured and systematic documentation and allowing downstream tools to examine the library contents. 

Each `.hat` file has the convenient property that it is simultaneously a valid h-file and a valid TOML file. In other words, the file is structured such that a C compiler will ignore the TOML metadata, while a TOML parser will understand the entire file as a valid TOML file. We accomplish this using a technique we call *the hat trick*, which is explained below. 

# What problem does the HAT format solve? 

C is among the most popular programming languages, but it also has serious shortcomings. In particular, C libraries are typically opaque and lack mechanisms for systematic documentation and introspection. This is best explained with an example: 

Say that we use C to implement an in-place column-wise normalization of a 10x10 matrix. In other words, this function takes a 10x10 matrix `A` and divides each column by the Euclidean norm of that column. A highly-optimized implementation of this function would be tailored to the target computer's specific hardware properties, such as its cache size, the number of CPU cores, and perhaps even the presence of a GPU. The declaration of this function in an h-file would look something like this:
```
void normalize(float* A);
```
The accompanying object file would contain the compiled machine code for this function. The object/header file pair is opaque, as it provides very little information on how the function should be used, its dependencies, and how it was implemented. Specifically:

* What does the pointer `float* A` point to? By convention, it is reasonable to assume that `A` points to the first element of an array that contains the 100 matrix elements, but this is not stated explicitly.
* Does the function expect the matrix elements to appear in row-major order, column-major order, Z-order, or something else?
* What is the size of the array `A`? We may have auxiliary knowledge that the array is 100 elements long, but this information is not stated explicitly. 
* We can see that `A` is not `const`, so we know that its elements can be changed by the function, but is it an "output-only" array (its initial values are overwritten) or is it an "input/output" array? We have auxiliary knowledge that `A` is both an input and an output, but this is not stated explicitly. 
* Is this function compiled for Windows or Linux? 
* Does it need to be linked to a C runtime library or any other library?
* For which instruction set is the function compiled? Does it rely on SSE extensions? AVX? AVX512?
* Is this a multi-threaded implementation? Does the implementation assume a fixed number of CPU cores? 
* Does the implementation rely on GPU hardware?   
* Who created this library? Does it have a version number? Is it distributed under an open-source license?

Some of the questions above can be answered by reading the human-readable documentation provided in h-file comments, in `README.txt` or `LICENSE.txt` files, or in a web page that describes the library. Some of the information may be implied by the library name or the function name (e.g., imagine that the function was named "normalize_10x10_singlecore") or by common sense (e.g., if a GPU is not mentioned anywhere, the function probably doesn't require one). Nevertheless, C does not have a schematized systematic way to express all of this important information. Moreover, human-readable documentation does not expose this information to downstream programming tools. For example, imagine a downstream tool that examines a library and automatically creates tests that measure the performance of each function. 

The HAT library format attempts to replace this opacity with transparency, by annotating each declared function with descriptive metadata in TOML.

# The HAT trick

As mentioned above, the `.hat` file is simultaneously a valid h-file and a valid TOML file. It tricks the C parser into only seeing the valid C parts of the file, while maintaining the structure of a valid TOML file. This is accomplished with the following file structure:
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

What does a C compiler see? Assuming that the `TOML` macro is not defined, the parser ignores everything that appears between `#ifdef TOML` and `#endif`. This leaves whatever appears instead of `// Add C declarations here`. 

What does a TOML parser see? First note that `#` is a comment escape character in TOML, so the `#ifdef` and `#endif` lines are ignored as comments. Any TOML code that appears instead of `// Add TOML here` is parsed normally. Finally, a special TOML table named `[declaration]` is defined, and inside it a key named `code` with all of the C declarations as a multiline string.

Why is it important for the TOML and the C declarations to live in the same file? Why not put the TOML metadata in a separate file? The fact that C already splits the library code between object files and h-files is already a concern, because the user has to worry about distributing the an hfile with an incorrect version of the `.lib` file. We don't want to make things worse by adding yet another separate file. Keeping the metadata in the same file as the function declaration ensures that each declaration is never separated from its metadata. 

# Multiple `.hat` files

The HAT format defines different types of metadata. Some types of metadata (such as the targeted operating system and the license) apply to all of the functions in the file, while other types of metadata (such as the specification of input and output arguments) are defined per function. Therefore, if two functions require different metadata, and that type of metadata is defined at the file scope, then those functions must be declared in separate `.hat` files. For example, if the first function is compiled with AVX512 instructions and the second function makes do with SSE instructions, then these functions must be defined in separate `.hat` files. On the other hand, both functions can co-exist in a single `.lib` file (recall that a HAT library has a single `.lib` file).

The constraint above influences the number of `.hat` files in a library. Additionally, the author of the library may decide to split functions up into separate `.hat` files to improve readability or organization.

# HAT schema

The TOML metadata in each `.hat` file follows the [TOML schema](https://github.com/brunoborges/toml-schema), defined in [`schema/hat.tosd`](/schema/hat.tosd). `.hat` file samples can be found in the [samples directory](/samples).

