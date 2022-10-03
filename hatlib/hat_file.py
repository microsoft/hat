#!/usr/bin/env python3

# Utility to parse the TOML metadata from HAT files
import os
import tomlkit
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List

# TODO : type-checking on leaf node values


def _read_toml_file(filepath):
    path = os.path.abspath(filepath)
    toml_doc = None
    with open(path, "r") as f:
        file_contents = f.read()
        toml_doc = tomlkit.parse(file_contents)
    return toml_doc


def _check_required_table_entry(table, key):
    if key not in table:
        # TODO : add more context to this error message
        raise ValueError(f"Invalid HAT file: missing required key {key}")


def _check_required_table_entries(table, keys):
    for key in keys:
        _check_required_table_entry(table, key)


class ParameterType(Enum):
    AffineArray = "affine_array"
    RuntimeArray = "runtime_array"
    Element = "element"
    Void = "void"


class UsageType(Enum):
    Input = "input"
    Output = "output"
    InputOutput = "input_output"


class CallingConventionType(Enum):
    StdCall = "stdcall"
    CDecl = "cdecl"
    FastCall = "fastcall"
    VectorCall = "vectorcall"
    Device = "devicecall"


class TargetType(Enum):
    CPU = "CPU"
    GPU = "GPU"


class OperatingSystem(Enum):
    Windows = "windows"
    MacOS = "macos"
    Linux = "linux"

    @staticmethod
    def host():
        import platform
        platform_name = platform.system().lower()
        if platform_name == "darwin":
            return OperatingSystem.MacOS
        return OperatingSystem(platform_name)


@dataclass
class AuxiliarySupportedTable:
    AuxiliaryKey = "auxiliary"
    auxiliary: dict = field(default_factory=dict)

    def add_auxiliary_table(self, table):
        if len(self.auxiliary) > 0:
            table.add(self.AuxiliaryKey, self.auxiliary)

    @staticmethod
    def parse_auxiliary(table):
        if AuxiliarySupportedTable.AuxiliaryKey in table:
            return table[AuxiliarySupportedTable.AuxiliaryKey]
        else:
            return {}


@dataclass
class Description(AuxiliarySupportedTable):
    TableName: str = "description"
    comment: str = ""
    author: str = ""
    version: str = ""
    license_url: str = ""

    def to_table(self):
        description_table = tomlkit.table()
        description_table.add("comment", self.comment)
        description_table.add("author", self.author)
        description_table.add("version", self.version)
        description_table.add("license_url", self.license_url)

        self.add_auxiliary_table(description_table)

        return description_table

    @staticmethod
    def parse_from_table(table):
        return Description(
            author=table["author"],
            version=table["version"],
            license_url=table["license_url"],
            auxiliary=AuxiliarySupportedTable.parse_auxiliary(table)
        )


@dataclass
class Parameter:
    # All parameter keys
    name: str = ""
    description: str = ""
    logical_type: ParameterType = None
    declared_type: str = ""
    element_type: str = ""
    usage: UsageType = None

    # Affine array parameter keys
    shape: list = field(default_factory=list)
    affine_map: list = field(default_factory=list)
    affine_offset: int = -1

    # Runtime array parameter keys
    size: str = ""

    def to_table(self):
        table = tomlkit.inline_table()
        table.append("name", self.name)
        table.append("description", self.description)
        table.append("logical_type", self.logical_type.value)
        table.append("declared_type", self.declared_type)
        table.append("element_type", self.element_type)
        table.append("usage", self.usage.value)

        if self.logical_type == ParameterType.AffineArray:
            table.append("shape", self.shape)
            table.append("affine_map", self.affine_map)
            table.append("affine_offset", self.affine_offset)
        elif self.logical_type == ParameterType.RuntimeArray:
            table.append("size", self.size)

        return table

    # TODO : change "usage" to "role" in schema
    @staticmethod
    def parse_from_table(param_table):
        required_table_entries = ["name", "description", "logical_type", "declared_type", "element_type", "usage"]
        _check_required_table_entries(param_table, required_table_entries)
        affine_array_required_table_entries = ["shape", "affine_map", "affine_offset"]
        runtime_array_required_table_entries = ["size"]

        name = param_table["name"]
        description = param_table["description"]
        logical_type = ParameterType(param_table["logical_type"])
        declared_type = param_table["declared_type"]
        element_type = param_table["element_type"]
        usage = UsageType(param_table["usage"])

        param = Parameter(
            name=name,
            description=description,
            logical_type=logical_type,
            declared_type=declared_type,
            element_type=element_type,
            usage=usage
        )

        if logical_type == ParameterType.AffineArray:
            _check_required_table_entries(param_table, affine_array_required_table_entries)
            param.shape = param_table["shape"]
            param.affine_map = param_table["affine_map"]
            param.affine_offset = param_table["affine_offset"]

        elif logical_type == ParameterType.RuntimeArray:
            _check_required_table_entries(param_table, runtime_array_required_table_entries)
            param.size = param_table["size"]

        return param

    @staticmethod
    def void():
        return Parameter(
            logical_type=ParameterType.Void,
            declared_type="void",
            element_type="void",
            usage=UsageType.Output
        )

@dataclass
class Function(AuxiliarySupportedTable):
    # required
    arguments: List[Parameter] = field(default_factory=list)
    calling_convention: CallingConventionType = None
    description: str = ""
    hat_file: any = None
    link_target: Path = None
    name: str = ""
    return_info: Parameter = None

    # optional
    launch_parameters: list = field(default_factory=list)
    dynamic_shared_mem_bytes: int = 0
    launches: str = ""
    provider: str = ""
    runtime: str = ""

    def to_table(self):
        table = tomlkit.table()
        table.add("name", self.name)
        table.add("description", self.description)
        table.add("calling_convention", self.calling_convention.value)
        arg_tables = [arg.to_table() for arg in self.arguments]
        arg_array = tomlkit.array()
        for arg_table in arg_tables:
            arg_array.append(arg_table)
        table.add(
            "arguments", arg_array
        )    # TODO : figure out why this isn't indenting after serialization in some cases

        if self.launch_parameters:
            table.add("launch_parameters", self.launch_parameters)

        if self.dynamic_shared_mem_bytes:
            table.add("dynamic_shared_mem_bytes", self.dynamic_shared_mem_bytes)

        if self.launches:
            table.add("launches", self.launches)

        if self.provider:
            table.add("provider", self.provider)

        if self.runtime:
            table.add("runtime", self.runtime)

        table.add("return", self.return_info.to_table())

        self.add_auxiliary_table(table)

        return table

    @staticmethod
    def parse_from_table(function_table):
        required_table_entries = ["name", "description", "calling_convention", "arguments", "return"]
        _check_required_table_entries(function_table, required_table_entries)
        arguments = [Parameter.parse_from_table(param_table) for param_table in function_table["arguments"]]

        launch_parameters = function_table["launch_parameters"] if "launch_parameters" in function_table else []

        dynamic_shared_mem_bytes = function_table["dynamic_shared_mem_bytes"] if "dynamic_shared_mem_bytes" in function_table else 0

        launches = function_table["launches"] if "launches" in function_table else ""

        provider = function_table["provider"] if "provider" in function_table else ""

        runtime = function_table["runtime"] if "runtime" in function_table else ""

        return_info = Parameter.parse_from_table(function_table["return"])

        return Function(
            name=function_table["name"],
            description=function_table["description"],
            calling_convention=CallingConventionType(function_table["calling_convention"]),
            arguments=arguments,
            return_info=return_info,
            launch_parameters=launch_parameters,
            dynamic_shared_mem_bytes=dynamic_shared_mem_bytes,
            launches=launches,
            provider=provider,
            runtime=runtime,
            auxiliary=AuxiliarySupportedTable.parse_auxiliary(function_table)
        )


class FunctionTableCommon:

    def __init__(self, function_map):
        self.function_map = function_map

    def to_table(self):
        func_table = tomlkit.table()
        for function_key in self.function_map:
            func_table.add(function_key, self.function_map[function_key].to_table())
        return func_table

    @classmethod
    def parse_from_table(cls, all_functions_table):
        function_map = {
            function_key: Function.parse_from_table(all_functions_table[function_key])
            for function_key in all_functions_table
        }
        return cls(function_map)


class FunctionTable(FunctionTableCommon):
    TableName = "functions"


class DeviceFunctionTable(FunctionTableCommon):
    TableName = "device_functions"


@dataclass
class Target:

    @dataclass
    class Required:

        @dataclass
        class CPU:
            TableName = TargetType.CPU.value

            # required
            architecture: str = ""
            extensions: list = field(default_factory=list)

            # optional
            runtime: str = ""

            def to_table(self):
                table = tomlkit.table()
                table.add("architecture", self.architecture)
                table.add("extensions", self.extensions)

                if self.runtime:
                    table.add("runtime", self.runtime)

                return table

            @staticmethod
            def parse_from_table(table):
                required_table_entries = ["architecture", "extensions"]
                _check_required_table_entries(table, required_table_entries)

                runtime = table.get("runtime", "")

                return Target.Required.CPU(
                    architecture=table["architecture"], extensions=table["extensions"], runtime=runtime
                )

        @dataclass
        class GPU:
            TableName = TargetType.GPU.value
            blocks: int = 0
            instruction_set_version: str = ""
            min_threads: int = 0
            min_global_memory_KB: int = 0
            min_shared_memory_KB: int = 0
            min_texture_memory_KB: int = 0
            model: str = ""
            runtime: str = ""

            def to_table(self):
                table = tomlkit.table()
                table.add("model", self.model)
                table.add("runtime", self.runtime)
                table.add("blocks", self.blocks)
                table.add("instruction_set_version", self.instruction_set_version)
                table.add("min_threads", self.min_threads)
                table.add("min_global_memory_KB", self.min_global_memory_KB)
                table.add("min_shared_memory_KB", self.min_shared_memory_KB)
                table.add("min_texture_memory_KB", self.min_texture_memory_KB)

                return table

            @staticmethod
            def parse_from_table(table):
                required_table_entries = [
                    "runtime",
                    "model",
                ]
                _check_required_table_entries(table, required_table_entries)

                return Target.Required.GPU(
                    runtime=table["runtime"],
                    model=table["model"],
                    blocks=table["blocks"],
                    instruction_set_version=table["instruction_set_version"],
                    min_threads=table["min_threads"],
                    min_global_memory_KB=table["min_global_memory_KB"],
                    min_shared_memory_KB=table["min_shared_memory_KB"],
                    min_texture_memory_KB=table["min_texture_memory_KB"]
                )

        TableName = "required"
        os: OperatingSystem = OperatingSystem.host()
        cpu: CPU = CPU()
        gpu: GPU = None

        def to_table(self):
            table = tomlkit.table()
            table.add("os", self.os.value)
            table.add(Target.Required.CPU.TableName, self.cpu.to_table())
            if self.gpu and self.gpu.runtime:
                table.add(Target.Required.GPU.TableName, self.gpu.to_table())
            return table

        @staticmethod
        def parse_from_table(table):
            required_table_entries = ["os", Target.Required.CPU.TableName]
            _check_required_table_entries(table, required_table_entries)
            cpu_info = Target.Required.CPU.parse_from_table(table[Target.Required.CPU.TableName])
            if Target.Required.GPU.TableName in table:
                gpu_info = Target.Required.GPU.parse_from_table(table[Target.Required.GPU.TableName])
            else:
                gpu_info = Target.Required.GPU()
            return Target.Required(os=table["os"], cpu=cpu_info, gpu=gpu_info)

    # TODO : support optimized_for table
    class OptimizedFor:
        TableName = "optimized_for"

        def to_table(self):
            return tomlkit.table()

        @staticmethod
        def parse_from_table(table):
            pass

    TableName = "target"
    required: Required = Required()
    optimized_for: OptimizedFor = Required()

    def to_table(self):
        table = tomlkit.table()
        table.add(Target.Required.TableName, self.required.to_table())
        if self.optimized_for is not None:
            table.add(Target.OptimizedFor.TableName, self.optimized_for.to_table())
        return table

    @staticmethod
    def parse_from_table(target_table):
        required_table_entries = [Target.Required.TableName]
        _check_required_table_entries(target_table, required_table_entries)
        required_data = Target.Required.parse_from_table(target_table[Target.Required.TableName])
        if Target.OptimizedFor.TableName in target_table:
            optimized_for_data = Target.OptimizedFor.parse_from_table(target_table[Target.OptimizedFor.TableName])
        else:
            optimized_for_data = Target.OptimizedFor()
        return Target(required=required_data, optimized_for=optimized_for_data)


@dataclass
class LibraryReference:
    name: str = ""
    version: str = ""
    target_file: str = ""

    def to_table(self):
        table = tomlkit.inline_table()
        table.append("name", self.name)
        table.append("version", self.version)
        table.append("target_file", self.target_file)
        return table

    @staticmethod
    def parse_from_table(table):
        return LibraryReference(name=table["name"], version=table["version"], target_file=table["target_file"])


@dataclass
class Dependencies(AuxiliarySupportedTable):
    TableName = "dependencies"
    link_target: str = ""
    deploy_files: list = field(default_factory=list)
    dynamic: list = field(default_factory=list)

    def to_table(self):
        table = tomlkit.table()
        table.add("link_target", self.link_target)
        table.add("deploy_files", self.deploy_files)

        dynamic_arr = tomlkit.array()
        for elt in self.dynamic:
            dynamic_arr.append(elt.to_table())
        table.add("dynamic", dynamic_arr)

        self.add_auxiliary_table(table)
        return table

    @staticmethod
    def parse_from_table(dependencies_table):
        required_table_entries = ["link_target", "deploy_files", "dynamic"]
        _check_required_table_entries(dependencies_table, required_table_entries)
        dynamic = [LibraryReference.parse_from_table(lib_ref_table) for lib_ref_table in dependencies_table["dynamic"]]
        return Dependencies(
            link_target=dependencies_table["link_target"],
            deploy_files=dependencies_table["deploy_files"],
            dynamic=dynamic,
            auxiliary=AuxiliarySupportedTable.parse_auxiliary(dependencies_table)
        )


@dataclass
class CompiledWith:
    TableName = "compiled_with"
    compiler: str = ""
    flags: str = ""
    crt: str = ""
    libraries: list = field(default_factory=list)

    def to_table(self):
        table = tomlkit.table()
        table.add("compiler", self.compiler)
        table.add("flags", self.flags)
        table.add("crt", self.crt)

        libraries_arr = tomlkit.array()
        for elt in self.libraries:
            libraries_arr.append(elt.to_table())
        table.add("libraries", libraries_arr)

        return table

    @staticmethod
    def parse_from_table(compiled_with_table):
        required_table_entries = ["compiler", "flags", "crt", "libraries"]
        _check_required_table_entries(compiled_with_table, required_table_entries)
        libraries = [
            LibraryReference.parse_from_table(lib_ref_table) for lib_ref_table in compiled_with_table["libraries"]
        ]
        return CompiledWith(
            compiler=compiled_with_table["compiler"],
            flags=compiled_with_table["flags"],
            crt=compiled_with_table["crt"],
            libraries=libraries
        )


@dataclass
class Declaration:
    TableName = "declaration"
    code: str = ""

    def to_table(self):
        table = tomlkit.table()
        code_str = self.code
        if len(code_str) > 0 and code_str[0] != '\n':
            code_str = "\n" + code_str
        code_str = tomlkit.string(code_str, multiline=True)
        table.add("code", code_str)
        return table

    @staticmethod
    def parse_from_table(declaration_table):
        required_table_entries = ["code"]
        _check_required_table_entries(declaration_table, required_table_entries)
        return Declaration(code=declaration_table["code"])


@dataclass
class HATFile:
    """Encapsulates a HAT file. An instance of this class can be created by calling the
    Deserialize class method e.g.:
        some_hat_file = Deserialize('someFile.hat')
    Similarly, HAT files can be serialized but creating/modifying a HATFile instance
    and then calling Serilize e.g.:
        some_hat_file.name = 'some new name'
        some_hat_file.Serialize(`someFile.hat`)
    """
    name: str = ""
    description: Description = Description()
    _function_table: FunctionTable = FunctionTable({})
    _device_function_table: DeviceFunctionTable = DeviceFunctionTable({})
    functions: list = field(default_factory=list)
    device_functions: list = field(default_factory=list)
    function_map: Dict[str, Function] = field(default_factory=dict)
    device_function_map: Dict[str, Function] = field(default_factory=dict)
    target: Target = Target()
    dependencies: Dependencies = Dependencies()
    compiled_with: CompiledWith = CompiledWith()
    declaration: Declaration = Declaration()
    path: Path = None

    HATPrologue = "\n#ifndef __{0}__\n#define __{0}__\n\n#ifdef TOML\n"
    HATEpilogue = "\n#endif // TOML\n\n#endif // __{0}__\n"

    def __post_init__(self):
        for func in self.functions:
            func.hat_file = self
            func.link_target = Path(self.path).resolve().parent / self.dependencies.link_target

    @property
    def functions(self):
        return self._function_table.function_map.values()

    @functions.setter
    def functions(self, func_list_or_dict):
        if isinstance(func_list_or_dict, property):
            return
        if isinstance(func_list_or_dict, dict):
            name_to_func_map = func_list_or_dict
        else:
            name_to_func_map = { func.name : func for func in func_list_or_dict }
        if self._function_table is None:
            self._function_table = FunctionTable()
        self._function_table.function_map = name_to_func_map

    @property
    def function_map(self):
        return self._function_table.function_map

    @function_map.setter
    def function_map(self, func_map):
        if isinstance(func_map, property):
            return
        self._function_table.function_map = func_map

    @property
    def device_functions(self):
        return self._device_function_table.function_map.values()

    @device_functions.setter
    def device_functions(self, func_list_or_dict):
        if isinstance(func_list_or_dict, property):
            return
        if isinstance(func_list_or_dict, dict):
            name_to_func_map = func_list_or_dict
        else:
            name_to_func_map = { func.name : func for func in func_list_or_dict }
        if self._device_function_table is None:
            self._device_function_table = FunctionTable()
        self._device_function_table.function_map = name_to_func_map

    @property
    def device_function_map(self):
        return self._device_function_table.function_map

    @device_function_map.setter
    def device_function_map(self, func_map):
        if isinstance(func_map, property):
            return
        self._device_function_table.function_map = func_map

    def Serialize(self, filepath=None):
        """Serializes the HATFile to disk using the file location specified by `filepath`.
        If `filepath` is not specified then the object's `path` attribute is used."""
        if filepath is None:
            filepath = self.path
        root_table = tomlkit.table()
        root_table.add(Description.TableName, self.description.to_table())
        root_table.add(FunctionTable.TableName, self._function_table.to_table())
        if self.device_function_map:
            root_table.add(DeviceFunctionTable.TableName, self._device_function_table.to_table())
        root_table.add(Target.TableName, self.target.to_table())
        root_table.add(Dependencies.TableName, self.dependencies.to_table())
        root_table.add(CompiledWith.TableName, self.compiled_with.to_table())
        root_table.add(Declaration.TableName, self.declaration.to_table())
        with open(filepath, "w") as out_file:
            # MSVC does not allow "." in macro definitions
            name = self.name.replace(".", "_")
            out_file.write(self.HATPrologue.format(name))
            out_file.write(tomlkit.dumps(root_table))
            out_file.write(self.HATEpilogue.format(name))

    @staticmethod
    def Deserialize(filepath) -> "HATFile":
        """Creates an instance of A HATFile class by deserializing the contents of the file at `filepath`"""
        hat_toml = _read_toml_file(filepath)
        name = os.path.splitext(os.path.basename(filepath))[0]
        required_entries = [
            Description.TableName, FunctionTable.TableName, Target.TableName, Dependencies.TableName,
            CompiledWith.TableName, Declaration.TableName
        ]
        _check_required_table_entries(hat_toml, required_entries)
        device_function_table = DeviceFunctionTable({})
        if DeviceFunctionTable.TableName in hat_toml:
            device_function_table = DeviceFunctionTable.parse_from_table(hat_toml[DeviceFunctionTable.TableName])
        hat_file = HATFile(
            name=name,
            description=Description.parse_from_table(hat_toml[Description.TableName]),
            _function_table=FunctionTable.parse_from_table(hat_toml[FunctionTable.TableName]),
            _device_function_table=device_function_table,
            target=Target.parse_from_table(hat_toml[Target.TableName]),
            dependencies=Dependencies.parse_from_table(hat_toml[Dependencies.TableName]),
            compiled_with=CompiledWith.parse_from_table(hat_toml[CompiledWith.TableName]),
            declaration=Declaration.parse_from_table(hat_toml[Declaration.TableName]),
            path=Path(filepath).resolve()
        )
        return hat_file
