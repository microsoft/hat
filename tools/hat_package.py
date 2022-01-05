#!/usr/bin/env python3

# Utility to parse and validate a HAT package

from hat_file import HATFile

import os

class HATPackage:
    def __init__(self, hat_file_path):
        """A HAT Package is defined to be a HAT file and corresponding object file, located in the same directory.
        The object file is specified in the HAT file's link_target attribute.
        The same object file can be referenced by many HAT files.
        Many HAT packages can exist in the same directory.
        An instance of HATPackage is created by giving HATPackage the file path to the .hat file."""
        self.path = Path(dirpath).resolve()
        assert self.path.is_dir()

        self.name = self.path.name
        self.hat_files = [HATFile.Deserialize(hat_file_path) for hat_file_path in self.path.glob("*.hat")]

        # Find all referenced link targets and ensure they are also part of the package
        self.link_targets = []
        for hat_file in self.hat_files:
            link_target_path = self.path / hat_file.dependencies.link_target
            if not os.path.isfile(link_target_path):
                raise ValueError(f"HAT file {hat_file.path} references link_target {hat_file.dependencies.link_target} which is not part of the HAT package at {self.path}")
            self.link_targets.append(link_target_path)

    def get_functions(self):
        functions = []
        for hat_file in self.hat_files:
            functions += hat_file.functions
        return functions

    def get_functions_for_target(self, os: str, arch: str, required_extensions:list = []):
        all_functions = self.get_functions()
        def matches_target(hat_function):
            hat_file = hat_function.hat_file
            if hat_file.target.required.os != os or hat_file.target.required.cpu.architecture != arch:
                return False
            for required_ext in required_extensions:
                if required_ext not in hat_file.target.required.cpu.extensions:
                    return False
            return True

        return list(filter(matches_target, all_functions))
