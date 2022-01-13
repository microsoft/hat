#!/usr/bin/env python3

# Utility to parse and validate a HAT package

from .hat_file import HATFile

import os

class HATPackage:
    def __init__(self, hat_file_path):
        """A HAT Package is defined to be a HAT file and corresponding binary file, located in the same directory.
        The binary file is specified in the HAT file's link_target attribute.
        The same binary file can be referenced by many HAT files.
        Many HAT packages can exist in the same directory.
        An instance of HATPackage is created by giving HATPackage the file path to the .hat file."""
        self.name = os.path.basename(hat_file_path)
        self.hat_file_path = hat_file_path
        self.hat_file = HATFile.Deserialize(hat_file_path)

        self.link_target = self.hat_file.dependencies.link_target
        self.link_target_path =  os.path.join(os.path.split(self.hat_file_path)[0], self.hat_file.dependencies.link_target)
        if not os.path.isfile(self.link_target_path):
            raise ValueError(f"HAT file {self.hat_file_path} references link_target {self.hat_file.dependencies.link_target} which is not found in same directory as HAT file (expecting it to be in {os.path.split(self.hat_file_path)[0]}")        
        self.functions = self.hat_file.functions

    def get_functions(self):
        return self.hat_file.functions

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
