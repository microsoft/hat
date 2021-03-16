#!/usr/bin/env python3

# Utility to parse and validate a HAT package

from .hat_file import HATFile

import os

class HATPackage:
    def __init__(self, dirpath):
        assert(os.path.isdir(dirpath))
        self.hat_file_map = {}
        for entry in os.scandir(dirpath):
            if entry.path.endswith(".hat"):
                self.hat_file_map[entry.path] = HATFile.Deserialize(entry.path)

        # Find all referenced link targets and ensure they are also part of the package
        self.link_targets = []
        self.hat_file_to_link_target_mapping = {}
        for hat_file_path in self.hat_file_map:
            hat_file = self.hat_file_map[hat_file_path]
            link_target_path = os.path.join(dirpath, hat_file.dependencies.link_target)
            if not os.path.isfile(link_target_path):
                raise ValueError(f"HAT file {hat_file_path} references link_target {hat_file.dependencies.link_target} which is not part of the HAT package at {dirpath}")
            self.hat_file_to_link_target_mapping[hat_file_path] = link_target_path
            self.link_targets.append(link_target_path)
