# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Module for inspecting PyTorch models and tracking their execution.

This module is used to inspect PyTorch models and track their execution. It is
used to find the modules that are executed and the modules that are not
executed. It is also used to wrap the forward methods of the modules to track
the execution time and input/output types.

In ModuleInspector, we use `vars(obj)` to get the members of an object.

Double references to the same module may cause issues. The inspector takes the
first reference to the module that is inspected. So, even if the second
reference is used for inference, the first will be returned for wrapping.

Object paths used in ModuleInfo are relative to the root module and are in all
dot notation, even for dictionaries and lists. (e.g., '.list.0.layers`,
'.dict.key.layers`, `.encoder.layers.0.self_attn.q_proj.weight`).

The depth of a module is the number of executed modules above it in the object
tree, see `ModuleInfo.depth`. Containers such as lists, dicts, `nn.ModuleList`,
and modules whose forward never ran do not count.
The walk stops at a path depth calculated from `depth`.
Modules past that path are not registered.

Environment variables that could be used:

    TORCH_TWEAK_INSPECT_DEBUG:
        Whether to enable more verbose debug mode for inspecting. Adds visited
        nodes, and execution order. Default is False.

    TORCH_TWEAK_INSPECT_DEBUG_RAISE:
        Whether to raise an error when an error occurs during inspecting -
        ignored by default. Default is False.

"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache, wraps
from logging import getLogger
from typing import Any

import torch

from torch_tweak.torch.config import get_bool_env_variable
from torch_tweak.torch.inspecting.module_info import (
    DictOfModulesInfo,
    ListOfModulesInfo,
    ModuleInfo,
    ObjectOfModulesInfo,
)
from torch_tweak.torch.utils.xpu import synchronize

logger = getLogger(__name__)

EXTRA_PATH_SEGMENTS = 4

DEFAULT_INSPECT_DEBUG = False

DEFAULT_INSPECT_DEBUG_RAISE = False


@dataclass
class InspectContext:
    """Context for inspecting modules."""

    path_depth: int = 0
    object_path: str = ""
    module_parent: ModuleInfo | None = None

    @property
    def name(self) -> str:
        """The name."""
        return self.object_path.split(".")[-1]

    def create_module_info(self, module: torch.nn.Module | list | dict | Any) -> ModuleInfo:
        """Get the ModuleInfo based on current context and provided object type."""
        cls = ObjectOfModulesInfo
        if isinstance(module, torch.nn.Module):
            cls = ModuleInfo
        elif isinstance(module, list):
            cls = ListOfModulesInfo
        elif isinstance(module, dict):
            cls = DictOfModulesInfo

        return cls(
            name=self.name or module.__class__.__name__,
            module=module,
            parent=self.module_parent,
            object_path=self.object_path,
        )

    def next(self, name: str = "", parent: ModuleInfo | None = None) -> "InspectContext":
        """Get the next inspect context - increment depth and add name to object path."""
        return InspectContext(
            path_depth=self.path_depth + 1,
            object_path=f"{self.object_path}.{name}" if name else self.object_path,
            module_parent=parent,
        )

    def clone(self, **kwargs) -> "InspectContext":
        """Get the inspect context, with optional changes."""
        return replace(self, **kwargs)


class ModuleInspector:
    """Class for inspecting PyTorch modules and tracking their execution."""

    def __init__(self, depth: int = 0):
        """Initialize the module inspector.

        Args:
            depth: Depth at which `get_modules` cuts the model.
        """
        if depth < 0:
            raise ValueError(f"Expected depth >= 0, got {depth}")
        self._module_info: dict[Any, ModuleInfo] = {}
        self._original_forward: dict[Any, Any] = {}
        self._inspected_objects: set = set()  # Track objects that have been inspected
        self._depth = depth
        # Depth is known only after execution, so traversal is bounded by the object path.
        # One depth level allows two path segments: the module and a container, as in `layers.0`.
        # EXTRA_PATH_SEGMENTS covers further containers and modules that never run.
        self._max_path_depth = 2 * (depth + 1) + EXTRA_PATH_SEGMENTS
        self._stopped_at_path_limit = False
        self._execution_parent_module: list[ModuleInfo] = []  # used for logging the execution path only
        if _get_inspect_debug():
            self._debug_inspecting_functions()

    def inspect(self, obj: Any) -> None:
        """Inspect an object and its members for PyTorch modules.

        Args:
            obj: The object to inspect
        """
        self.reset()
        self._inspect_object(obj, context=InspectContext())

    def reset(self) -> None:
        """Reset the inspector state."""
        # Restore original forward methods
        for module, original_forward in self._original_forward.items():
            module.forward = original_forward

        self._inspected_objects.clear()  # Clear the set when starting a new inspection

        self._module_info.clear()
        self._original_forward.clear()
        self._stopped_at_path_limit = False

    def get_modules(self) -> list[ModuleInfo]:
        """Get executed modules that cut the model at `depth`.

        The result holds every executed module at `depth`, plus every shallower executed module
        without executed children. No returned module is nested in another one.

        Returns:
            Modules in inspection order.
        """
        executed = [m for m in self._module_info.values() if m.forward_called]
        candidates = [m for m in executed if m.depth <= self._depth]
        parents = {id(m.executed_parent) for m in candidates if m.executed_parent is not None}
        return [m for m in candidates if id(m) not in parents]

    def _inspect_object(self, obj: Any, *, context: InspectContext) -> None:
        """Inspect an object and its members for PyTorch modules.

        Args:
            obj: The object to inspect
            context: The context of the inspection
        """
        # Check recursion depth
        if context.path_depth > self._max_path_depth:
            self._stopped_at_path_limit = True
            return

        # Skip Python built-in objects
        if obj.__class__.__module__ == "builtins":
            if not isinstance(obj, dict | list):
                return

        # Skip if object has already been inspected
        obj_id = id(obj)
        if obj_id in self._inspected_objects:
            return

        self._inspected_objects.add(obj_id)

        # Check if object is a PyTorch module
        try:
            if isinstance(obj, torch.nn.Module):
                self._inspect_module(obj, context=context)
                return
        except Exception as e:
            self._debug_error("Failed to inspect module %s", context.object_path, error=e)

        # Handle dictionaries
        if isinstance(obj, dict):
            self._inspect_dict(obj, context=context)
            return

        # Handle lists
        if isinstance(obj, list):
            self._inspect_list(obj, context=context)
            return

        # Handle objects
        if hasattr(obj, "__dict__"):
            self._inspect_members(obj, members={}, context=context.clone(module_parent=context.create_module_info(obj)))
            return

    def _inspect_dict(self, obj: dict, *, context: InspectContext) -> None:
        """Inspect the values of a dictionary.

        Args:
            obj: The dictionary to inspect
            context: The context of the inspection
        """
        new_parent = context.create_module_info(obj)
        for key, value in obj.items():
            try:
                self._inspect_object(value, context=context.next(key, parent=new_parent))
            except Exception as e:
                self._debug_error("Failed to inspect dict[%s] of %s", key, context.object_path, error=e)

    def _inspect_list(self, obj: list, *, context: InspectContext) -> None:
        """Inspect the elements of a list.

        Args:
            obj: The list to inspect
            context: The context of the inspection
        """
        new_parent = context.create_module_info(obj)
        for i, item in enumerate(obj):
            try:
                self._inspect_object(item, context=context.next(str(i), parent=new_parent))
            except Exception as e:
                self._debug_error("Failed to inspect list[%d] of %s", i, context.object_path, error=e)

    def _inspect_members(self, obj: Any, members: dict[str, Any], *, context: InspectContext) -> None:
        """Inspect the members of an object.

        Args:
            obj: The object to inspect
            members: The members of the object to inspect
            context: The context of the inspection
        """
        for name in vars(obj).keys():
            if self._should_skip_member(obj, name):
                continue

            try:
                member = getattr(obj, name)
                if name not in members:
                    members[name] = member
            except Exception as e:
                self._debug_error("Failed to inspect member %s of %s", name, context.object_path, error=e)

        for name, member in members.items():
            self._inspect_object(member, context=context.next(name, parent=context.module_parent))

    def _should_skip_member(self, obj: Any, member_name: str) -> bool:
        """Check if a member should be skipped."""
        if member_name.startswith("__"):
            return True

        # check if name is nn.Module default name, if so, skip
        if isinstance(obj, torch.nn.Module):
            if member_name in ModuleInspector.nn_module_base_members():
                return True

        return False

    def _inspect_module(self, module: torch.nn.Module, *, context: InspectContext) -> None:
        """Start inspecting a module and its submodules.

        Args:
            module: The PyTorch module to inspect
            context: The context of the inspection
        """
        parent: ModuleInfo | None = None
        if module not in self._module_info:
            parent = self._register_module(module, context=context)
        else:
            parent = self._module_info[module]  # it is dead code, check it and remove if not needed

        self._inspect_members(
            module,
            members=dict(module.named_children()),  # it can be simplified by named_modules
            context=context.clone(module_parent=parent),
        )

    def _register_module(self, module: torch.nn.Module, *, context: InspectContext) -> ModuleInfo | None:
        """Register a module and its submodules in the inspector.

        Args:
            module: The module to register
            parent: The parent module info if any
            context: The context of the inspection
        """
        module_info = context.create_module_info(module)
        self._module_info[module] = module_info

        self._wrap_forward_methods(module)

        return module_info

    def _wrap_forward_methods(self, module) -> None:
        """Wrap forward methods of a module and its submodules.

        Args:
            module: The module whose forward method to wrap
        """
        if module in self._original_forward:  # dead code, check it and remove if not needed
            return

        original_forward = module.forward
        self._original_forward[module] = original_forward

        def _debug_run(module_info: ModuleInfo, exec_parent: ModuleInfo, depth: int = 0):
            if _get_inspect_debug():
                logger.debug("%-75s from %s", " " * depth * 2 + module_info.object_path + "()", exec_parent.object_path)

        def wrapped_forward(*args, **kwargs):
            module_info = self._module_info[module]

            execution_parent_module = ModuleInfo(name="root", module=None, object_path="root")
            if self._execution_parent_module:
                execution_parent_module = self._execution_parent_module[-1]

            self._execution_parent_module.append(module_info)

            _debug_run(module_info, execution_parent_module, len(self._execution_parent_module))
            module_info.forward_called = True
            module_info.execution_count += 1

            # Record input types
            input_info = {
                "args": [type(arg).__name__ for arg in args],
                "kwargs": {k: type(v).__name__ for k, v in kwargs.items()},
            }
            module_info.input_types.append(input_info)

            # Synchronize xpu before starting timing
            synchronize()

            # Measure execution time
            start_time = time.perf_counter()
            output = original_forward(*args, **kwargs)

            # Synchronize xpu after execution
            synchronize()

            # Collect end time
            end_time = time.perf_counter()

            # Update execution time
            module_info.total_execution_time += end_time - start_time

            # Record output type
            if isinstance(output, (tuple, list)):
                output_info = {"type": type(output).__name__, "elements": [type(x).__name__ for x in output]}
            else:
                output_info = {"type": type(output).__name__}
            module_info.output_types.append(output_info)

            self._execution_parent_module.pop()

            return output

        module.forward = wrapped_forward

    @cache
    @staticmethod
    def nn_module_base_members() -> list[str]:
        """Get the base class members of a PyTorch module.

        Returns:
            List of names
        """
        return list(dir(torch.nn.Module()))

    def _wrap_debug_inspect(self, method: Callable, collection: bool = False) -> Callable:
        """Wrap a method to add debug logging."""

        @wraps(method)
        def inspect_debug(*args, **kwargs):
            context = kwargs["context"]
            indent = "│" * context.path_depth
            logger.debug("%s%s(%d)%s for %s", indent, "├", context.path_depth, method.__name__, context.object_path)
            try:
                return method(*args, **kwargs)
            finally:
                if collection:
                    logger.debug("%s└", indent)

        return inspect_debug

    def _debug_error(self, message: str, *args: Any, error: Exception) -> None:
        """Debug an error."""
        if _get_inspect_debug():
            logger.debug(message, *args)
        if _get_inspect_debug_raise():
            raise error

    def _debug_inspecting_functions(self) -> None:
        """Debug inspecting functions."""
        logger.debug("Enabled debug mode for inspecting.")
        logger.debug("Wrapping inspecting functions with debug logging.")
        self._inspect_object = self._wrap_debug_inspect(self._inspect_object, collection=True)
        self._inspect_dict = self._wrap_debug_inspect(self._inspect_dict, collection=True)
        self._inspect_list = self._wrap_debug_inspect(self._inspect_list, collection=True)
        self._inspect_members = self._wrap_debug_inspect(self._inspect_members)
        self._inspect_module = self._wrap_debug_inspect(self._inspect_module)
        self._register_module = self._wrap_debug_inspect(self._register_module)


def _get_inspect_debug() -> bool:
    return get_bool_env_variable("TORCH_TWEAK_INSPECT_DEBUG", DEFAULT_INSPECT_DEBUG)


def _get_inspect_debug_raise() -> bool:
    return get_bool_env_variable("TORCH_TWEAK_INSPECT_DEBUG_RAISE", DEFAULT_INSPECT_DEBUG_RAISE)
