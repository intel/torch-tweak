# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Locator for an object in a nested structure."""

import copy
import dataclasses
import itertools
from collections.abc import Generator
from enum import IntEnum
from functools import cache
from typing import Any, ClassVar

import torch


class ObjectType(IntEnum):
    """Object type."""

    SEQUENCE = 0
    DICT = 1
    DATACLASS = 2
    USER_TYPE = 3


def _validate_accessor(accessor: object, obj_type: ObjectType) -> None:
    """Validate an accessor against the container type that consumes it."""
    if obj_type == ObjectType.SEQUENCE:
        valid = type(accessor) is int
        expected = "int"
    elif obj_type == ObjectType.DICT:
        valid = type(accessor) in (int, str)
        expected = "int or str"
    else:
        valid = type(accessor) is str
        expected = "str"
    if not valid:
        raise ValueError(
            f"Invalid {Locator.__name__} accessor for {obj_type.name}: expected {expected}, got {accessor!r}"
        )


@cache
def get_class_data_fields_cached(cls):
    """Inspects class ONCE. Returns a PRE-SORTED tuple."""
    class_fields = []
    for attr in dir(cls):
        if attr.startswith("_"):
            continue

        # This lookup is the expensive part we want to avoid repeating
        value = getattr(cls, attr, None)

        if isinstance(value, property) or callable(value):
            continue
        class_fields.append(attr)

    # Sort once here, so we don't have to sort later for simple objects
    return tuple(class_fields)


@cache
def get_class_fields_set_cached(cls):
    """Returns a set of class fields for fast lookup."""
    return set(get_class_data_fields_cached(cls))


def get_object_fields(obj):
    """Returns SORTED public data fields."""
    # 1. Get pre-calculated, pre-sorted fields for this Class
    # We create a new list from the cached one to avoid modifying the cache
    cls_fields = get_class_data_fields_cached(type(obj))

    # 2. Check for dynamic instance variables (rare in strict schemas, common in loose ones)
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict:
        cls_fields_set = get_class_fields_set_cached(type(obj))

        new_fields = [attr for attr in obj_dict if not attr.startswith("_") and attr not in cls_fields_set]

        if new_fields:
            return sorted(itertools.chain(cls_fields, new_fields))

    return cls_fields


class Locator:
    """A path-based locator for navigating and manipulating nested data structures.

    The Locator class provides mechanisms to identify and access elements (such as Python primitives,
    torch tensors, etc.) within nested structures-including sequences, dictionaries, dataclasses,
    and custom user types. It maintains a path that traverses the structure, allowing retrieval
    and mutation of values at those paths, as well as traversing nested contents and supporting
    custom types.

    The path is stored as a sequence of (accessor, ObjectType) tuples. The accessor is either
    an integer index (for sequences) or a string key/attribute name (for dicts, dataclasses,
    or user types).

    The primary entry point for this class is the `find_leaves` method, which yields all leaves
    found within a nested structure.

    Attributes:
        _user_types: Class-level dict of registered user-defined types that should be traversed like dataclasses.
        If the value is True, only tensors are returned, otherwise all objects are returned.
        _ignored_types: Class-level set of types that should not be traversed nor returned by find_leaves.

    """

    _user_types: ClassVar[dict[type, bool]] = dict[type, bool]()
    _ignored_types: ClassVar[set[type]] = set()

    __slots__ = ("_path",)

    def __init__(self, path: tuple[tuple[int | str, ObjectType], ...]):
        """Initialize the locator.

        Args:
            path: Tuple of (accessor, ObjectType) tuples defining the path
                through a nested structure. The accessor is an int for sequences
                or a str for dicts/dataclasses/user types.
        """
        self._path = path  # make it immutable

    @property
    def depth(self) -> int:
        """The depth of the locator."""
        return len(self._path)

    @property
    def leaf_name(self) -> int | str:
        """The leaf of the locator."""
        return self._path[-1][0]

    @property
    def root_name(self) -> int | str:
        """The root of the locator."""
        return self._path[0][0] if self._path else ""

    @staticmethod
    @cache
    def _compute_sanitized_name(path: tuple) -> str:
        """Compute the sanitized name for a given path."""
        result = []
        for accessor, obj_type in path:
            if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
                result.append(f".{accessor}")
            else:
                result.append(f"_{accessor}")
        return "".join(result)

    @property
    def sanitized_name(self) -> str:
        """A sanitized string representation for tensor names."""
        return self._compute_sanitized_name(self._path)

    def accessor_at(self, index: int) -> int | str:
        """Get the accessor at the given index."""
        return self._path[index][0]

    def path_iter(self):
        """Yield (accessor, ObjectType) tuples for each step in the path."""
        return iter(self._path)

    def to_dict(self) -> dict[str, Any]:
        """Convert Locator to a dictionary built only from primitive values.

        Raises:
            ValueError: If the path contains an accessor that cannot be represented
                safely or does not match its object type.
        """
        for accessor, obj_type in self._path:
            _validate_accessor(accessor, obj_type)
        return {
            "type": self.__class__.__name__,
            "path": [[accessor, int(obj_type)] for accessor, obj_type in self._path],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Locator":
        """Create Locator from dictionary.

        Raises:
            ValueError: If the dictionary is not a serialized Locator or holds an
                accessor or object type that cannot appear in a valid path.
        """
        if data.get("type") != Locator.__name__:
            raise ValueError(f"Invalid dictionary format for {Locator.__name__}")

        path = []
        for accessor, obj_type in data["path"]:
            obj_type = ObjectType(obj_type)
            _validate_accessor(accessor, obj_type)
            path.append((accessor, obj_type))
        return Locator(tuple(path))

    def __eq__(self, value: object, /) -> bool:
        """Equality operator."""
        if not isinstance(value, Locator):
            return False
        return self._path == value._path

    def __hash__(self) -> int:
        """Hash of the locator."""
        return hash(self._path)

    def __repr__(self):
        """Representation of the locator."""
        return self.__str__()

    def __str__(self):
        """String representation of the locator."""
        result = []
        for accessor, obj_type in self._path:
            if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
                result.append(f".{accessor}")
            elif obj_type == ObjectType.SEQUENCE:
                result.append(f"[{accessor}]")
            elif obj_type == ObjectType.DICT:
                result.append(f"['{accessor}']")
        return "".join(result)

    def get_value(self, obj: object) -> object:
        """Get the value of the locator."""
        for accessor, obj_type in self._path:
            obj = Locator._child(obj, accessor, obj_type)

        return obj

    def with_value(self, obj: object, value: object) -> object:
        """Return a copy of *obj* that holds *value* at the locator, leaving *obj* unchanged.

        Only the containers on the path are copied, and only shallowly. The copy of a frozen dataclass takes the value
        as well.

        Example:
            >>> locator = Locator(((1, ObjectType.SEQUENCE), ("b", ObjectType.DICT)))
            >>> obj = (0, {"a": 1, "b": 2})
            >>> locator.with_value(obj, 3), obj
            ((0, {'a': 1, 'b': 3}), (0, {'a': 1, 'b': 2}))
        """
        if not self._path:
            return value
        (accessor, obj_type), *rest = self._path
        child = Locator(tuple(rest)).with_value(Locator._child(obj, accessor, obj_type), value)
        if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
            copied = copy.copy(obj)
            object.__setattr__(copied, accessor, child)
            return copied
        if isinstance(obj, tuple):
            return Locator._tuple_with(obj, accessor, child)  # type: ignore[arg-type]
        copied = copy.copy(obj)
        copied[accessor] = child
        return copied

    @staticmethod
    def _tuple_with(obj: tuple, index: int, child: object) -> tuple:
        if type(obj) is tuple:
            return obj[:index] + (child,) + obj[index + 1 :]
        # With no or an empty __dict__, a namedtuple holds nothing but its items, so _replace rebuilds all of its state.
        if hasattr(obj, "_fields") and not getattr(obj, "__dict__", None):
            return obj._replace(**{obj._fields[index]: child})  # type: ignore[attr-defined]
        raise TypeError(
            f"Cannot replace an item of {type(obj).__qualname__}: only a tuple or a namedtuple without instance"
            " attributes can be rebuilt"
        )

    @staticmethod
    def _child(obj: object, accessor: int | str, obj_type: ObjectType) -> object:
        if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
            return getattr(obj, accessor)
        return obj[accessor]  # type: ignore[index]

    def set_value(self, obj: object, value: object) -> object:
        """Set the value of the locator."""
        num_paths = len(self._path)
        root = obj
        if num_paths == 0:
            return value
        for i, (accessor, obj_type) in enumerate(self._path):
            if i == num_paths - 1:
                break
            obj = Locator._child(obj, accessor, obj_type)
        if obj_type == ObjectType.DATACLASS or obj_type == ObjectType.USER_TYPE:
            setattr(obj, accessor, value)
        elif isinstance(obj, tuple):
            obj = list(obj)
            obj[accessor] = value  # type: ignore
            obj = tuple(obj)
            if i > 0:  # we have to modify parent with changed child reference
                parent_locator = Locator(self._path[:i])
                return parent_locator.set_value(root, obj)
            else:
                return obj
        else:
            obj[accessor] = value  # type: ignore
        return root

    @staticmethod
    def _iter_sequence(current: list | tuple):
        """Iterate over a sequence.

        Args:
            current: The sequence to iterate.

        Yields:
            Tuple of (accessor, value, object_type).
        """
        for idx, value in enumerate(current):
            yield idx, value, ObjectType.SEQUENCE

    @staticmethod
    def _iter_dict(current: dict):
        """Iterate over a dictionary.

        Args:
            current: The dictionary to iterate.

        Yields:
            Tuple of (accessor, value, object_type).
        """
        for key, value in current.items():
            yield key, value, ObjectType.DICT

    @staticmethod
    def _iter_dataclass(current: object):
        """Iterate over a dataclass.

        Args:
            current: The dataclass to iterate.

        Yields:
            Tuple of (accessor, value, object_type).
        """
        for field in dataclasses.fields(current):  # type: ignore
            value = getattr(current, field.name)
            yield field.name, value, ObjectType.DATACLASS

    @staticmethod
    def _iter_user_type(current: object):
        """Iterate over a user type.

        Args:
            current: The user type object to iterate.

        Yields:
            Tuple of (accessor, value, object_type).
        """
        for field in get_object_fields(current):
            value = getattr(current, field)
            yield field, value, ObjectType.USER_TYPE

    @staticmethod
    @cache
    def _get_iter_handler(obj_type):
        """Get the iter handler for the given object type.

        The purpose of this is to cache lookups and speed up the find leaves operation.

        Args:
            obj_type: The object type to get the iter handler for.

        Returns:
            The iter handler for the given object type.
        """
        if issubclass(obj_type, list | tuple):
            return Locator._iter_sequence
        elif issubclass(obj_type, dict):
            return Locator._iter_dict
        elif dataclasses.is_dataclass(obj_type):
            return Locator._iter_dataclass
        elif obj_type in Locator._user_types:
            return Locator._iter_user_type

        return None

    @staticmethod
    @cache
    def _resolve_only_tensors(obj_type: type, parent_only_tensors: bool) -> bool:
        """Resolve the only_tensors flag for the current object."""
        if obj_type in Locator._user_types:
            return Locator._user_types[obj_type]
        return parent_only_tensors

    @staticmethod
    def find_leaves(obj: object, only_tensors: bool = False) -> Generator[tuple["Locator", object], None, None]:
        """Walks the object and yields locators and their leaf values.

        Leaf value is anything that is stored in a sequence, dictionary, or dataclass. If `only_tensors` is True,
        only tensors are returned, otherwise all objects are returned.

        Args:
            obj: The object to walk.
            only_tensors: Whether to only return tensors.

        Yields:
            A tuple of (locator, value) where locator is the path to the leaf and value is the leaf value.

        Note:
        - if a user type is marked as ignore, it will not be traversed nor returned.
        - if a user type is registered, it will be traversed like a dataclass. If it was registered with
              `only_tensors=True`, only tensors are returned irrespective of method argument `only_tensors`.
              Such a case may happen when we want to handle LLM cache and look only for tensors inside cache object.
        """
        if type(obj) in Locator._ignored_types:
            return

        iter_handler = Locator._get_iter_handler(type(obj))
        if not iter_handler:
            # there is only a simple type that is not a container
            if not only_tensors or isinstance(obj, torch.Tensor):
                yield Locator(()), obj  # type: ignore[arg-type]
            return

        child_only_tensors = Locator._resolve_only_tensors(type(obj), only_tensors)

        stack = [(iter_handler(obj), child_only_tensors)]
        path = []

        while stack:
            parent_iter, current_only_tensors = stack[-1]
            try:
                # Get next child
                accessor, value, obj_type = next(parent_iter)

                if type(value) in Locator._ignored_types:
                    continue

                path.append((accessor, obj_type))

                child_iter_handler = Locator._get_iter_handler(type(value))
                if child_iter_handler:
                    # It is a container, push to stack
                    next_only_tensors = Locator._resolve_only_tensors(type(value), current_only_tensors)
                    stack.append((child_iter_handler(value), next_only_tensors))
                else:
                    # It is a leaf
                    if not current_only_tensors or isinstance(value, torch.Tensor):
                        yield Locator(tuple(path)), value
                    path.pop()

            except StopIteration:
                stack.pop()
                if stack:
                    path.pop()

    @staticmethod
    def register_user_type(cls: type, only_tensors: bool = False):
        """Register a user type."""
        Locator._user_types[cls] = only_tensors
        Locator.unignore_type(cls)
        Locator._get_iter_handler.cache_clear()
        Locator._resolve_only_tensors.cache_clear()

    @staticmethod
    def unregister_user_type(cls: type):
        """Unregister a user type."""
        Locator._user_types.pop(cls, None)
        Locator._get_iter_handler.cache_clear()
        Locator._resolve_only_tensors.cache_clear()

    @staticmethod
    def ignore_type(cls: type):
        """Ignore a type."""
        Locator._ignored_types.add(cls)
        Locator.unregister_user_type(cls)
        Locator._get_iter_handler.cache_clear()

    @staticmethod
    def unignore_type(cls: type):
        """Unignore a type."""
        Locator._ignored_types.discard(cls)
        Locator._get_iter_handler.cache_clear()
