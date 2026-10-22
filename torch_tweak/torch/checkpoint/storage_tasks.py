# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Storage tasks."""

import hashlib
import logging
import shutil
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import count
from pathlib import Path
from typing import Any, Literal

import torch

from torch_tweak.torch.utils.path_utils import format_file_size, get_file_size
from torch_tweak.utils.permissions import secure_file, secure_mkdir, secure_tree

TT_EXTENSION = ".tt"
STATE_DICT_FILE = "state_dict.pt"

logger = logging.getLogger(__name__)


class SaveTask(ABC):
    """Base class to save state dict."""

    @abstractmethod
    def save(self, path: Path, state_dict: dict) -> None:
        """Save the state dictionary to the specified path.

        Args:
            path: The path where the state dictionary should be saved.
            state_dict: The state dictionary to save. Treated as read-only by
                all tasks except :class:`CopyBackendArtifactsTask`.
        """
        raise NotImplementedError("Subclass must implement this method")


class LoadTask(ABC):
    """Base class to load state dict."""

    @abstractmethod
    def load(self, path: Path) -> dict:
        """Load a state dictionary from the specified path.

        Args:
            path: The path from where the state dictionary should be loaded.

        Returns:
            dict: The loaded state dictionary.
        """
        raise NotImplementedError("Subclass must implement this method")


# Leaf value types that torch.load(weights_only=True) can reconstruct without
# importing any user-defined class. Verified against torch.load(weights_only=True)
# by tests/unit/torch/checkpoint/test_storage_tasks.py, which round trips every
# entry listed here and every container type traversed by _assert_weights_only_safe.
_WEIGHTS_ONLY_SAFE_LEAF_TYPES = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    torch.Tensor,
    torch.nn.Parameter,
    torch.Size,
    torch.dtype,
    torch.device,
)

# Container types the weights-only unpickler rebuilds itself, so their contents can be
# inspected recursively. Matched on the exact type: an arbitrary dict subclass
# (transformers ModelOutput, for instance) still needs its own class at load time, so it
# is rejected even though isinstance(obj, dict) would hold.
_MAPPING_TYPES = (dict, OrderedDict)
_SEQUENCE_TYPES = (list, tuple)


def _assert_weights_only_safe(obj: Any, path: str = "state_dict") -> None:
    """Raise ValueError when *obj* contains a type that weights_only=True cannot load.

    Mappings and sequences are traversed recursively. Matching is on the exact
    type rather than isinstance, because subclasses (namedtuples, dict subclasses
    such as the transformers ModelOutput family) need the originating class to be
    imported at load time and therefore cannot be loaded weights-only.

    Args:
        obj: Value to inspect recursively.
        path: Human-readable dotted-key path used in the error message so the
            caller can locate the offending value inside the state dict.

    Raises:
        ValueError: On the first value whose type is not safe for weights-only
            loading. The message names the type and the path to it.
    """
    if type(obj) in _WEIGHTS_ONLY_SAFE_LEAF_TYPES:
        return
    if type(obj) in _MAPPING_TYPES:
        for key, value in obj.items():
            _assert_weights_only_safe(key, f"{path}[<key {key!r}>]")
            _assert_weights_only_safe(value, f"{path}[{key!r}]")
    elif type(obj) in _SEQUENCE_TYPES:
        for index, item in enumerate(obj):
            _assert_weights_only_safe(item, f"{path}[{index}]")
    else:
        raise ValueError(
            f"{path}: value of type '{type(obj).__qualname__}' cannot be loaded with "
            "weights_only=True; serialize it to primitives or tensors before saving"
        )


def is_weights_only_safe(obj: Any) -> bool:
    """Return True when a checkpoint can hold *obj*, see :class:`TorchSaveTask`."""
    try:
        _assert_weights_only_safe(_paths_as_str(obj))
    except ValueError:
        return False
    return True


def _paths_as_str(obj: Any) -> Any:
    """Return a copy of *obj* with every Path value replaced by its string form.

    Every container that _assert_weights_only_safe traverses is rewritten, so the two
    functions agree on what counts as reachable. Dict keys are deliberately left alone:
    rewriting a key would silently change it, so a Path key is reported as unsupported
    instead.
    """
    if isinstance(obj, Path):
        return str(obj)
    if type(obj) in _MAPPING_TYPES:
        return type(obj)((key, _paths_as_str(value)) for key, value in obj.items())
    if type(obj) in _SEQUENCE_TYPES:
        return type(obj)(_paths_as_str(item) for item in obj)
    return obj


class TorchSaveTask(SaveTask):
    """Task to save a state dictionary using torch.save."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Save the state dictionary to the specified path.

        Args:
            path: The path where the state dictionary should be saved.
            state_dict: The state dictionary to save.

        Raises:
            ValueError: If state_dict contains a value that weights_only=True
                cannot load (e.g. a custom dataclass or namedtuple). The error
                is raised before anything is written to disk.
        """
        safe = _paths_as_str(state_dict)
        _assert_weights_only_safe(safe)
        file_path = path / STATE_DICT_FILE
        torch.save(safe, file_path)
        secure_file(file_path)

        file_size = get_file_size(file_path)
        logger.info("✅ State dict saved to %s [%s]", file_path, format_file_size(file_size))


class TorchLoadTask(LoadTask):
    """Task to load a state dictionary using torch.load."""

    def load(self, path: Path) -> dict:
        """Load a state dictionary from the specified path."""
        file_path = path / STATE_DICT_FILE
        if not file_path.exists():
            raise FileNotFoundError(f"State dictionary file not found: {file_path}")

        state_dict = torch_load_checkpoint(file_path)

        file_size = get_file_size(file_path)
        logger.info("✅ State dict loaded from %s [%s]", file_path, format_file_size(file_size))

        return state_dict


class MakeFolderTask(SaveTask):
    """Task to make a folder."""

    def __init__(self, overwrite: bool = True):
        """Initialize the task.

        Args:
            overwrite: Whether to overwrite the folder if it already exists.
        """
        self.overwrite = overwrite

    def save(self, path: Path, state_dict: dict) -> None:
        """Make a folder."""
        if path.exists():
            if self.overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Folder {path} already exists")
        secure_mkdir(path)


class RemoveFolderTask(SaveTask):
    """Task to remove a folder."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Remove a folder."""
        if path.exists():
            shutil.rmtree(path)


class CopyBackendArtifactsTask(SaveTask):
    """Task to copy backend artifacts."""

    @staticmethod
    def _copy_backend_artifact(source: Path, target_path: Path, counter: count) -> Path:
        """Copy a backend artifact into ``target_path``, preserving OpenVINO IR layout when needed."""
        if source.is_dir():
            dest_dir = target_path / f"{next(counter)}_{source.name}"
            shutil.copytree(source, dest_dir)
            secure_tree(dest_dir)
            return dest_dir

        if source.suffix == ".xml":
            sibling_bin = source.with_suffix(".bin")
            if not sibling_bin.is_file():
                raise FileNotFoundError(
                    f"OpenVINO IR weights file not found: {sibling_bin}. Expected a '.bin' sibling alongside '{source}'."
                )
            dest_dir = target_path / f"{next(counter)}_{source.parent.name}"
            secure_mkdir(dest_dir)
            dest_source = dest_dir / source.name
            dest_sibling = dest_dir / sibling_bin.name
            shutil.copy2(source, dest_source)
            shutil.copy2(sibling_bin, dest_sibling)
            secure_file(dest_source)
            secure_file(dest_sibling)
            return dest_dir

        dest = target_path / f"{next(counter)}_{source.name}"
        shutil.copy2(source, dest)
        secure_file(dest)
        return dest

    @staticmethod
    def _iter_nested_dicts(obj: object):
        """Yield every dict reachable via dict values and list/tuple elements."""
        if isinstance(obj, dict):
            yield obj
            for value in obj.values():
                yield from CopyBackendArtifactsTask._iter_nested_dicts(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from CopyBackendArtifactsTask._iter_nested_dicts(item)

    def save(self, target_path: Path, state_dict: dict) -> None:
        """Copy backend artifact files into *target_path* and update *state_dict* in-place.

        Walks nested dicts (including ``backends`` list entries from ``TunedModule.to_dict()``),
        copies each ``Path`` into ``target_path`` with a numeric prefix to avoid name collisions,
        and rewrites the dict in place to point at the copied locations.

        Args:
            target_path: Directory into which artifact files are copied.
            state_dict: The state dictionary to traverse. Path values are
                updated in-place to point to the copied locations.
        """
        counter = count(1)
        for nested_dict in self._iter_nested_dicts(state_dict):
            for key, value in nested_dict.items():
                if isinstance(value, Path):
                    nested_dict[key] = self._copy_backend_artifact(value, target_path, counter)


class ShaSumsSaveTask(SaveTask):
    """Task to save SHA hashes of files in the state dictionary."""

    def __init__(self, sha_type: Literal["256", "512"] = "256"):
        """Initialize the task.

        Args:
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".
        """
        self.sha_type = sha_type

    def save(self, path: Path, state_dict: dict) -> None:
        """Save SHA hashes of files in the specified path to a file.

        Args:
            path: The path containing files to hash.
            state_dict: The state dictionary (not used in this implementation).
        """
        sha_hashes = []

        # Iterate over all files in the specified path
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                # Calculate SHA hash of the file
                sha_hash = calculate_file_sha_hash(file_path, self.sha_type)
                sha_hashes.append((str(file_path), sha_hash))

        # Write SHA hashes to file
        sha_file_path = get_sha_sums_path(path, self.sha_type)
        with sha_file_path.open("w", encoding="utf-8") as fp:
            for file_path, sha_hash in sha_hashes:
                fp.write(f"{sha_hash}  {file_path}\n")
        secure_file(sha_file_path)


class ZipSaveTask(SaveTask):
    """Task to save a state dictionary to a zip file."""

    def save(self, path: Path, state_dict: dict) -> None:
        """Compress the folder under the given path to a zip file, including all nested folders.

        Args:
            path: The path to the folder that should be compressed to a zip file.
            state_dict: The state dictionary (not used in this implementation).

        Raises:
            FileNotFoundError: If the folder does not exist.
            ValueError: If the path is not a directory.
        """
        zip_path = path.with_name(path.name + TT_EXTENSION)

        logger.info("🔄 Compressing checkpoint...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zip_fp:
            # Recursively find all files in the directory tree
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path from the root folder to preserve directory structure
                    file_path_in_zip = file_path.relative_to(path)
                    zip_fp.write(file_path, file_path_in_zip)
                secure_file(zip_path)

        zip_file_size = get_file_size(zip_path)
        logger.info("✅ Checkpoint compressed and saved to %s [%s]", zip_path, format_file_size(zip_file_size))
        # copy sha hashes aside zip if they exist
        # Try both SHA-256 and SHA-512 files
        for sha_type in ["256", "512"]:
            sha_file_path = get_sha_sums_path(path, sha_type)
            if sha_file_path.exists():
                copy_sha_file_path = path.parent / sha_file_path.name
                shutil.copy(sha_file_path, copy_sha_file_path)
                secure_file(copy_sha_file_path)
                logger.info("✅ SHA hash file copied to %s", copy_sha_file_path)
                break


class ShaSumsLoadTask(LoadTask):
    """Task to load and verify SHA hashes of files."""

    def __init__(self, sha_type: Literal["256", "512"] = "256"):
        """Initialize the task.

        Args:
            sha_type: The SHA type to use for hashing. Must be either "256" or "512".
        """
        self.sha_type = sha_type

    def load(self, path: Path) -> dict:
        """Load and verify SHA hashes of files in the specified path.

        Args:
            path: The path containing files to verify.

        Raises:
            FileNotFoundError: If sha_hashes.txt file is not found.
            ValueError: If sha_hashes.txt file format is invalid.
        """
        sha_file_path = get_sha_sums_path(path, self.sha_type)

        if not sha_file_path.exists():
            raise FileNotFoundError(f"SHA hash file not found: {sha_file_path}")

        stored_hashes = {}
        try:
            with open(sha_file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("  ", 1)  # Split on double space
                        if len(parts) != 2:
                            raise ValueError(f"Invalid hash file format in line: {line}")
                        hash_value, file_path = parts
                        stored_hashes[file_path] = hash_value
        except Exception as e:
            raise ValueError("Error reading SHA hash file") from e

        failed_files = []
        for file_path, hash_value in stored_hashes.items():
            current_hash = calculate_file_sha_hash(file_path, self.sha_type)
            if current_hash != hash_value:
                failed_files.append(file_path)

        if failed_files:
            raise ValueError(f"Failed to verify SHA hashes for files: {failed_files}")

        return {}


class UnzipLoadTask(LoadTask):
    """Task to load a state dictionary from a zip file."""

    def load(self, path: Path) -> dict:
        """Extract the zip file at the given path to a folder with the same name.

        Args:
            path: The path to the zip file that should be extracted.

        Returns:
            dict: An empty dictionary (following the pattern of other load tasks).

        Raises:
            FileNotFoundError: If the zip file does not exist.
            ValueError: If the path is not a zip file.
        """
        zip_path = path.with_name(path.name + TT_EXTENSION)
        if not zip_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {zip_path}")

        file_size = get_file_size(zip_path)
        logger.info("🔄 Extracting checkpoint from: %s [%s]", path, format_file_size(file_size))

        if not check_checkpoint_valid(path):
            with zipfile.ZipFile(zip_path, "r") as zip_fp:
                _safe_extract_zip(zip_fp, path)
        logger.info("✅ Checkpoint extracted")
        return {}


def _safe_extract_zip(zip_fp: zipfile.ZipFile, target_path: Path) -> None:
    """Safe extract zip file by resolving paths to avoid zip slip."""
    target = target_path.resolve()
    for member in zip_fp.infolist():
        member_path = (target / member.filename).resolve()
        if not member_path.is_relative_to(target):
            raise ValueError(f"Zip slip detected: {member_path} is outside of {target}")
    zip_fp.extractall(target)
    secure_tree(target)


def torch_load_checkpoint(path: Path) -> dict:
    """Load a checkpoint state dict.

    The checkpoint may come from an untrusted source, so only tensors and plain Python
    values are unpickled. Loading fails for anything that would need an arbitrary class.

    Raises:
        ValueError: If torch cannot read the file. A refused object, a truncated file and
            an empty one all surface as unrelated exception types, so they are reported
            as one condition. The original error stays chained for the details.
    """
    try:
        return torch.load(path, weights_only=True)
    except Exception as error:
        raise ValueError(
            f"Cannot load checkpoint {path}. It may be corrupt, or created before torch_tweak "
            "started storing checkpoints without pickle; such checkpoints cannot be read and "
            "have to be rebuilt by tuning the model again."
        ) from error


def calculate_file_sha_hash(file_path: str | Path, sha_type: Literal["256", "512"] = "256") -> str:
    """Calculate SHA hash of a file.

    Args:
        file_path: Path to the file to hash.
        sha_type: The SHA type to use for hashing. Must be either "256" or "512".

    Returns:
        str: SHA hash of the file as a hexadecimal string.
    """
    if sha_type == "256":
        sha_hash = hashlib.sha256()
    elif sha_type == "512":
        sha_hash = hashlib.sha512()
    else:
        raise ValueError("sha_type must be either '256' or '512'")

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha_hash.update(chunk)
    return sha_hash.hexdigest()


def get_sha_sums_path(path: Path, sha_type: Literal["256", "512"] = "256") -> Path:
    """Get the path to the SHA sums file.

    Args:
        path: The path to the checkpoint directory.
        sha_type: The SHA type used for hashing. Must be either "256" or "512".

    Returns:
        Path: The path to the SHA sums file.
    """
    hashes_filename = f"sha{sha_type}_sums.txt"
    return path / (path.stem + "_" + hashes_filename)


def check_checkpoint_valid(path: Path) -> bool:
    """Check if the checkpoint is valid.

    Args:
        path: The path to the checkpoint.

    Returns:
        bool: True if the checkpoint is valid, False otherwise.
        Checkpoint is valid if there is sha file aside path and inside path are the same.
    """
    if not path.exists():
        return False  # there is no unzipped folder

    # Try both SHA-256 and SHA-512 files
    for sha_type in ["256", "512"]:
        aside_sha_path = path.parent / f"sha{sha_type}_sums.txt"
        inside_sha_path = path / f"sha{sha_type}_sums.txt"

        if aside_sha_path.exists() and inside_sha_path.exists():
            return aside_sha_path.read_text(encoding="utf-8") == inside_sha_path.read_text(encoding="utf-8")

    return False
