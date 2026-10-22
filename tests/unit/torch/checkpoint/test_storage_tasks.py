# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Tests for model state storage functionality."""

import dataclasses
import os
import shutil
import stat
from collections import OrderedDict, namedtuple
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tests.toy_models.torch_models import ToyTorchModel
from torch_tweak.torch.checkpoint.storage_tasks import (
    _WEIGHTS_ONLY_SAFE_LEAF_TYPES,
    STATE_DICT_FILE,
    TT_EXTENSION,
    CopyBackendArtifactsTask,
    MakeFolderTask,
    ShaSumsLoadTask,
    ShaSumsSaveTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
    _assert_weights_only_safe,
    _paths_as_str,
    check_checkpoint_valid,
    get_sha_sums_path,
    is_weights_only_safe,
    torch_load_checkpoint,
)
from torch_tweak.utils.permissions import SECURE_DIR_MODE, SECURE_FILE_MODE


@contextmanager
def permissive_umask():
    """Temporarily remove umask restrictions so tests catch inherited permissions."""
    old_umask = os.umask(0)
    try:
        yield
    finally:
        os.umask(old_umask)


def file_mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_save_load_torch_task(tmp_path):
    model = ToyTorchModel(is_linear=True)
    save_task = TorchSaveTask()
    load_task = TorchLoadTask()

    orig_state_dict = model.state_dict()
    orig_state_dict["extra_data"] = "for testing"

    save_task.save(tmp_path, orig_state_dict)
    state_dict = load_task.load(tmp_path)

    assert state_dict.keys() == orig_state_dict.keys()
    assert state_dict["extra_data"] == orig_state_dict["extra_data"]


def test_torch_save_task_stores_paths_as_str(tmp_path):
    artifact = tmp_path / "model.xml"

    TorchSaveTask().save(tmp_path, {"artifact": artifact})
    state_dict = TorchLoadTask().load(tmp_path)

    assert type(state_dict["artifact"]) is str
    assert state_dict["artifact"] == str(artifact)


def test_torch_save_task_converts_nested_paths(tmp_path):
    state_dict = {"backends": [{"artifact": tmp_path / "a.xml"}, {"artifact": tmp_path / "b.xml"}]}

    TorchSaveTask().save(tmp_path, state_dict)
    loaded = TorchLoadTask().load(tmp_path)

    assert [entry["artifact"] for entry in loaded["backends"]] == [str(tmp_path / "a.xml"), str(tmp_path / "b.xml")]


def test_torch_save_task_does_not_mutate_state_dict(tmp_path):
    artifact = tmp_path / "model.xml"
    state_dict = {"artifact": artifact, "backends": [{"artifact": artifact}]}

    TorchSaveTask().save(tmp_path, state_dict)

    assert state_dict["artifact"] is artifact
    assert state_dict["backends"][0]["artifact"] is artifact


def test_torch_save_task_output_loads_weights_only(tmp_path):
    model = ToyTorchModel(is_linear=True)
    state_dict = {
        "artifact": tmp_path / "model.xml",
        "module": model.state_dict(),
        "device": torch.device("cpu"),
    }

    TorchSaveTask().save(tmp_path, state_dict)
    loaded = torch.load(tmp_path / STATE_DICT_FILE, weights_only=True)

    assert loaded["artifact"] == str(tmp_path / "model.xml")


@dataclasses.dataclass
class _UserData:
    """Stands in for a custom object that only its own class can reconstruct."""

    x: int = 1


class _DictSubclass(dict):
    """Stands in for transformers ModelOutput, which forward() returns for HF models."""


@pytest.mark.parametrize(
    "state_dict,expected_path",
    [
        pytest.param({"data": _UserData()}, r"state_dict\['data'\]", id="dataclass"),
        pytest.param({"record": namedtuple("Record", "x")(1)}, r"state_dict\['record'\]", id="namedtuple"),
        pytest.param({"out": _DictSubclass(logits=1)}, r"state_dict\['out'\]", id="dict-subclass"),
        pytest.param({"samples": [(torch.zeros(2), _UserData())]}, r"state_dict\['samples'\]\[0\]\[1\]", id="nested"),
        pytest.param({namedtuple("Record", "x")(1): "v"}, r"state_dict\[<key Record\(x=1\)>\]", id="dict-key"),
    ],
)
def test_torch_save_task_rejects_unloadable_values(state_dict, expected_path, tmp_path):
    """Values needing their own class at load time must be named by path and never reach the disk."""
    with pytest.raises(ValueError, match=rf"{expected_path}.*weights_only=True"):
        TorchSaveTask().save(tmp_path, state_dict)

    assert not (tmp_path / STATE_DICT_FILE).exists()


# One sample per entry of _WEIGHTS_ONLY_SAFE_LEAF_TYPES, plus every container the guard
# traverses. Shared by the soundness test and the coverage test below so they cannot drift.
_SAFE_SAMPLE_VALUES = {
    "none": None,
    "bool": True,
    "int": 1,
    "float": 1.0,
    "complex": 1 + 2j,
    "str": "hello",
    "bytes": b"hi",
    "bytearray": bytearray(b"hi"),
    "tensor": torch.zeros(2),
    "parameter": torch.nn.Parameter(torch.zeros(2)),
    "size": torch.Size([1, 2]),
    "dtype": torch.float32,
    "device": torch.device("cpu"),
    "list": [1, 2, torch.ones(3)],
    "tuple": (1, "a"),
    "ordered-dict": OrderedDict(weight=torch.zeros(2)),
    "nested": {"a": [1, (2, 3)]},
}


@pytest.mark.parametrize("value", _SAFE_SAMPLE_VALUES.values(), ids=_SAFE_SAMPLE_VALUES.keys())
def test_accepted_values_really_survive_a_weights_only_load(value, tmp_path):
    """Whatever the guard accepts, torch must load.

    The guard may be stricter than torch, but never laxer: accepting something torch then
    refuses is what produces a checkpoint that fails hours later, at load time. This pins
    the allowlist to the real torch behaviour, so a torch upgrade that narrows it breaks
    the build instead of a user's checkpoint.
    """
    state_dict = {"value": value}
    _assert_weights_only_safe(state_dict)

    path = tmp_path / "probe.pt"
    torch.save(state_dict, path)
    torch.load(path, weights_only=True)


def test_every_safe_leaf_type_is_checked_against_torch():
    """Adding a leaf type without a sample value above would leave it unverified."""
    sampled = {type(value) for value in _SAFE_SAMPLE_VALUES.values()}

    assert set(_WEIGHTS_ONLY_SAFE_LEAF_TYPES) <= sampled


def test_paths_as_str_converts_every_container_the_guard_traverses(tmp_path):
    """_paths_as_str and _assert_weights_only_safe must agree on what is reachable."""
    state_dict = {
        "mapping": OrderedDict(artifact=tmp_path / "a.xml"),
        "list": [tmp_path / "b.xml"],
        "tuple": (tmp_path / "c.xml",),
        "untouched": (1, "a"),
    }

    result = _paths_as_str(state_dict)

    assert result["mapping"] == {"artifact": str(tmp_path / "a.xml")}
    assert type(result["mapping"]) is OrderedDict
    assert result["list"] == [str(tmp_path / "b.xml")]
    assert result["tuple"] == (str(tmp_path / "c.xml"),)
    assert result["untouched"] == (1, "a")
    _assert_weights_only_safe(result)


def test_paths_as_str_leaves_dict_keys_alone(tmp_path):
    """Rewriting a key would silently change it, so keys are reported by the guard instead."""
    key = tmp_path / "a.xml"

    assert _paths_as_str({key: 1}) == {key: 1}
    with pytest.raises(ValueError, match="weights_only=True"):
        TorchSaveTask().save(tmp_path, {key: 1})


def _write_pre_weights_only_checkpoint(path):
    """Write a checkpoint in the old format, holding a PosixPath that torch now refuses."""
    torch.save({"ov_model_dir": path.parent / "ov"}, path)


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(_write_pre_weights_only_checkpoint, id="pre-weights-only"),
        pytest.param(lambda path: path.write_bytes(b"not a checkpoint at all"), id="garbage"),
        pytest.param(lambda path: path.write_bytes(b""), id="empty"),
    ],
)
def test_torch_load_checkpoint_reports_unreadable_files_uniformly(write, tmp_path):
    """torch raises UnpicklingError, EOFError or RuntimeError here; all must reach the user alike."""
    path = tmp_path / STATE_DICT_FILE
    write(path)

    with pytest.raises(ValueError, match="have to be rebuilt") as excinfo:
        torch_load_checkpoint(path)

    # Chained, so the wrapper cannot hide which object torch actually refused.
    assert excinfo.value.__cause__ is not None


def test_torch_load_task_explains_pre_weights_only_checkpoints(tmp_path):
    _write_pre_weights_only_checkpoint(tmp_path / STATE_DICT_FILE)

    with pytest.raises(ValueError, match="have to be rebuilt"):
        TorchLoadTask().load(tmp_path)


def test_make_folder_task_overwrite(tmp_path):
    """Test MakeFolderTask with overwrite=True."""
    folder_path = tmp_path / "test_folder"

    # Create the folder task with overwrite=True
    make_folder_task = MakeFolderTask(overwrite=True)

    # Test creating a new folder
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()

    # Test that trying to create the same folder again overwrites it
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()


def test_make_folder_task_no_overwrite(tmp_path):
    """Test MakeFolderTask with overwrite=False."""
    folder_path = tmp_path / "test_folder"

    # Create the folder task with overwrite=False
    make_folder_task = MakeFolderTask(overwrite=False)

    # Test creating a new folder
    make_folder_task.save(folder_path, {})
    assert folder_path.exists()
    assert folder_path.is_dir()

    # Test that trying to create the same folder again raises FileExistsError
    with pytest.raises(FileExistsError, match=f"Folder {folder_path} already exists"):
        make_folder_task.save(folder_path, {})


def test_make_folder_task_uses_secure_permissions(tmp_path):
    """Test MakeFolderTask does not inherit permissive directory permissions."""
    folder_path = tmp_path / "test_folder"

    with permissive_umask():
        MakeFolderTask().save(folder_path, {})

    assert file_mode(folder_path) == SECURE_DIR_MODE


def test_copy_backend_artifacts_uses_secure_permissions(tmp_path):
    """Test copied backend artifacts do not preserve permissive source permissions."""
    source_file = tmp_path / "artifact.bin"
    source_file.write_text("artifact")
    source_file.chmod(0o666)

    target_path = tmp_path / "checkpoint"
    with permissive_umask():
        MakeFolderTask().save(target_path, {})
        state_dict = {"artifact": source_file}
        CopyBackendArtifactsTask().save(target_path, state_dict)

    copied_file = state_dict["artifact"]
    assert file_mode(target_path) == SECURE_DIR_MODE
    assert file_mode(copied_file) == SECURE_FILE_MODE


def test_copy_backend_artifact_directory_uses_secure_permissions(tmp_path):
    """Test copied artifact directories are normalized to secure permissions."""
    source_dir = tmp_path / "artifact_dir"
    source_dir.mkdir()
    nested_file = source_dir / "nested.bin"
    nested_file.write_text("artifact")
    source_dir.chmod(0o777)
    nested_file.chmod(0o666)

    target_path = tmp_path / "checkpoint"
    with permissive_umask():
        MakeFolderTask().save(target_path, {})
        state_dict = {"artifact_dir": source_dir}
        CopyBackendArtifactsTask().save(target_path, state_dict)

    copied_dir = state_dict["artifact_dir"]
    assert file_mode(copied_dir) == SECURE_DIR_MODE
    assert file_mode(copied_dir / "nested.bin") == SECURE_FILE_MODE


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_sha_sums_save_task(tmp_path, sha_type):
    # Create test files
    test_file1 = tmp_path / "test_file1.txt"
    test_file2 = tmp_path / "test_file2.txt"
    test_file1.write_text("content1")
    test_file2.write_text("content2")

    sha_sums_save_task = ShaSumsSaveTask(sha_type=sha_type)
    sha_sums_save_task.save(tmp_path, {})

    # Check that sha_sums.txt was created (default)
    sha_sums_file = get_sha_sums_path(tmp_path, sha_type)
    assert sha_sums_file.exists()

    sha_sums_load_task = ShaSumsLoadTask(sha_type=sha_type)
    result = sha_sums_load_task.load(tmp_path)

    # Should return empty dict on success
    assert result == {}

    # modify content of a file
    test_file1.write_text("modified_content")

    with pytest.raises(ValueError, match="Failed to verify SHA hashes for files"):
        sha_sums_load_task.load(tmp_path)

    # remove sha_sums.txt
    sha_sums_file.unlink()

    with pytest.raises(FileNotFoundError, match="SHA hash file not found"):
        sha_sums_load_task.load(tmp_path)


def test_sha_sums_save_task_uses_secure_permissions(tmp_path):
    """Test SHA sums files do not inherit permissive file permissions."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("content")

    with permissive_umask():
        ShaSumsSaveTask().save(tmp_path, {})

    assert file_mode(get_sha_sums_path(tmp_path, "256")) == SECURE_FILE_MODE


@pytest.mark.parametrize("checkpoint", ["valid", "invalid", "missing"])
def test_zip_unzip_tasks(tmp_path, checkpoint):
    """Test ZipSaveTask and UnzipLoadTask functionality."""
    # Create test files and subdirectories
    test_file1 = tmp_path / "test_file1.txt"
    test_file2 = tmp_path / "test_file2.txt"
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    test_file3 = subdir / "test_file3.txt"

    test_file1.write_text("content1")
    test_file2.write_text("content2")
    test_file3.write_text("content3")

    # Sha256 file should be copied aside zip
    sha_sums = ShaSumsSaveTask()
    sha_sums.save(tmp_path, {})

    # Test zip save task
    zip_save_task = ZipSaveTask()
    zip_save_task.save(tmp_path, {})

    # Check that zip file was created
    zip_file = tmp_path.with_suffix(TT_EXTENSION)
    assert zip_file.exists()
    sha_sums_file = get_sha_sums_path(tmp_path, "256")
    assert sha_sums_file.exists()

    if checkpoint == "missing":
        # remove completely
        shutil.rmtree(tmp_path)  # noqa: F821
    elif checkpoint == "invalid":
        # remove one file
        test_file1.unlink()  # noqa: F821
        # modify sha_sums file
        sha_sums_file.write_text("invalid_sha_sums")
    else:
        pass  # leave checkpoint as is

    # Test unzip load task
    unzip_load_task = UnzipLoadTask()
    result = unzip_load_task.load(tmp_path)

    # Should return empty dict on success
    assert result == {}

    # Check that files were extracted correctly
    extract_path = tmp_path.with_suffix("")
    assert extract_path.exists()
    assert extract_path.is_dir()

    # Verify all files are present and have correct content
    extracted_file1 = extract_path / "test_file1.txt"
    extracted_file2 = extract_path / "test_file2.txt"
    extracted_subdir = extract_path / "subdir"
    extracted_file3 = extracted_subdir / "test_file3.txt"

    assert extracted_file1.exists()
    assert extracted_file2.exists()
    assert extracted_subdir.exists()
    assert extracted_file3.exists()

    assert extracted_file1.read_text() == "content1"
    assert extracted_file2.read_text() == "content2"
    assert extracted_file3.read_text() == "content3"

    # Test that non-existent zip file raises FileNotFoundError
    non_existent_path = tmp_path / "non_existent"
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        unzip_load_task.load(non_existent_path)


def test_zip_unzip_tasks_use_secure_permissions(tmp_path):
    """Test zip checkpoints and extracted objects do not inherit permissive permissions."""
    source_file = tmp_path / "test_file.txt"
    source_file.write_text("content")
    ShaSumsSaveTask().save(tmp_path, {})

    with permissive_umask():
        ZipSaveTask().save(tmp_path, {})

    zip_file = tmp_path.with_suffix(TT_EXTENSION)
    aside_sha_file = tmp_path.parent / get_sha_sums_path(tmp_path, "256").name
    assert file_mode(zip_file) == SECURE_FILE_MODE
    assert file_mode(aside_sha_file) == SECURE_FILE_MODE

    shutil.rmtree(tmp_path)
    with permissive_umask():
        UnzipLoadTask().load(tmp_path)

    assert file_mode(tmp_path) == SECURE_DIR_MODE
    assert file_mode(tmp_path / "test_file.txt") == SECURE_FILE_MODE


def test_check_checkpoint_valid_nonexistent_path(tmp_path):
    """Test check_checkpoint_valid with a path that doesn't exist."""
    nonexistent_path = tmp_path / "nonexistent_checkpoint"

    result = check_checkpoint_valid(nonexistent_path)
    assert result is False


def test_check_checkpoint_valid_no_sha_sums_files(tmp_path):
    """Test check_checkpoint_valid with no sha_sums files."""
    # Create a directory but no sha files
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    result = check_checkpoint_valid(test_dir)
    assert result is False


def test_check_checkpoint_valid_only_aside_sha_sums(tmp_path):
    """Test check_checkpoint_valid with only aside sha_sums file."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create only the aside sha_sums file
    aside_sha_sums_file = tmp_path / "sha256_sums.txt"
    aside_sha_sums_file.write_text("test_content")

    result = check_checkpoint_valid(test_dir)
    assert result is False


def test_check_checkpoint_valid_only_inside_sha_sums(tmp_path):
    """Test check_checkpoint_valid with only inside sha_sums file."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create only the inside sha_sums file
    inside_sha_sums_file = test_dir / "sha256_sums.txt"
    inside_sha_sums_file.write_text("test_content")

    result = check_checkpoint_valid(test_dir)
    assert result is False


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_check_checkpoint_valid_matching_sha_sums_files(tmp_path, sha_type):
    """Test check_checkpoint_valid with matching sha_sums files."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create both sha files with matching content
    aside_sha_file = tmp_path / f"sha{sha_type}_sums.txt"
    inside_sha_file = test_dir / f"sha{sha_type}_sums.txt"
    test_content = "test_content"

    aside_sha_file.write_text(test_content)
    inside_sha_file.write_text(test_content)

    result = check_checkpoint_valid(test_dir)
    assert result is True


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_check_checkpoint_valid_different_sha_sums_files(tmp_path, sha_type):
    """Test check_checkpoint_valid with different sha_sums files."""
    test_dir = tmp_path / "test_checkpoint"
    test_dir.mkdir()

    # Create both sha files with different content
    aside_sha_file = tmp_path / f"sha{sha_type}_sums.txt"
    inside_sha_file = test_dir / f"sha{sha_type}_sums.txt"

    aside_sha_file.write_text("content1")
    inside_sha_file.write_text("content2")

    result = check_checkpoint_valid(test_dir)
    assert result is False


@pytest.mark.parametrize("sha_type", ["256", "512"])
def test_get_sha_sums_path(sha_type):
    """Test get_sha_sums_path function."""
    path = Path("/test/path")

    # Test SHA-256 (default)
    sha_path = get_sha_sums_path(path, sha_type)
    assert sha_path == path / (path.stem + f"_sha{sha_type}_sums.txt")


class _Opaque:
    pass


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({"x": [torch.ones(1), Path("a")]}, True),
        ({"x": [torch.ones(1), _Opaque()]}, False),
    ],
    ids=["tensors-and-paths", "opaque-object"],
)
def test_is_weights_only_safe(obj, expected):
    assert is_weights_only_safe(obj) is expected
