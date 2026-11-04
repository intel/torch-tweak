# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the header of ``.tt`` checkpoint files."""

import io
import shutil

import pytest
import torch

from torch_tweak.torch.checkpoint.checkpoint_header import (
    IncompatibleCheckpointError,
    read_header,
    tuning_environment,
    write_header,
)
from torch_tweak.torch.checkpoint.storage_tasks import UnzipLoadTask, ZipSaveTask

HEADER_OF_HOST_A = b"\x01\x00\x00\x00" + b"\x0d\x00\x00\x00" + b'{"host": "a"}'


def test_header_is_the_format_version_then_the_environment_record():
    fp = io.BytesIO()

    write_header(fp, {"host": "a"})

    assert fp.getvalue() == HEADER_OF_HOST_A


def test_reading_the_header_leaves_the_file_at_the_archive():
    fp = io.BytesIO(HEADER_OF_HOST_A + b"PK\x03\x04")

    read_header(fp, {"host": "a"})

    assert fp.read() == b"PK\x03\x04"


def test_another_format_version_is_rejected_before_the_environment_is_read():
    fp = io.BytesIO(b"\x02\x00\x00\x00" + b"not a header of version 1")

    with pytest.raises(IncompatibleCheckpointError, match="format version 2 is not 1"):
        read_header(fp, {"host": "a"})


def test_a_checkpoint_without_a_format_version_is_rejected():
    fp = io.BytesIO(b"PK\x03\x04")

    with pytest.raises(IncompatibleCheckpointError, match="format version 67324752 is not 1"):
        read_header(fp, {"host": "a"})


def test_an_empty_checkpoint_is_rejected():
    fp = io.BytesIO(b"")

    with pytest.raises(IncompatibleCheckpointError, match="format version None is not 1"):
        read_header(fp, {"host": "a"})


def test_a_checkpoint_tuned_on_another_host_is_rejected():
    fp = io.BytesIO(HEADER_OF_HOST_A)

    with pytest.raises(IncompatibleCheckpointError, match="host: tuned with 'a', running 'b'"):
        read_header(fp, {"host": "b"})


def test_a_checkpoint_tuned_without_a_library_now_installed_is_rejected():
    fp = io.BytesIO(HEADER_OF_HOST_A)

    with pytest.raises(IncompatibleCheckpointError, match="openvino: tuned with None, running '2026.1.0'"):
        read_header(fp, {"host": "a", "openvino": "2026.1.0"})


def test_a_checkpoint_loads_in_the_environment_it_was_tuned_in(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"host": "a"}).save(folder, {})
    shutil.rmtree(folder)

    UnzipLoadTask(environment={"host": "a"}).load(folder)

    assert (folder / "state.txt").read_text() == "tuned"


def test_a_checkpoint_tuned_in_another_environment_is_not_extracted(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"host": "a"}).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError):
        UnzipLoadTask(environment={"host": "b"}).load(folder)

    assert not folder.exists()


def test_tuning_environment_records_the_torch_version():
    environment = tuning_environment()

    assert environment["torch"] == torch.__version__
