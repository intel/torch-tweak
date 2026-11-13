# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the header of ``.tt`` checkpoint files."""

import io
import logging
import shutil

import pytest
import torch

from torch_tweak.torch.checkpoint import storage_tasks
from torch_tweak.torch.checkpoint.checkpoint_header import (
    IncompatibleCheckpointError,
    read_header,
    tuning_environment,
    write_header,
)
from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.checkpoint.storage_tasks import UnzipLoadTask, ZipSaveTask

HEADER_OF_CPU_A = b"\x01\x00\x00\x00" + b"\x0c\x00\x00\x00" + b'{"cpu": "a"}'


def test_header_is_the_format_version_then_the_environment_record():
    fp = io.BytesIO()

    write_header(fp, {"cpu": "a"})

    assert fp.getvalue() == HEADER_OF_CPU_A


def test_reading_the_header_leaves_the_file_at_the_archive():
    fp = io.BytesIO(HEADER_OF_CPU_A + b"PK\x03\x04")

    read_header(fp, {"cpu": "a"})

    assert fp.read() == b"PK\x03\x04"


def test_reading_the_header_returns_the_recorded_environment():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    assert read_header(fp, {"cpu": "b"}) == {"cpu": "a"}


def test_another_format_version_is_rejected_before_the_environment_is_read():
    fp = io.BytesIO(b"\x02\x00\x00\x00" + b"not a header of version 1")

    with pytest.raises(IncompatibleCheckpointError, match="format version 2 is not 1"):
        read_header(fp, {"cpu": "a"})


def test_a_checkpoint_saved_before_the_format_version_is_rejected():
    fp = io.BytesIO(b"PK\x03\x04")

    with pytest.raises(IncompatibleCheckpointError, match="saved by an older torch-tweak"):
        read_header(fp, {"cpu": "a"})


def test_an_empty_checkpoint_is_rejected():
    fp = io.BytesIO(b"")

    with pytest.raises(IncompatibleCheckpointError, match="format version None is not 1"):
        read_header(fp, {"cpu": "a"})


def test_a_checkpoint_tuned_in_another_environment_is_reported_in_a_warning(caplog):
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with caplog.at_level(logging.WARNING):
        read_header(fp, {"cpu": "b"})

    assert "cpu: tuned with 'a', running 'b'" in caplog.text


def test_a_checkpoint_tuned_in_the_same_environment_logs_no_warning(caplog):
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with caplog.at_level(logging.WARNING):
        read_header(fp, {"cpu": "a"})

    assert caplog.records == []


def test_strict_reading_rejects_a_checkpoint_tuned_in_another_environment():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with pytest.raises(IncompatibleCheckpointError, match="cpu: tuned with 'a', running 'b'"):
        read_header(fp, {"cpu": "b"}, strict=True)


def test_strict_reading_rejects_a_checkpoint_tuned_without_a_library_now_installed():
    fp = io.BytesIO(HEADER_OF_CPU_A)

    with pytest.raises(IncompatibleCheckpointError, match="openvino: tuned with None, running '2026.1.0'"):
        read_header(fp, {"cpu": "a", "openvino": "2026.1.0"}, strict=True)


def test_a_checkpoint_tuned_in_another_environment_is_extracted(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"cpu": "a"}).save(folder, {})
    shutil.rmtree(folder)

    UnzipLoadTask(environment={"cpu": "b"}).load(folder)

    assert (folder / "state.txt").read_text() == "tuned"


def test_strict_loading_does_not_extract_a_checkpoint_tuned_in_another_environment(tmp_path):
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "state.txt").write_text("tuned")
    ZipSaveTask(environment={"cpu": "a"}).save(folder, {})
    shutil.rmtree(folder)

    with pytest.raises(IncompatibleCheckpointError, match="cpu: tuned with 'a', running 'b'"):
        UnzipLoadTask(environment={"cpu": "b"}, strict_environment=True).load(folder)

    assert not folder.exists()


def test_strict_local_storage_rejects_a_checkpoint_tuned_in_another_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_tasks, "tuning_environment", lambda: {"cpu": "a"})
    LocalTorchStorage(base_folder=tmp_path).save("model.tt", {"weight": torch.tensor([1.0])})
    monkeypatch.setattr(storage_tasks, "tuning_environment", lambda: {"cpu": "b"})

    with pytest.raises(IncompatibleCheckpointError, match="cpu: tuned with 'a', running 'b'"):
        LocalTorchStorage(base_folder=tmp_path, strict_environment=True).load("model.tt")


def test_tuning_environment_records_the_torch_version():
    environment = tuning_environment()

    assert environment["torch"] == torch.__version__
