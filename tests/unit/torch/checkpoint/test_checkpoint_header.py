# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the header of ``.tt`` checkpoint files."""

import io
import json
import logging
import shutil
import struct

import pytest
import torch

from tests.utilities.helpers import requires_xpu
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


def test_a_newer_format_version_asks_to_upgrade_torch_tweak():
    fp = io.BytesIO(struct.pack("<I", 9999999) + b"not a header")

    with pytest.raises(IncompatibleCheckpointError, match="format version 9999999 is newer than .*Upgrade torch-tweak"):
        read_header(fp, {"cpu": "a"})


def test_an_older_format_version_asks_to_tune_again():
    fp = io.BytesIO(struct.pack("<I", 0) + b"not a header")

    with pytest.raises(IncompatibleCheckpointError, match="format version 0 is not .*Tune the model again"):
        read_header(fp, {"cpu": "a"})


def test_a_checkpoint_saved_before_the_format_version_is_rejected():
    fp = io.BytesIO(b"PK\x03\x04")

    with pytest.raises(IncompatibleCheckpointError, match="saved by an older torch-tweak"):
        read_header(fp, {"cpu": "a"})


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"\x01\x00",
        b"PK\x03",
        b"\x01\x00\x00\x00",
        b"\x01\x00\x00\x00" + b"\x0c\x00",
        b"\x01\x00\x00\x00" + b"\x0c\x00\x00\x00" + b'{"cpu":',
    ],
)
def test_a_truncated_checkpoint_header_is_rejected(header):
    fp = io.BytesIO(header)

    with pytest.raises(IncompatibleCheckpointError, match="truncated"):
        read_header(fp, {"cpu": "a"})


def test_an_environment_record_over_the_limit_is_rejected():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x01\x00")

    with pytest.raises(IncompatibleCheckpointError, match="65537 bytes, the limit is 65536"):
        read_header(fp, {"cpu": "a"})


def test_an_environment_record_that_is_not_json_keeps_the_decode_error():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"{")

    with pytest.raises(IncompatibleCheckpointError, match="not UTF-8 JSON") as raised:
        read_header(fp, {"cpu": "a"})

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_an_environment_record_that_is_not_utf8_keeps_the_decode_error():
    fp = io.BytesIO(b"\x01\x00\x00\x00" + b"\x01\x00\x00\x00" + b"\xff")

    with pytest.raises(IncompatibleCheckpointError, match="not UTF-8 JSON") as raised:
        read_header(fp, {"cpu": "a"})

    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    "header",
    [
        b"\x01\x00\x00\x00" + b"\x02\x00\x00\x00" + b"[]",
        b"\x01\x00\x00\x00" + b"\x04\x00\x00\x00" + b"null",
        b"\x01\x00\x00\x00" + b"\x0a\x00\x00\x00" + b'{"cpu": 1}',
    ],
)
def test_an_environment_record_that_is_not_an_object_of_strings_is_rejected(header):
    fp = io.BytesIO(header)

    with pytest.raises(IncompatibleCheckpointError, match="not an object of strings"):
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


@requires_xpu
def test_tuning_environment_records_the_accelerator_name_apart_from_the_driver():
    environment = tuning_environment()

    assert environment["accelerator"] == f"xpu: {torch.xpu.get_device_properties(0).name}"
    assert environment["driver"] == torch.xpu.get_device_properties(0).driver_version


@requires_xpu
def test_tuning_environment_records_only_the_type_of_an_accelerator_without_device_properties(monkeypatch):
    """torch.mps has no get_device_properties. Removing it from torch.xpu stands in for MPS."""
    monkeypatch.delattr(torch.xpu, "get_device_properties")

    environment = tuning_environment()

    assert environment["accelerator"] == "xpu"
    assert "driver" not in environment
