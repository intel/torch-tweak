# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Header that precedes the zip archive in a ``.tt`` checkpoint file.

Layout, all integers unsigned 32-bit little-endian:

1. format version, :data:`TT_FORMAT_VERSION`;
2. byte length of the environment record;
3. environment record, UTF-8 JSON object of strings, see :func:`tuning_environment`.

A reader checks the format version before it reads anything else, and the environment
before it extracts the archive.
"""

import json
import platform
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO

import torch

from torch_tweak.__version__ import __version__ as torch_tweak_version

TT_FORMAT_VERSION = 1

_UINT32 = struct.Struct("<I")

_LIBRARIES = ("torch", "openvino", "torchao", "triton", "triton-xpu", "intel-sycl-rt")


class IncompatibleCheckpointError(ValueError):
    """The checkpoint has another format version or was tuned in another environment."""


def tuning_environment() -> dict[str, str]:
    """Return the host, hardware, and library versions that tuning results depend on."""
    environment = {
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "python": platform.python_version(),
        "torch_tweak": torch_tweak_version,
    }
    for library in _LIBRARIES:
        environment[library] = _installed_version(library)
    environment.update(_accelerators())
    return environment


def write_header(fp: BinaryIO, environment: dict[str, str]) -> None:
    """Write the format version and *environment* at the position of *fp*."""
    record = json.dumps(environment, sort_keys=True).encode("utf-8")
    fp.write(_UINT32.pack(TT_FORMAT_VERSION))
    fp.write(_UINT32.pack(len(record)))
    fp.write(record)


def read_header(fp: BinaryIO, environment: dict[str, str]) -> None:
    """Read the header from *fp* and leave *fp* at the start of the archive.

    Raises:
        IncompatibleCheckpointError: If the format version is not :data:`TT_FORMAT_VERSION`,
            or the recorded environment differs from *environment*.
    """
    format_version = _read_uint32(fp)
    if format_version != TT_FORMAT_VERSION:
        raise IncompatibleCheckpointError(f"Checkpoint format version {format_version} is not {TT_FORMAT_VERSION}.")
    record_length = _read_uint32(fp)
    recorded = json.loads(fp.read(record_length).decode("utf-8"))
    differences = [
        f"{key}: tuned with {recorded.get(key)!r}, running {environment.get(key)!r}"
        for key in sorted(recorded.keys() | environment.keys())
        if recorded.get(key) != environment.get(key)
    ]
    if differences:
        raise IncompatibleCheckpointError(f"Checkpoint was tuned in another environment: {'; '.join(differences)}.")


def _read_uint32(fp: BinaryIO) -> int | None:
    data = fp.read(_UINT32.size)
    if len(data) < _UINT32.size:
        return None
    return _UINT32.unpack(data)[0]


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name":
                return value.strip()
    return platform.processor()


def _accelerators() -> dict[str, str]:
    if not torch.accelerator.is_available():
        return {}
    device_type = torch.accelerator.current_accelerator().type
    device_module = getattr(torch, device_type)
    accelerators = {}
    for index in range(torch.accelerator.device_count()):
        properties = device_module.get_device_properties(index)
        driver = getattr(properties, "driver_version", None) or torch.version.cuda or torch.version.hip
        accelerators[f"{device_type}:{index}"] = f"{properties.name}, driver {driver}"
    return accelerators
