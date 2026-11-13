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
import logging
import platform
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO

import torch

from torch_tweak.__version__ import __version__ as torch_tweak_version

TT_FORMAT_VERSION = 1

_UINT32 = struct.Struct("<I")

_ZIP_SIGNATURE = b"PK\x03\x04"

_LIBRARIES = ("torch", "openvino", "torchao", "triton", "triton-xpu", "intel-sycl-rt")

logger = logging.getLogger(__name__)


class IncompatibleCheckpointError(ValueError):
    """The checkpoint has another format version, or was tuned in another environment and loading is strict."""


def tuning_environment() -> dict[str, str]:
    """Return the hardware and library versions that tuning results depend on."""
    environment = {
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


def read_header(fp: BinaryIO, environment: dict[str, str], strict: bool = False) -> dict[str, str]:
    """Read the header from *fp* and leave *fp* at the start of the archive.

    Args:
        fp: Checkpoint file positioned at its start.
        environment: Environment of the running process.
        strict: Raise instead of warning when the recorded environment differs from *environment*.

    Returns:
        The environment recorded in the checkpoint.

    Raises:
        IncompatibleCheckpointError: If the format version is not :data:`TT_FORMAT_VERSION`,
            or *strict* is set and the recorded environment differs from *environment*.
    """
    format_bytes = fp.read(_UINT32.size)
    if format_bytes == _ZIP_SIGNATURE:
        raise IncompatibleCheckpointError(
            "Checkpoint has no format version, it was saved by an older torch-tweak. Tune the model again."
        )
    format_version = _UINT32.unpack(format_bytes)[0] if len(format_bytes) == _UINT32.size else None
    if format_version != TT_FORMAT_VERSION:
        raise IncompatibleCheckpointError(
            f"Checkpoint format version {format_version} is not {TT_FORMAT_VERSION}. Tune the model again."
        )
    record_length = _UINT32.unpack(fp.read(_UINT32.size))[0]
    recorded = json.loads(fp.read(record_length).decode("utf-8"))
    differences = [
        f"{key}: tuned with {recorded.get(key)!r}, running {environment.get(key)!r}"
        for key in sorted(recorded.keys() | environment.keys())
        if recorded.get(key) != environment.get(key)
    ]
    if differences and strict:
        raise IncompatibleCheckpointError(
            "Checkpoint was tuned in another environment. Tune the model again.\n" + "\n".join(differences)
        )
    if differences:
        logger.warning(
            "Checkpoint was tuned in another environment, its backends may not be the fastest here. "
            "Tune the model again on this machine for the best performance.\n%s",
            "\n".join(differences),
        )
    return recorded


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
    descriptions = set()
    for index in range(torch.accelerator.device_count()):
        properties = device_module.get_device_properties(index)
        driver = getattr(properties, "driver_version", None) or torch.version.cuda or torch.version.hip
        descriptions.add(f"{properties.name}, driver {driver}")
    return {device_type: "; ".join(sorted(descriptions))}
