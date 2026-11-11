# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO GPU inference that reads and writes PyTorch XPU memory in place.

The OpenVINO Python bindings cannot pass a ``void*`` remote property, so this module drives the
OpenVINO C API, OpenCL, and Level Zero through ``ctypes``.
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import openvino as ov
import torch

LEVEL_ZERO_INTEROP_MISSING = (
    "OpenVINO on XPU needs the Intel GPU driver to run OpenCL on Level Zero (cl_intel_level_zero_interop). "
    "On Battlemage, start the process with EnableLEO=1. On newer GPUs, update the GPU driver."
)


class LevelZeroInteropError(RuntimeError):
    """The Intel GPU driver cannot share PyTorch XPU memory with OpenVINO."""


_P = ctypes.c_void_p

_CL_SUCCESS = 0
_CL_DEVICE_TYPE_GPU = 1 << 2
_CL_CONTEXT_L0_HANDLE_INTEL = 0x42B0
_CL_DEVICE_L0_HANDLE_INTEL = 0x42B4

_ZE_RESULT_SUCCESS = 0
_ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x15
_ZE_MEMORY_TYPE_DEVICE = 2

# Values of ov_element_type_e in openvino/c/ov_common.h.
_OV_ELEMENT_TYPES = {
    torch.bool: 1,
    torch.bfloat16: 2,
    torch.float16: 3,
    torch.float32: 4,
    torch.float64: 5,
    torch.int8: 7,
    torch.int16: 8,
    torch.int32: 9,
    torch.int64: 10,
    torch.uint8: 16,
}


class _OvShape(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int64), ("dims", ctypes.POINTER(ctypes.c_int64))]


class _OvDimension(ctypes.Structure):
    _fields_ = [("min", ctypes.c_int64), ("max", ctypes.c_int64)]


class _OvPartialShape(ctypes.Structure):
    _fields_ = [("rank", _OvDimension), ("dims", ctypes.POINTER(_OvDimension))]


class _OvProperty(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", _P)]


class _ZeMemoryAllocationProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", _P),
        ("type", ctypes.c_int),
        ("id", ctypes.c_uint64),
        ("pageSize", ctypes.c_uint64),
    ]


_OUT = ctypes.POINTER(_P)
_PROPERTIES = ctypes.POINTER(_OvProperty)

_OPENVINO_C_SIGNATURES = {
    "ov_core_create": [_OUT],
    "ov_core_read_model": [_P, ctypes.c_char_p, ctypes.c_char_p, _OUT],
    "ov_model_reshape_input_by_name": [_P, ctypes.c_char_p, _OvPartialShape],
    "ov_core_create_context_props": [_P, ctypes.c_char_p, ctypes.c_size_t, _PROPERTIES, _OUT],
    "ov_core_compile_model_with_context_props": [_P, _P, _P, ctypes.c_size_t, _PROPERTIES, _OUT],
    "ov_compiled_model_create_infer_request": [_P, _OUT],
    "ov_remote_context_create_tensor_props": [_P, ctypes.c_int, _OvShape, ctypes.c_size_t, _PROPERTIES, _OUT],
    "ov_infer_request_set_tensor": [_P, ctypes.c_char_p, _P],
    "ov_infer_request_infer": [_P],
    "ov_tensor_free": [_P],
    "ov_infer_request_free": [_P],
    "ov_compiled_model_free": [_P],
    "ov_remote_context_free": [_P],
    "ov_model_free": [_P],
    "ov_core_free": [_P],
}


@functools.cache
def _openvino_c() -> ctypes.CDLL:
    libs = Path(ov.__file__).parent / "libs"
    path = next(libs.glob("libopenvino_c.so*"), None)
    if path is None:
        raise RuntimeError(f"The OpenVINO C library is not in {libs}")
    lib = ctypes.CDLL(str(path))
    for name, argtypes in _OPENVINO_C_SIGNATURES.items():
        getattr(lib, name).argtypes = argtypes
    lib.ov_get_last_err_msg.restype = ctypes.c_char_p
    return lib


@functools.cache
def _level_zero() -> ctypes.CDLL:
    try:
        lib = ctypes.CDLL("libze_loader.so.1")
    except OSError as error:
        raise LevelZeroInteropError(f"The Level Zero loader cannot be loaded: {error}") from error
    lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), _OUT]
    lib.zeDriverGetDefaultContext.argtypes = [_P]
    lib.zeDriverGetDefaultContext.restype = _P
    lib.zeMemGetAllocProperties.argtypes = [_P, _P, ctypes.POINTER(_ZeMemoryAllocationProperties), _OUT]
    return lib


@functools.cache
def _opencl() -> ctypes.CDLL:
    try:
        lib = ctypes.CDLL("libOpenCL.so.1")
    except OSError as error:
        raise LevelZeroInteropError(f"The OpenCL loader cannot be loaded: {error}") from error
    lib.clGetPlatformIDs.argtypes = [ctypes.c_uint32, _OUT, ctypes.POINTER(ctypes.c_uint32)]
    lib.clGetDeviceIDs.argtypes = [_P, ctypes.c_uint64, ctypes.c_uint32, _OUT, ctypes.POINTER(ctypes.c_uint32)]
    lib.clGetDeviceInfo.argtypes = [_P, ctypes.c_uint32, ctypes.c_size_t, _P, ctypes.POINTER(ctypes.c_size_t)]
    lib.clCreateContext.argtypes = [
        ctypes.POINTER(ctypes.c_ssize_t),
        ctypes.c_uint32,
        _OUT,
        _P,
        _P,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.clCreateContext.restype = _P
    lib.clReleaseContext.argtypes = [_P]
    return lib


def ov_element_type(dtype: torch.dtype) -> int:
    """Returns the OpenVINO C ``ov_element_type_e`` value for a ``torch`` dtype."""
    try:
        return _OV_ELEMENT_TYPES[dtype]
    except KeyError:
        raise ValueError(f"OpenVINO on XPU does not support {dtype}") from None


def _check(status: int, call: str) -> None:
    if status != 0:
        raise RuntimeError(f"{call} failed: {_openvino_c().ov_get_last_err_msg().decode()}")


def _ov_properties(pairs: Sequence[tuple[bytes, bytes | int]]) -> ctypes.Array[_OvProperty]:
    """Packs key and value pairs into an ``ov_property_t`` array.

    A ``bytes`` value is passed by address. The caller keeps it alive until the C call returns.
    """
    entries = [
        (key, ctypes.cast(ctypes.c_char_p(value), _P).value if isinstance(value, bytes) else value)
        for key, value in pairs
    ]
    return (_OvProperty * len(entries))(*entries)


def _level_zero_handles(device: torch.device) -> tuple[int, int]:
    """Returns a Level Zero context and the Level Zero device that hold the memory of ``device``."""
    ze = _level_zero()
    probe = torch.empty(1, device=device)
    count = ctypes.c_uint32()
    ze.zeDriverGet(ctypes.byref(count), None)
    drivers = (_P * count.value)()
    ze.zeDriverGet(ctypes.byref(count), drivers)
    for driver in drivers:
        # The driver keeps one USM table for all its contexts, so its default context resolves
        # the allocations PyTorch made in its own context.
        context = ze.zeDriverGetDefaultContext(driver)
        if context is None:
            continue
        properties = _ZeMemoryAllocationProperties(stype=_ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES)
        ze_device = _P()
        status = ze.zeMemGetAllocProperties(
            context, probe.data_ptr(), ctypes.byref(properties), ctypes.byref(ze_device)
        )
        if status == _ZE_RESULT_SUCCESS and properties.type == _ZE_MEMORY_TYPE_DEVICE:
            return context, ze_device.value
    raise LevelZeroInteropError(f"No Level Zero driver holds the memory of {device}")


def _opencl_gpus() -> list[int]:
    cl = _opencl()
    count = ctypes.c_uint32()
    if cl.clGetPlatformIDs(0, None, ctypes.byref(count)) != _CL_SUCCESS:
        return []
    platforms = (_P * count.value)()
    cl.clGetPlatformIDs(count, platforms, None)
    gpus: list[int] = []
    for platform in platforms:
        if cl.clGetDeviceIDs(platform, _CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(count)) != _CL_SUCCESS:
            continue
        devices = (_P * count.value)()
        cl.clGetDeviceIDs(platform, _CL_DEVICE_TYPE_GPU, count, devices, None)
        gpus.extend(devices)
    return gpus


def _opencl_context(ze_context: int, ze_device: int) -> int:
    """Returns an OpenCL context built on ``ze_context``, so it shares USM with Level Zero."""
    cl = _opencl()
    for cl_device in _opencl_gpus():
        handle = _P()
        status = cl.clGetDeviceInfo(
            cl_device, _CL_DEVICE_L0_HANDLE_INTEL, ctypes.sizeof(handle), ctypes.byref(handle), None
        )
        if status == _CL_SUCCESS and handle.value == ze_device:
            break
    else:
        raise LevelZeroInteropError(LEVEL_ZERO_INTEROP_MISSING)
    properties = (ctypes.c_ssize_t * 3)(_CL_CONTEXT_L0_HANDLE_INTEL, ze_context, 0)
    error = ctypes.c_int32()
    context = cl.clCreateContext(properties, 1, (_P * 1)(cl_device), None, None, ctypes.byref(error))
    if error.value != _CL_SUCCESS:
        raise LevelZeroInteropError(f"clCreateContext on the Level Zero context failed with OpenCL error {error.value}")
    return context


def has_level_zero_interop(device: torch.device) -> bool:
    """Returns whether OpenVINO can read and write the memory of the XPU ``device`` in place."""
    try:
        context = _opencl_context(*_level_zero_handles(device))
    except LevelZeroInteropError:
        return False
    _opencl().clReleaseContext(context)
    return True


class XpuInferRequest:
    """An OpenVINO GPU infer request that binds PyTorch XPU tensors as remote tensors."""

    def __init__(
        self,
        model_xml: Path,
        device: torch.device,
        *,
        ov_device: str,
        config: Mapping[str, str],
        scalar_inputs: Sequence[str],
    ):
        """Compiles ``model_xml`` for ``ov_device``, the OpenVINO GPU that is the XPU ``device``.

        Args:
            model_xml: OpenVINO IR to compile.
            device: PyTorch XPU device whose tensors the request reads and writes.
            ov_device: OpenVINO GPU name from ``openvino_device_for_xpu``.
            config: OpenVINO compile properties.
            scalar_inputs: Model inputs to reshape to a static scalar before compiling.
        """
        self._device = device
        self._release: list[tuple[Callable[[int], None], int]] = []
        try:
            self._compile(model_xml, ov_device, config, scalar_inputs)
        except BaseException:
            self.close()
            raise

    def _own(self, release: Callable[[int], None], handle: int) -> int:
        self._release.append((release, handle))
        return handle

    def _compile(self, model_xml: Path, ov_device: str, config: Mapping[str, str], scalar_inputs: Sequence[str]):
        lib = _openvino_c()
        cl_context = self._own(_opencl().clReleaseContext, _opencl_context(*_level_zero_handles(self._device)))

        out = _P()
        _check(lib.ov_core_create(ctypes.byref(out)), "ov_core_create")
        core = self._own(lib.ov_core_free, out.value)

        context_properties = [(b"CONTEXT_TYPE", b"OCL"), (b"OCL_CONTEXT", cl_context)]
        _check(
            lib.ov_core_create_context_props(
                core,
                ov_device.encode(),
                len(context_properties),
                _ov_properties(context_properties),
                ctypes.byref(out),
            ),
            "ov_core_create_context_props",
        )
        self._context = self._own(lib.ov_remote_context_free, out.value)

        _check(lib.ov_core_read_model(core, str(model_xml).encode(), None, ctypes.byref(out)), "ov_core_read_model")
        model = out.value
        try:
            scalar = _OvPartialShape(rank=_OvDimension(0, 0))
            for name in scalar_inputs:
                _check(
                    lib.ov_model_reshape_input_by_name(model, name.encode(), scalar), "ov_model_reshape_input_by_name"
                )
            compile_properties = [(key.encode(), value.encode()) for key, value in config.items()]
            _check(
                lib.ov_core_compile_model_with_context_props(
                    core,
                    model,
                    self._context,
                    len(compile_properties),
                    _ov_properties(compile_properties),
                    ctypes.byref(out),
                ),
                "ov_core_compile_model_with_context_props",
            )
        finally:
            lib.ov_model_free(model)
        compiled_model = self._own(lib.ov_compiled_model_free, out.value)

        _check(
            lib.ov_compiled_model_create_infer_request(compiled_model, ctypes.byref(out)),
            "ov_compiled_model_create_infer_request",
        )
        self._request = self._own(lib.ov_infer_request_free, out.value)

    def _remote_tensor(self, tensor: torch.Tensor) -> int:
        lib = _openvino_c()
        shape = _OvShape(tensor.dim(), (ctypes.c_int64 * tensor.dim())(*tensor.shape))
        properties = [(b"SHARED_MEM_TYPE", b"USM_USER_BUFFER"), (b"MEM_HANDLE", tensor.data_ptr())]
        out = _P()
        _check(
            lib.ov_remote_context_create_tensor_props(
                self._context,
                ov_element_type(tensor.dtype),
                shape,
                len(properties),
                _ov_properties(properties),
                ctypes.byref(out),
            ),
            "ov_remote_context_create_tensor_props",
        )
        return out.value

    def infer(self, inputs: Mapping[str, torch.Tensor], outputs: Mapping[str, torch.Tensor]) -> None:
        """Runs the model on ``inputs`` and writes the results into ``outputs``.

        Every tensor is contiguous and lives on the XPU device of this request. Each mapping is keyed
        by OpenVINO tensor name.
        """
        lib = _openvino_c()
        remote_tensors: list[int] = []
        try:
            for name, tensor in (*inputs.items(), *outputs.items()):
                remote_tensors.append(self._remote_tensor(tensor))
                _check(
                    lib.ov_infer_request_set_tensor(self._request, name.encode(), remote_tensors[-1]),
                    "ov_infer_request_set_tensor",
                )
            # OpenVINO runs on its own queue and does not wait for the PyTorch stream.
            torch.xpu.current_stream(self._device).synchronize()
            _check(lib.ov_infer_request_infer(self._request), "ov_infer_request_infer")
        finally:
            for remote_tensor in remote_tensors:
                lib.ov_tensor_free(remote_tensor)

    def close(self) -> None:
        """Releases the infer request, the compiled model, and the OpenCL context."""
        while self._release:
            release, handle = self._release.pop()
            release(handle)
