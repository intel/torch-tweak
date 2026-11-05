# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optional backends load only when a checkpoint names them."""

import sys

import pytest

from torch_tweak.torch.backend import import_optional_backend

_TORCHAO_BACKEND = "torch_tweak.torch.backend.torchao_backend"


def test_checkpoint_without_torchao_does_not_import_it(monkeypatch):
    monkeypatch.delitem(sys.modules, _TORCHAO_BACKEND, raising=False)

    import_optional_backend("TorchInductorBackend")

    assert _TORCHAO_BACKEND not in sys.modules


def test_named_backend_without_its_package_raises(monkeypatch):
    hidden = [name for name in list(sys.modules) if name == _TORCHAO_BACKEND or name.split(".", 1)[0] == "torchao"]
    for name in hidden:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torchao", None)

    with pytest.raises(ImportError):
        import_optional_backend("TorchAOBackend")


def test_checkpoint_with_torchao_imports_it(monkeypatch):
    pytest.importorskip("torchao")
    monkeypatch.delitem(sys.modules, _TORCHAO_BACKEND, raising=False)

    import_optional_backend("TorchAOBackend")

    assert _TORCHAO_BACKEND in sys.modules
