# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch import guard in torch_tweak.torch."""

import importlib.util
import re
import sys

import pytest


def import_torch_tweak_torch_copy():
    # A copy under another name keeps the imported torch_tweak.torch in sys.modules intact.
    origin = importlib.util.find_spec("torch_tweak.torch").origin
    spec = importlib.util.spec_from_file_location("torch_tweak_torch_copy", origin)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


@pytest.mark.parametrize(
    "message",
    [
        "PyTorch is not installed",
        "pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu torch",
        "pip install --index-url https://download.pytorch.org/whl/cpu torch",
    ],
)
def test_missing_torch_is_reported_with_install_commands(monkeypatch, message):
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=re.escape(message)):
        import_torch_tweak_torch_copy()


def test_missing_torch_dependency_keeps_its_own_error(monkeypatch, tmp_path):
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text("import missing_torch_dependency\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.syspath_prepend(tmp_path)

    with pytest.raises(ModuleNotFoundError, match="missing_torch_dependency"):
        import_torch_tweak_torch_copy()
