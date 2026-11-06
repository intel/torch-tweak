# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Optional backends load only when a checkpoint names them.

Each test runs in a fresh interpreter. Importing torchao_backend again in the pytest process
adds a second TorchAOBackend class to Backend.__subclasses__ for every later test.
"""

import pytest


def test_checkpoint_without_torchao_does_not_import_it(pytester):
    script = pytester.makepyfile(
        """
        import sys

        import torch

        from torch_tweak.torch.backend import TorchInductorBackend
        from torch_tweak.torch.module.sample_metadata import SampleMetadata
        from torch_tweak.torch.module.tuned_module import TunedModule

        module = torch.nn.Linear(2, 2)
        backend = TorchInductorBackend()
        backend._orig_module = module
        metadata = SampleMetadata.from_inputs((torch.zeros(2),), {}, strict=True)
        checkpoint = TunedModule(backends={metadata: backend}).to_dict()

        TunedModule.from_dict(module, checkpoint)

        print("torchao backend imported:", "torch_tweak.torch.backend.torchao_backend" in sys.modules)
        """
    )

    result = pytester.runpython(script)

    assert result.ret == 0
    result.stdout.fnmatch_lines(["torchao backend imported: False"])


def test_named_backend_without_its_package_raises(pytester):
    script = pytester.makepyfile(
        """
        import sys

        sys.modules["torchao"] = None

        from torch_tweak.torch.backend import import_optional_backend

        import_optional_backend("TorchAOBackend")
        """
    )

    result = pytester.runpython(script)

    assert result.ret == 1
    result.stderr.re_match_lines([r"ModuleNotFoundError: No module named 'torchao(\.\w+)*'.*"])


def test_checkpoint_with_torchao_imports_it(pytester):
    pytest.importorskip("torchao")
    script = pytester.makepyfile(
        """
        import sys

        from torch_tweak.torch.backend import import_optional_backend

        import_optional_backend("TorchAOBackend")

        print("torchao backend imported:", "torch_tweak.torch.backend.torchao_backend" in sys.modules)
        """
    )

    result = pytester.runpython(script)

    assert result.ret == 0
    result.stdout.fnmatch_lines(["torchao backend imported: True"])
