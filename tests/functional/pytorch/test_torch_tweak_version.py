# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.


import pytest
import torch

import torch_tweak


@pytest.mark.functional
def test_versions():
    assert torch.__version__
    assert torch_tweak.__version__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
