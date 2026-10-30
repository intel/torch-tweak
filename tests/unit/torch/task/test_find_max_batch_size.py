# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
import pytest

from tests.toy_models.torch_models import ToyTorchModel
from torch_tweak.torch.task.find_max_batch_size import find_max_batch_size
from torch_tweak.torch.task.profiling import (
    AllSamplesProfilingStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    NumStepsMeasuringStopStrategy,
    ProfilingConfig,
)


@pytest.fixture
def mock_profiling_config():
    return ProfilingConfig(
        measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=1),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
        batch_sizes=[1, 2, 4],
    )


def test_highest_throughput_strategy_find_max_batch_size(mock_profiling_config, torch_device, tmp_path):
    model = ToyTorchModel()
    sample = model.sample().to(torch_device)
    _, throughput, _ = find_max_batch_size(
        model, "test", model.graph_spec(), [((sample,), {})], mock_profiling_config, torch_device, tmp_path
    )
    assert throughput > 0
