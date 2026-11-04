# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from torch_tweak.torch import ServedModule, ServingConfig, ServingMode


def linear(weight):
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def eager_config(workdir):
    return ServingConfig(workdir=workdir, backends=("eager",), fallback="eager")


def call_once(workdir, x):
    """One server run in auto mode: the first run in an empty workdir records x, the next one tunes."""
    module = linear(torch.tensor([[2.0, 0.0]]))
    ServedModule(module, "toy", eager_config(workdir))
    module(x)


def test_empty_workdir_resolves_to_record(tmp_path):
    served = ServedModule(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.RECORD


def test_recording_resolves_to_tune(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))

    served = ServedModule(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.TUNE


def test_tuned_choice_resolves_to_serve(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))

    served = ServedModule(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.SERVE


def test_tune_returns_the_module_output_for_the_recorded_shape(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    module = linear(torch.tensor([[2.0, 0.0]]))
    ServedModule(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(actual, torch.tensor([[6.0]]))


def test_serve_uses_the_weights_of_the_served_module(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    module = linear(torch.tensor([[0.0, 1.0]]))
    ServedModule(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(actual, torch.tensor([[4.0]]))


def test_unrecorded_shape_returns_the_module_output(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    module = linear(torch.tensor([[2.0, 0.0]]))
    ServedModule(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[1.0, 0.0], [5.0, 0.0]]))

    torch.testing.assert_close(actual, torch.tensor([[2.0], [10.0]]))


def test_inductor_fallback_compiles_one_graph_for_two_unrecorded_shapes(tmp_path):
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    call_once(tmp_path, torch.tensor([[3.0, 4.0]]))
    module = linear(torch.tensor([[2.0, 0.0]]))
    ServedModule(module, "toy", ServingConfig(workdir=tmp_path, backends=("eager",)))
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    first = module(torch.tensor([[1.0, 0.0], [5.0, 0.0], [7.0, 0.0]]))
    second = module(torch.tensor([[1.0, 0.0], [5.0, 0.0], [7.0, 0.0], [9.0, 0.0]]))

    torch.testing.assert_close(first, torch.tensor([[2.0], [10.0], [14.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0], [10.0], [14.0], [18.0]]))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_serve_without_tuned_choice_is_rejected(tmp_path):
    config = ServingConfig(workdir=tmp_path, mode="serve", backends=("eager",))

    with pytest.raises(FileNotFoundError, match="No tuned choice"):
        ServedModule(linear(torch.tensor([[2.0, 0.0]])), "toy", config)


def test_tune_without_recording_is_rejected(tmp_path):
    config = ServingConfig(workdir=tmp_path, mode="tune", backends=("eager",))

    with pytest.raises(FileNotFoundError, match="No recording"):
        ServedModule(linear(torch.tensor([[2.0, 0.0]])), "toy", config)


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown backends \\['tensorrt'\\]"):
        ServingConfig(workdir=tmp_path, backends=("eager", "tensorrt"))


def test_empty_backends_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="backends is empty"):
        ServingConfig(workdir=tmp_path, backends=())


def test_long_names_with_a_shared_tail_get_separate_directories(tmp_path):
    tail = "x" * 300
    first = linear(torch.tensor([[2.0, 0.0]]))
    second = linear(torch.tensor([[2.0, 0.0]]))
    ServedModule(first, f"a.{tail}", eager_config(tmp_path))
    ServedModule(second, f"b.{tail}", eager_config(tmp_path))

    first(torch.tensor([[3.0, 4.0]]))
    second(torch.tensor([[3.0, 4.0]]))

    assert len(list(tmp_path.iterdir())) == 2
