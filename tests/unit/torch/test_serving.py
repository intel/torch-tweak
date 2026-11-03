# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from torch_tweak.torch import ServingConfig, ServingMode, serve


def linear(weight):
    layer = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return layer


def eager_config(workdir):
    return ServingConfig(workdir=workdir, backends=("eager",))


def serve_once(workdir):
    """The first run in an empty workdir records [[3, 4]], the second tunes it."""
    module = linear(torch.tensor([[2.0, 0.0]]))
    serve(module, "toy", eager_config(workdir))
    module(torch.tensor([[3.0, 4.0]]))


def test_empty_workdir_resolves_to_record(tmp_path):
    served = serve(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.RECORD


def test_recording_resolves_to_tune(tmp_path):
    serve_once(tmp_path)

    served = serve(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.TUNE


def test_tuned_choice_resolves_to_serve(tmp_path):
    serve_once(tmp_path)
    serve_once(tmp_path)

    served = serve(linear(torch.tensor([[2.0, 0.0]])), "toy", eager_config(tmp_path))

    assert served.serving_mode == ServingMode.SERVE


def test_tune_serves_the_recorded_shape(tmp_path):
    serve_once(tmp_path)
    module = linear(torch.tensor([[2.0, 0.0]]))
    serve(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(actual, torch.tensor([[6.0]]))


def test_serve_uses_the_weights_of_the_served_module(tmp_path):
    serve_once(tmp_path)
    serve_once(tmp_path)
    module = linear(torch.tensor([[0.0, 1.0]]))
    serve(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(actual, torch.tensor([[4.0]]))


def test_unrecorded_shape_runs_the_original_module(tmp_path):
    serve_once(tmp_path)
    serve_once(tmp_path)
    module = linear(torch.tensor([[2.0, 0.0]]))
    serve(module, "toy", eager_config(tmp_path))

    actual = module(torch.tensor([[1.0, 0.0], [5.0, 0.0]]))

    torch.testing.assert_close(actual, torch.tensor([[2.0], [10.0]]))


def test_serve_without_tuned_choice_is_rejected(tmp_path):
    config = ServingConfig(workdir=tmp_path, mode="serve", backends=("eager",))

    with pytest.raises(FileNotFoundError, match="No tuned choice"):
        serve(linear(torch.tensor([[2.0, 0.0]])), "toy", config)


def test_tune_without_recording_is_rejected(tmp_path):
    config = ServingConfig(workdir=tmp_path, mode="tune", backends=("eager",))

    with pytest.raises(FileNotFoundError, match="No recording"):
        serve(linear(torch.tensor([[2.0, 0.0]])), "toy", config)


def test_unknown_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown backends \\['tensorrt'\\]"):
        ServingConfig(workdir=tmp_path, backends=("eager", "tensorrt"))


def test_long_names_with_a_shared_tail_get_separate_directories(tmp_path):
    tail = "x" * 300
    first = linear(torch.tensor([[2.0, 0.0]]))
    second = linear(torch.tensor([[2.0, 0.0]]))
    serve(first, f"a.{tail}", eager_config(tmp_path))
    serve(second, f"b.{tail}", eager_config(tmp_path))

    first(torch.tensor([[3.0, 4.0]]))
    second(torch.tensor([[3.0, 4.0]]))

    assert len(list(tmp_path.iterdir())) == 2
