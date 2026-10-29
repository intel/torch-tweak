# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import gc
import pathlib
import tempfile
from logging import getLogger

import pytest
import torch
from diffusers import StableDiffusionPipeline

from torch_tweak.torch.backend import TorchEagerBackend, TorchInductorBackend
from torch_tweak.torch.module import Module
from torch_tweak.torch.tune_strategy import FirstWinsStrategy
from torch_tweak.torch.tuning import load, save, tune

logger = getLogger(__name__)

PROMPT = "a photo of a cat sitting on a windowsill, natural lighting, detailed fur, photorealistic"


def _get_pipeline(model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5", device: str = "xpu"):
    """Get a pretrained Flux model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load the model on

    Returns:
        FluxPipeline: The loaded Flux pipeline
    """
    model = StableDiffusionPipeline.from_pretrained(model_name)
    model.to(device)
    return model


def _tune_and_save(save_path: pathlib.Path, device: str = "xpu"):
    pipeline = _get_pipeline(device=device)

    pipeline.text_encoder = Module(
        pipeline.text_encoder,
        "text_encoder",
        strategy=FirstWinsStrategy(backends=[TorchEagerBackend(), TorchInductorBackend()]),
    )

    tune(
        pipeline,
        [PROMPT],
        batch_sizes=[1],
        max_num_batches_per_batch_size=1,
        device=device,
        disable_external_logging=False,
    )

    save(pipeline, save_path)


def _load_and_infer(load_path: pathlib.Path, device: str = "xpu"):
    pipeline = _get_pipeline(device=device)

    load(pipeline, load_path, disable_external_logging=False)

    res = pipeline(PROMPT, generator=torch.Generator(device="xpu").manual_seed(42))
    opt_image = res[0][0]

    return opt_image


def _clean_up():
    gc.collect()
    torch.xpu.empty_cache()


@pytest.mark.nightly
def test_pipeline_serialization_with_first_module_only():
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = pathlib.Path(temp_dir) / "sd-dev-tuned.pt"
        _tune_and_save(save_path)
        _clean_up()
        opt_image = _load_and_infer(save_path)

    assert opt_image is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
