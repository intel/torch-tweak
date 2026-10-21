# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Model utilities for ASR pipeline."""

from copy import deepcopy

import nemo.collections.asr
import nemo.core.neural_types.neural_type
import torch
from nemo.core.classes.common import typecheck
from omegaconf import open_dict

import torch_tweak.torch as tt


def get_model(model_name: str = "nvidia/parakeet-rnnt-1.1b"):
    """Get a pretrained ASR model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path

    Returns:
        ASR model
    """
    # Note: Disable typechecking for nemo, as passing inputs between onnx and nemo fails
    typecheck.set_typecheck_enabled(False)

    # Note: Allowing nemo object to be un/serialized (torch.load)
    torch.serialization.add_safe_globals([
        nemo.core.neural_types.neural_type.NeuralType,
        nemo.core.neural_types.elements.MelSpectrogramType,
        nemo.core.neural_types.axes.AxisType,
        nemo.core.neural_types.axes.AxisKind,
        nemo.core.neural_types.elements.LengthsType,
        nemo.core.neural_types.elements.SpectrogramType,
        nemo.core.neural_types.elements.IntType,
        nemo.core.neural_types.elements.AcousticEncodedRepresentation,
    ])

    asr_model = nemo.collections.asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)

    cfg = deepcopy(asr_model.decoding.cfg)
    with open_dict(cfg):
        cfg.greedy.use_cuda_graph_decoder = False
        cfg.strategy = "greedy_batch"

    asr_model.change_decoding_strategy(cfg)

    asr_model.eval()
    asr_model = asr_model.to("xpu")

    # Note: Move STFT window to GPU
    asr_model.preprocessor.featurizer.window = asr_model.preprocessor.featurizer.window.to("xpu")

    return asr_model


def wrap_pipeline(name: str, pipeline: nemo.collections.asr.models.EncDecRNNTBPEModel, strategy: tt.TuneStrategy):
    """Wrap modules in the pipeline.

    Args:
        name: The name of the pipeline
        pipeline: The ASR pipeline
        strategy: The tuning strategy

    Returns:
        ASR model
    """
    pipeline.encoder = tt.Module(
        pipeline.encoder,
        name=f"{name}-encoder",
        strategy=strategy,
    )

    pipeline.decoder.prediction["dec_rnn"] = tt.Module(
        pipeline.decoder.prediction["dec_rnn"],
        name=f"{name}-decoder-rnn",
        strategy=strategy,
    )

    return pipeline
