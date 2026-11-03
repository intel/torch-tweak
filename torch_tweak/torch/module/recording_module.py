# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Inplace wrap modules."""

import itertools
import json
import logging
import tempfile
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from torch_tweak.torch.config import TorchTweakConfig
from torch_tweak.torch.config import config as global_config
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.utils.path_utils import sanitize_filename
from torch_tweak.utils.permissions import secure_file

INPUT_METADATA_PREFIX = "input"
OUTPUT_METADATA_PREFIX = "output"
INDEX_FILE = "recording.json"

logger = logging.getLogger(__name__)

Sample = tuple[tuple, dict]


class RecordingModule:
    """Module that records samples for tuning."""

    def __init__(
        self,
        module: torch.nn.Module,
        name: str,
        config: TorchTweakConfig | None = None,
        samples_dir: Path | None = None,
    ) -> None:
        """Initialize BaseModule.

        Args:
            module: module to be tuned.
            name: name of the module.
            config: Configuration for the module, if not provided, global config is used.
            samples_dir: Directory that keeps the samples and an index of the graphs after the process exits.
                None stores samples in a temporary directory removed by cleanup.
        """
        super().__init__()
        if not isinstance(module, torch.nn.Module):
            raise TypeError("Only torch modules are supported.")

        self._module = module
        self._name = name
        self._config = config if config is not None else global_config
        self._forward_call = module.__call__

        self._samples = defaultdict(list)
        self._total_num_samples = 0

        if samples_dir is None:
            # make temp directory to store samples, it has to be a field so that is not prematurely removed
            tempdir_prefix = sanitize_filename(self._name)
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"{tempdir_prefix}_")
            self._samples_dir = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self._samples_dir = Path(samples_dir)
            self._samples_dir.mkdir(parents=True, exist_ok=True)
        self._graph_specs: OrderedDict[SampleMetadata, GraphSpec] = OrderedDict()
        self._graphs_counter = itertools.count()

    @classmethod
    def load(
        cls, module: torch.nn.Module, name: str, samples_dir: Path, config: TorchTweakConfig | None = None
    ) -> "RecordingModule":
        """Restores a recording written to samples_dir by an earlier process."""
        recording = cls(module, name, config=config, samples_dir=samples_dir)
        index = json.loads((recording._samples_dir / INDEX_FILE).read_text(encoding="utf-8"))
        for entry in index["graphs"]:
            graph_spec = GraphSpec.from_dict(entry["graph_spec"])
            recording._graph_specs[graph_spec.input_spec] = graph_spec
            recording._samples[graph_spec.input_spec] = [recording._samples_dir / file for file in entry["samples"]]
        recording._total_num_samples = index["total_num_samples"]
        recording._graphs_counter = itertools.count(len(recording._graph_specs))
        return recording

    @staticmethod
    def has_recording(samples_dir: Path) -> bool:
        """True when samples_dir holds a recording written with a samples_dir."""
        return (Path(samples_dir) / INDEX_FILE).is_file()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a sample and run the module.

        Before calling forward, we need to make input sample metadata and a copy of args and kwargs, since the
        forward call may have side effects on the inputs (like KV cache in LLM models).
        Inputs are copied only when the sample is going to be stored.
        """
        logger.debug("Calling recording %s module.", self._name)
        inputs_metadata = SampleMetadata.from_inputs(
            args, kwargs, strict=self._config.strict_mode, exact_shapes=self._config.exact_shapes
        )
        stores = len(self._samples.get(inputs_metadata, ())) < self._config.max_num_samples_stored
        inputs = self._copy_inputs(args, kwargs) if stores else None
        outputs = self._forward_call(*args, **kwargs)
        outputs_metadata = SampleMetadata.from_outputs(outputs, strict=self._config.strict_mode)

        self.record_sample(inputs, inputs_metadata, outputs_metadata)

        return outputs

    @property
    def config(self) -> TorchTweakConfig:
        """The configuration the graphs of this recording are keyed with."""
        return self._config

    @property
    def device(self) -> torch.device:
        """The device of the module."""
        return next(self._module.parameters()).device

    def record_sample(self, inputs, inputs_metadata, outputs_metadata) -> None:
        """Record a sample from the module. inputs None counts the call without storing it."""
        if inputs_metadata in self._graph_specs:
            # graphs share same hash but can have different min, max seen shapes
            self._graph_specs[inputs_metadata].update_shapes_seen(inputs_metadata, outputs_metadata)
        else:
            # create a new graph spec for the sample metadata
            graph_name = f"{next(self._graphs_counter)}"
            if inputs_metadata.llm_phase:
                graph_name += f" (LLM phase {inputs_metadata.llm_phase})"
            self._graph_specs[inputs_metadata] = GraphSpec(
                name=graph_name, input_spec=inputs_metadata, output_spec=outputs_metadata
            )

        if inputs is not None and len(self._samples[inputs_metadata]) < self._config.max_num_samples_stored:
            sample_path = self._samples_dir / f"{self._total_num_samples}.pt"
            self._total_num_samples += 1
            torch.save(inputs, sample_path)
            secure_file(sample_path)
            self._samples[inputs_metadata].append(sample_path)
            if self._temp_dir is None:
                self._write_index()

    def _write_index(self) -> None:
        """Writes the graphs and their sample files, so a later process can load this recording."""
        index = {
            "total_num_samples": self._total_num_samples,
            "graphs": [
                {
                    "graph_spec": graph_spec.to_dict(),
                    "samples": [path.name for path in self._samples[input_spec]],
                }
                for input_spec, graph_spec in self._graph_specs.items()
            ],
        }
        index_path = self._samples_dir / INDEX_FILE
        staging = index_path.with_suffix(".tmp")
        staging.write_text(json.dumps(index), encoding="utf-8")
        staging.replace(index_path)
        secure_file(index_path)

    @property
    def is_ready_for_optimization(self) -> bool:
        """Check if the module is ready for tuning.

        All graphs would have proper amount of samples.
        """
        return all(len(samples) >= self._config.min_num_samples for samples in self._samples.values())

    @property
    def graph_specs(self) -> list[GraphSpec]:
        """The graph specs."""
        return list(self._graph_specs.values())

    def samples_for_graph_spec(self, graph_spec: GraphSpec) -> list[Sample]:
        """Get the samples.

        Samples are arbitrary forward() arguments, so they cannot be restored
        weights-only. Unlike a checkpoint, these files are written by this instance into
        its own temporary directory and never come from outside the process.
        """
        result = []
        for path in self._samples[graph_spec.input_spec]:
            result.append(torch.load(path, weights_only=False))  # nosec B614
        return result

    def cleanup(self) -> None:
        """Release the temporary directory holding recorded samples. A samples_dir is kept."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    @staticmethod
    def _copy_inputs(args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Create deep copies of args and kwargs.

        Args:
            args: Positional arguments tuple.
            kwargs: Keyword arguments dict.

        Returns:
            Tuple containing deep copies of args and kwargs.

        Raises:
            RuntimeError: If cannot copy model inputs with tip to fix it
        """
        try:
            return deepcopy(args), deepcopy(kwargs)
        except RuntimeError as e:
            # raise helpful error message
            raise RuntimeError(
                "Cannot copy model inputs. Model is not in inference mode. "
                "Tip: you can use \n---\nwith torch.no_grad():\n    model()\n---\ncontext manager to fix this."
            ) from e
