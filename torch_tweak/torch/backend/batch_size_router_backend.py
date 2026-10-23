# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Backend that sends each call to the backend tuned for its batch size."""

from pathlib import Path
from typing import Any, Self

import torch
import torch.nn as nn

from torch_tweak.torch.backend.backend import Backend, BackendState
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.utils.trace import annotate

# The largest batch size of a route, or None for every larger batch size.
Route = tuple[int | None, Backend]


class BatchSizeRouterBackend(Backend):
    """Holds several built backends and sends each call to the one tuned for its batch size.

    HighestThroughputStrategy creates it when different backends win at different batch sizes.
    routes are in ascending order. A call goes to the first route that holds its batch size.
    The last route takes every larger batch size.
    """

    STATE_TYPE = "type"
    STATE_ROUTES = "routes"
    STATE_INPUT_SPEC = "input_spec"

    def __init__(self, routes: list[Route], input_spec: SampleMetadata):
        """Initializes backend.

        Args:
            routes: (largest batch size, backend) pairs in ascending order. The last bound is None.
            input_spec: Input spec of the graph spec, used to read the batch size of a call.
        """
        super().__init__()
        if not routes or routes[-1][0] is not None:
            raise ValueError("The last route must take every larger batch size")
        self._routes = routes
        self._input_spec = input_spec
        self._device = routes[0][1].device
        if all(backend.is_active for _, backend in routes):
            self.state = BackendState.ACTIVE

    @property
    def routes(self) -> list[Route]:
        """(largest batch size, backend) pairs in ascending order."""
        return self._routes

    @property
    def requires_original_module(self) -> bool:  # type: ignore[override]
        """True when any routed backend runs the original module."""
        return any(backend.requires_original_module for _, backend in self._routes)

    def key(self) -> str:
        """Returns the key of the backend."""
        return "_".join([self.__class__.__name__, *(backend.key() for _, backend in self._routes)])

    def describe(self) -> str:
        """Returns the description of the backend."""
        parts = []
        smallest = 1
        for largest, backend in self._routes:
            sizes = f"{smallest}+" if largest is None else f"{smallest}-{largest}"
            parts.append(f"{sizes}: {backend.describe()}")
            smallest = (largest or 0) + 1
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def backend_for(self, batch_size: int) -> Backend:
        """Returns the backend that runs batch_size."""
        for largest, backend in self._routes:
            if largest is None or batch_size <= largest:
                return backend
        raise AssertionError("The last route takes every batch size")

    def _batch_size_of(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
        """Reads the batch size from the first input tensor with a batch axis."""
        for locator, tensor_spec in self._input_spec.tensor_data:
            multipliers = tensor_spec.get_batch_axis_multipliers()
            if not multipliers:
                continue
            value = locator.get_value(args if tensor_spec.name.startswith("args") else kwargs)
            axis, multiplier = next(iter(multipliers.items()))
            return value.shape[axis] // multiplier
        return None

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        raise RuntimeError(f"{self.name} is created from backends that are already built")

    def _activate(self) -> None:
        for _, backend in self._routes:
            backend.activate()

    def _deactivate(self) -> None:
        for _, backend in self._routes:
            backend.deactivate()

    def _deploy(self) -> None:
        for _, backend in self._routes:
            backend.deploy(self._device)

    @annotate(message="BatchSizeRouterBackend.infer", domain="Torch Tweak")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        batch_size = self._batch_size_of(args, kwargs)
        backend = self._routes[-1][1] if batch_size is None else self.backend_for(batch_size)
        return backend.infer(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Returns the state_dict of the backend."""
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_ROUTES: [(largest, backend.to_dict()) for largest, backend in self._routes],
            self.STATE_INPUT_SPEC: self._input_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict[str, Any]) -> Self:
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")
        backend_classes = {backend_class.__name__: backend_class for backend_class in Backend.__subclasses__()}
        routes: list[Route] = []
        for largest, backend_state in state_dict[cls.STATE_ROUTES]:
            backend_type = backend_state[cls.STATE_TYPE]
            if backend_type not in backend_classes:
                raise ValueError(f"Unknown backend type: {backend_type}")
            routes.append((largest, backend_classes[backend_type].from_dict(module, backend_state)))
        backend = cls(routes, SampleMetadata.from_dict(state_dict[cls.STATE_INPUT_SPEC]))
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
