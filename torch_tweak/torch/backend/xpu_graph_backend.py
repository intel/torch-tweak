# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""XPU graph backend."""

import warnings
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Self

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_map_only

from torch_tweak.torch.backend.backend import Backend, BackendConfig, BackendState
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.utils.trace import annotate

logger = getLogger(__name__)


@dataclass
class XPUGraphBackendConfig(BackendConfig):
    """Configuration for an XPU graph backend.

    Args:
        autocast_enabled: If True, enable autocast while recording. Replay runs the recorded casts.
            The autocast cache is disabled so later eager calls cannot reuse tensors captured in the graph.
        autocast_dtype: The dtype to use for autocast.
    """

    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> Self:
        """Convert dict to XPUGraphBackendConfig."""
        return cls(**state_dict)


class XPUGraphBackend(Backend):
    """Backend that records one XPU graph while tuning and replays it for inference."""

    _devices: ClassVar[list[str]] = ["xpu"]
    requires_original_module: ClassVar[bool] = True

    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_DEVICE = "device"

    def __init__(self, config: XPUGraphBackendConfig | None = None):
        """Initializes backend.

        Args:
            config: Recording and autocast configuration.
        """
        super().__init__()
        self._config: XPUGraphBackendConfig = config or XPUGraphBackendConfig()
        self._orig_module: nn.Module | None = None
        self._data: list[Sample] | None = None
        self._graph: torch.xpu.XPUGraph | None = None
        self._static_args: tuple[list[Any], TreeSpec] | None = None
        self._static_kwargs: tuple[list[Any], TreeSpec] | None = None
        self._captured: Any = None
        self._warned_eager = False

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Records the XPU graph once and checks it against eager."""
        if graph_spec.has_dynamic_axis():
            raise RuntimeError("XPU graph replay needs static tensor shapes")
        for sample_args, sample_kwargs in data:
            _require_xpu_tensors(sample_args, sample_kwargs)
        self._config.save_to_file(cache_dir)
        self._orig_module = module
        self._data = data
        self._record()
        return self

    def _activate(self) -> None:
        """Records the graph again after a checkpoint load or a deactivate."""
        self._release_graph()
        self._record()

    def _deactivate(self) -> None:
        """Drops the recorded graph."""
        self._release_graph()

    def _deploy(self) -> None:
        """Records the graph for deployment and drops the tuning samples."""
        self._activate()
        self._data = None

    @annotate(message="XPUGraphBackend.infer", domain="Torch Tweak")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Replays the recorded graph.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            The replayed outputs. Tensor outputs are clones, so the next replay can reuse the static buffers.
            Inputs the recording cannot take run on the original module.
        """
        with torch.no_grad():
            if not self._load_inputs(args, kwargs):
                return self._infer_eager(args, kwargs)
            self._graph.replay()
            # replay and this clone are both on the current stream, in submission order.
            return _clone_tree(self._captured)

    def _load_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        """Copies args and kwargs into the recorded inputs. Copies nothing and returns False when they do not fit."""
        assert self._static_args is not None and self._static_kwargs is not None
        live_args = _live_leaves(self._static_args, args)
        live_kwargs = _live_leaves(self._static_kwargs, kwargs)
        if live_args is None or live_kwargs is None:
            return False
        _copy_into(self._static_args[0], live_args)
        _copy_into(self._static_kwargs[0], live_kwargs)
        return True

    def _infer_eager(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Runs the original module on inputs that differ from the recorded call."""
        assert self._orig_module is not None
        if not self._warned_eager:
            logger.warning("XPU graph inputs do not match the recorded call. Running the original module.")
            self._warned_eager = True
        with self._autocast():
            return self._orig_module(*args, **kwargs)

    def _record(self) -> None:
        """Warm up, record, and reject the graph when replay disagrees with eager."""
        assert self._orig_module is not None
        assert self._data

        self._orig_module.to(self._device)
        args, kwargs = self._data[0]
        static_args = _clone_tree(args)
        static_kwargs = _clone_tree(kwargs)

        logger.info("Recording XPU graph.")
        try:
            self._graph = torch.xpu.XPUGraph()
            with self._autocast(), torch.no_grad():
                # Warmup. The first oneDNN GEMM for a shape selects an algorithm and allocates its workspace.
                # That work stays outside the recording.
                self._orig_module(*_clone_tree(static_args), **_clone_tree(static_kwargs))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    # Records this call on default_capture_stream.
                    # Exiting the block ends the recording.
                    with torch.xpu.graph(self._graph):
                        captured = self._orig_module(*static_args, **static_kwargs)
            if any("XPU Graph is empty" in str(item.message) for item in caught):
                raise RuntimeError("XPU graph capture recorded no device work")
            # Recording runs on default_capture_stream. replay runs on the current stream.
            # Both write captured, so the current stream waits for the recording stream.
            stream = torch.xpu.graph.default_capture_stream
            assert stream is not None
            torch.xpu.current_stream().wait_stream(stream)
            self._static_args = tree_flatten(static_args)
            self._static_kwargs = tree_flatten(static_kwargs)
            self._captured = captured
            self._replay_matches_eager(static_args, static_kwargs)
        except Exception:
            self._release_graph()
            raise
        logger.info("XPU graph recorded.")

    def _replay_matches_eager(self, static_args: Any, static_kwargs: Any) -> None:
        """Copies every sample and a perturbed copy into the static inputs and compares replay to eager."""
        assert self._orig_module is not None
        assert self._data
        assert self._graph is not None
        for sample_args, sample_kwargs in self._data:
            for args, kwargs in (
                (sample_args, sample_kwargs),
                (_perturb(sample_args), _perturb(sample_kwargs)),
            ):
                if not self._load_inputs(args, kwargs):
                    raise RuntimeError("XPU graph inputs do not match the captured call")
                # replay may write static_args in place. Eager must see the earlier values.
                frozen_args = _clone_tree(static_args)
                frozen_kwargs = _clone_tree(static_kwargs)
                self._graph.replay()
                with self._autocast(), torch.no_grad():
                    # The module may return an input. Compare a tensor replay did not write.
                    expected = _clone_tree(self._orig_module(*frozen_args, **frozen_kwargs))
                _assert_close(self._captured, expected)

    def _autocast(self) -> torch.autocast:
        """Returns the autocast context used for warmup, capture, and the eager check."""
        # The autocast cache aliases weight casts. Those addresses are part of the recording.
        return torch.autocast(
            device_type=self._device.type,
            dtype=self._config.autocast_dtype,
            enabled=self._config.autocast_enabled,
            cache_enabled=False,
        )

    def _release_graph(self) -> None:
        """Drops the graph and the static tensors it captured."""
        if self._graph is not None:
            self._graph.reset()
        self._graph = None
        self._static_args = None
        self._static_kwargs = None
        self._captured = None

    def to_dict(self) -> dict[str, Any]:
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_DATA: self._data,
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: nn.Module | None, state_dict: dict[str, Any]) -> Self:
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")
        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = XPUGraphBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])
        backend = cls(config=config)
        backend._data = state_dict[cls.STATE_DATA]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


def _clone_tree(tree: Any) -> Any:
    """Clones tensors and leaves every other leaf as it is."""
    return tree_map_only(torch.Tensor, lambda tensor: tensor.detach().clone(), tree)


def _live_leaves(static: tuple[list[Any], TreeSpec], live: Any) -> list[Any] | None:
    """Returns the leaves of live, or None when they cannot be copied into the inputs recorded by the graph."""
    static_leaves, static_spec = static
    live_leaves, live_spec = tree_flatten(live)
    if static_spec != live_spec:
        return None
    for dst, src in zip(static_leaves, live_leaves, strict=True):
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
            if (dst.shape, dst.dtype, dst.device) != (src.shape, src.dtype, src.device):
                return None
        elif isinstance(dst, torch.Tensor) or isinstance(src, torch.Tensor):
            # != of a tensor and a non-tensor returns a tensor.
            # The next branch would not reject this pair.
            return None
        elif dst != src:
            return None
    return live_leaves


def _copy_into(static_leaves: list[Any], live_leaves: list[Any]) -> None:
    """Copies live tensors into the tensors recorded by the graph. live_leaves comes from _live_leaves."""
    for dst, src in zip(static_leaves, live_leaves, strict=True):
        if isinstance(dst, torch.Tensor):
            dst.copy_(src)


def _require_xpu_tensors(*trees: Any) -> None:
    """Rejects a recording whose inputs are not XPU tensors."""
    for tree in trees:
        leaves, _ = tree_flatten(tree)
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor) and leaf.device.type != "xpu":
                raise RuntimeError("XPU graph capture needs XPU tensor inputs")


def _perturb(tree: Any) -> Any:
    """Returns a tree with different tensor values and the same other leaves."""
    return tree_map_only(torch.Tensor, _perturb_tensor, tree)


def _perturb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Returns a tensor with different values and the same shape."""
    cloned = tensor.detach().clone()
    if cloned.is_floating_point() or cloned.is_complex():
        return cloned + torch.randn_like(cloned)
    if cloned.ndim == 0:
        return ~cloned if cloned.dtype == torch.bool else cloned + 1
    return cloned.roll(1, 0)


def _assert_close(actual: Any, expected: Any) -> None:
    """Raises when replay outputs are too far from eager outputs."""
    actual_leaves, actual_spec = tree_flatten(actual)
    expected_leaves, expected_spec = tree_flatten(expected)
    if actual_spec != expected_spec:
        raise RuntimeError("XPU graph output structure does not match eager")
    for got, ref in zip(actual_leaves, expected_leaves, strict=True):
        if not isinstance(ref, torch.Tensor):
            if got != ref:
                raise RuntimeError("XPU graph output does not match eager")
            continue
        if not isinstance(got, torch.Tensor) or got.shape != ref.shape or got.dtype != ref.dtype:
            raise RuntimeError("XPU graph output tensor does not match eager")
        rtol, atol = (1e-2, 1e-2) if ref.dtype in (torch.float16, torch.bfloat16) else (1e-3, 1e-3)
        if torch.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
            continue
        if got.is_floating_point() or got.is_complex():
            difference = (got - ref).abs().max().item()
            raise RuntimeError(f"XPU graph output differs from eager (max abs {difference})")
        raise RuntimeError("XPU graph output differs from eager")
