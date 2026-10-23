# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""XPU graph backend."""

import copy
import warnings
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Self

import torch
import torch.nn as nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_map, tree_map_only, tree_unflatten

from torch_tweak.torch.backend.backend import Backend, BackendConfig, BackendState
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tensor_spec import logical_batch_bounds
from torch_tweak.utils.trace import annotate

logger = getLogger(__name__)

# Maps a tensor axis to its batch multiplier. The axis holds multiplier * batch_size elements.
BatchAxes = dict[int, int]


@dataclass
class XPUGraphBackendConfig(BackendConfig):
    """Configuration for an XPU graph backend.

    Args:
        autocast_enabled: If True, enable autocast while recording. Replay runs the recorded casts.
            The autocast cache is disabled so later eager calls cannot reuse tensors captured in the graph.
        autocast_dtype: The dtype to use for autocast.
        max_recorded_batch_size: Caps the largest batch size recorded for a graph spec with a batch axis.
            Graphs are recorded for 1, 2, 4, every multiple of 8 below the limit, and the limit itself.
            The limit is this value or the largest batch size seen during tuning, whichever is smaller.
            A larger batch replays in chunks of the largest recorded size.
        compile_with_inductor: If True, record the kernels Torch Inductor generates for the module
            instead of the eager kernels. Compilation runs during the warmup before each recording.
    """

    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None
    max_recorded_batch_size: int = 256
    compile_with_inductor: bool = False

    def __post_init__(self) -> None:
        """Rejects a max_recorded_batch_size below 1."""
        if self.max_recorded_batch_size < 1:
            raise ValueError(f"max_recorded_batch_size must be at least 1, got {self.max_recorded_batch_size}")

    @classmethod
    def from_dict(cls, state_dict: dict[str, Any]) -> Self:
        """Convert dict to XPUGraphBackendConfig."""
        return cls(**state_dict)


@dataclass
class _Recording:
    """One recorded graph and the tensors it reads and writes."""

    graph: torch.xpu.XPUGraph
    static_leaves: list[Any]
    captured_leaves: list[Any]


@dataclass
class _Layout:
    """Batch axes of every flattened input and output leaf. Non-tensor leaves have no batch axes."""

    input_spec: TreeSpec
    input_axes: list[BatchAxes]
    output_spec: TreeSpec
    output_axes: list[BatchAxes]

    @property
    def has_batch_axis(self) -> bool:
        return any(self.input_axes)

    @property
    def can_chunk(self) -> bool:
        """True when every leaf has at most one batch axis. Replay concatenates chunks along that axis."""
        return all(len(axes) <= 1 for axes in self.input_axes + self.output_axes)


class XPUGraphBackend(Backend):
    """Backend that records XPU graphs while tuning and replays them for inference.

    A graph spec with static shapes records one graph.
    A graph spec with a batch axis records graphs for 1, 2, 4, every multiple of 8 below the largest batch size seen, and that size.
    A call pads its batch with zeros up to the nearest recorded size and gets the real rows back.
    """

    _devices: ClassVar[list[str]] = ["xpu"]
    requires_original_module: ClassVar[bool] = True

    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_DEVICE = "device"
    STATE_INPUT_SPEC = "input_spec"

    def __init__(self, config: XPUGraphBackendConfig | None = None):
        """Initializes backend.

        Args:
            config: Recording and autocast configuration.
        """
        super().__init__()
        self._config: XPUGraphBackendConfig = config or XPUGraphBackendConfig()
        self._orig_module: nn.Module | None = None
        self._data: list[Sample] | None = None
        self._input_spec: SampleMetadata | None = None
        self._layout: _Layout | None = None
        # Keyed by batch size in ascending order. A graph spec without a batch axis has one recording under key 1.
        self._recordings: dict[int, _Recording] = {}
        # The callable that warmup and recording run: the original module or its Inductor compilation.
        self._recorded_module: Any = None
        self._pool: Any = None
        self._warned_eager = False

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    @property
    def recorded_batch_sizes(self) -> list[int]:
        """Batch sizes with a recorded graph, in ascending order."""
        return list(self._recordings)

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Records the XPU graphs and checks them against eager."""
        tensor_specs = graph_spec.input_spec.tensor_specs + graph_spec.output_spec.tensor_specs
        if any(tensor_spec.has_dynamic_axis() for tensor_spec in tensor_specs):
            raise RuntimeError("XPU graph replay cannot change a dynamic axis other than the batch axis")
        for sample_args, sample_kwargs in data:
            _require_xpu_tensors(sample_args, sample_kwargs)
        self._config.save_to_file(cache_dir)
        self._orig_module = module
        self._data = data
        self._input_spec = graph_spec.input_spec if graph_spec.input_spec.has_batch_axis() else None
        self._record()
        return self

    def _activate(self) -> None:
        """Records the graphs again after a checkpoint load or a deactivate."""
        self._release_graphs()
        self._record()

    def _deactivate(self) -> None:
        """Drops the recorded graphs."""
        self._release_graphs()

    def _deploy(self) -> None:
        """Records the graphs for deployment and drops the tuning samples."""
        self._activate()
        self._data = None

    @annotate(message="XPUGraphBackend.infer", domain="Torch Tweak")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Replays the recorded graph that fits the call.

        _replay_rows clones tensor outputs so the next replay reuses the static buffers.
        """
        with torch.no_grad():
            outputs = self._replay((args, kwargs))
            if outputs is None:
                return self._infer_eager(args, kwargs)
            return outputs

    def _replay(self, sample: Sample) -> Any | None:
        """Replays the recordings for sample. Returns None when no recording fits."""
        assert self._layout is not None
        leaves, spec = tree_flatten(sample)
        batch_size = self._batch_size_of(leaves, spec)
        if batch_size is None:
            return None
        largest = next(reversed(self._recordings))
        if batch_size <= largest:
            return tree_unflatten(self._replay_rows(leaves, 0, batch_size), self._layout.output_spec)
        if not self._layout.can_chunk:
            return None
        chunks = [
            self._replay_rows(leaves, start, min(largest, batch_size - start))
            for start in range(0, batch_size, largest)
        ]
        merged = [
            torch.cat([chunk[index] for chunk in chunks], dim=next(iter(axes))) if axes else chunks[0][index]
            for index, axes in enumerate(self._layout.output_axes)
        ]
        return tree_unflatten(merged, self._layout.output_spec)

    def _batch_size_of(self, leaves: list[Any], spec: TreeSpec) -> int | None:
        """Returns the batch size of a call, or None when its leaves do not fit the recordings."""
        assert self._layout is not None
        if spec != self._layout.input_spec:
            return None
        batch_size = 1
        for leaf, axes in zip(leaves, self._layout.input_axes, strict=True):
            if axes:
                if not isinstance(leaf, torch.Tensor):
                    return None
                axis, multiplier = next(iter(axes.items()))
                batch_size, remainder = divmod(leaf.shape[axis], multiplier)
                if batch_size == 0 or remainder:
                    return None
                break
        reference = next(iter(self._recordings.values())).static_leaves
        for leaf, static, axes in zip(leaves, reference, self._layout.input_axes, strict=True):
            if not _leaf_fits(leaf, static, axes, batch_size):
                return None
        return batch_size

    def _replay_rows(self, leaves: list[Any], start: int, rows: int, size: int | None = None) -> list[Any]:
        """Replays rows start to start + rows of the call, padded with zeros. Returns cloned output leaves.

        size: the recorded batch size to replay. Defaults to the smallest one that holds rows.
        """
        assert self._layout is not None
        if size is None:
            size = next(recorded for recorded in self._recordings if recorded >= rows)
        recording = self._recordings[size]
        for live, static, axes in zip(leaves, recording.static_leaves, self._layout.input_axes, strict=True):
            if not isinstance(static, torch.Tensor):
                continue
            if not axes:
                static.copy_(live)
                continue
            source = live[_rows_index(live.ndim, axes, start, rows)]
            if rows == size:
                static.copy_(source)
                continue
            if len(axes) == 1:
                # Stale rows from an earlier call or a larger graph sharing the buffer must not reach the padding.
                static[_rows_index(static.ndim, axes, rows, size - rows)].zero_()
            else:
                static.zero_()
            static[_rows_index(static.ndim, axes, 0, rows)].copy_(source)
        recording.graph.replay()
        # replay and these clones are both on the current stream, in submission order.
        return [
            captured[_rows_index(captured.ndim, axes, 0, rows)].clone() if axes else _clone_leaf(captured)
            for captured, axes in zip(recording.captured_leaves, self._layout.output_axes, strict=True)
        ]

    def _infer_eager(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Runs the original module on inputs that differ from the recorded calls."""
        assert self._orig_module is not None
        if not self._warned_eager:
            logger.warning("XPU graph inputs do not match the recorded calls. Running the original module.")
            self._warned_eager = True
        with self._autocast():
            return self._orig_module(*args, **kwargs)

    def _eager(self, sample: Sample) -> Any:
        """Runs the original module on a copy of sample, so in-place writes do not change sample."""
        assert self._orig_module is not None
        args, kwargs = _clone_tree(sample)
        with self._autocast(), torch.no_grad():
            return _clone_tree(self._orig_module(*args, **kwargs))

    def _record(self) -> None:
        """Records every batch size, then rejects the backend when replay disagrees with eager."""
        assert self._orig_module is not None
        assert self._data
        self._orig_module.to(self._device)
        try:
            self._layout = self._detect_layout()
            self._recorded_module = (
                torch.compile(self._orig_module) if self._config.compile_with_inductor else self._orig_module
            )
            self._pool = torch.xpu.graph_pool_handle()
            sizes = self._batch_sizes_to_record() if self._layout.has_batch_axis else [1]
            largest = sizes[-1]
            buffer = tree_flatten(_clone_tree(self._sample_of_size(largest)))[0]
            # The largest graph sizes the shared pool. Smaller graphs reuse its blocks.
            for size in reversed(sizes):
                try:
                    self._recordings[size] = self._record_size(size, largest, buffer)
                except torch.OutOfMemoryError:
                    if size == sizes[0]:
                        raise
                    logger.warning("Out of memory while recording an XPU graph for batch size %d. Skipping it.", size)
                    torch.xpu.empty_cache()
            self._recordings = dict(sorted(self._recordings.items()))
            self._replay_matches_eager()
        except Exception:
            self._release_graphs()
            raise
        logger.info("XPU graphs recorded for batch sizes %s.", self.recorded_batch_sizes)

    def _batch_sizes_to_record(self) -> list[int]:
        """1, 2, 4, then multiples of 8, through the largest batch size seen.

        The last size is that batch, or max_recorded_batch_size when it is smaller.
        Above batch size 8, padding adds at most 7 rows. Every power of two profiled by
        HighestThroughputStrategy is on this grid, so profiling replays without padding.
        """
        assert self._input_spec is not None
        _, seen = logical_batch_bounds(self._input_spec.tensor_specs)
        cap = min(self._config.max_recorded_batch_size, seen)
        return [size for size in [1, 2, 4, *range(8, cap, 8)] if size < cap] + [cap]

    def _sample_of_size(self, size: int) -> Sample:
        """Returns the first tuning sample with batch size size."""
        assert self._data
        args, kwargs = self._data[0]
        if self._input_spec is None:
            return _clone_tree((args, kwargs))
        return self._input_spec.make_batch(args, kwargs, size)

    def _detect_layout(self) -> _Layout:
        """Finds the batch axes of every input and output leaf from samples of batch size 1 and 2."""
        one = self._sample_of_size(1)
        one_leaves, input_spec = tree_flatten(one)
        outputs_one, output_spec = tree_flatten(self._eager(one))
        if self._input_spec is None:
            return _Layout(input_spec, [{} for _ in one_leaves], output_spec, [{} for _ in outputs_one])
        two = self._sample_of_size(2)
        two_leaves, two_spec = tree_flatten(two)
        outputs_two, outputs_two_spec = tree_flatten(self._eager(two))
        if two_spec != input_spec or outputs_two_spec != output_spec:
            raise RuntimeError("XPU graph inputs or outputs change structure with the batch size")
        layout = _Layout(
            input_spec=input_spec,
            input_axes=[_batch_axes(a, b) for a, b in zip(one_leaves, two_leaves, strict=True)],
            output_spec=output_spec,
            output_axes=[_batch_axes(a, b) for a, b in zip(outputs_one, outputs_two, strict=True)],
        )
        if not layout.has_batch_axis:
            raise RuntimeError("XPU graph found no batch axis in the inputs")
        outputs = zip(outputs_one, layout.output_axes, strict=True)
        if any(isinstance(leaf, torch.Tensor) and not axes for leaf, axes in outputs):
            raise RuntimeError("XPU graph output has no batch axis, so padded rows cannot be removed")
        for one, two in zip(outputs_one, outputs_two, strict=True):
            one_shapes = [tuple(tensor.shape) for tensor in _state_tensors(one)]
            two_shapes = [tuple(tensor.shape) for tensor in _state_tensors(two)]
            if one_shapes != two_shapes:
                raise RuntimeError(
                    "XPU graph output holds tensors that are not returned as tensors, so padded rows cannot be removed"
                )
        return layout

    def _record_size(self, size: int, largest: int, buffer: list[Any]) -> _Recording:
        """Warm up and record the graph for one batch size. buffer holds the input leaves at batch size largest."""
        assert self._orig_module is not None
        assert self._layout is not None
        static_leaves = self._static_leaves(size, largest, buffer)
        static_args, static_kwargs = tree_unflatten(static_leaves, self._layout.input_spec)
        graph = torch.xpu.XPUGraph()
        try:
            with self._autocast(), torch.no_grad():
                # Warmup. The first oneDNN GEMM for a shape selects an algorithm and allocates its workspace.
                # That work stays outside the recording.
                # With compile_with_inductor, this call also compiles for the shape.
                self._recorded_module(*_clone_tree(static_args), **_clone_tree(static_kwargs))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    # Records this call on default_capture_stream.
                    # Exiting the block ends the recording.
                    with torch.xpu.graph(graph, pool=self._pool):
                        captured = self._recorded_module(*static_args, **static_kwargs)
            if any("XPU Graph is empty" in str(item.message) for item in caught):
                raise RuntimeError("XPU graph capture recorded no device work")
            captured_leaves, captured_spec = tree_flatten(captured)
            if captured_spec != self._layout.output_spec:
                raise RuntimeError("XPU graph outputs change structure with the batch size")
            for leaf, axes in zip(captured_leaves, self._layout.output_axes, strict=True):
                if isinstance(leaf, torch.Tensor) and any(leaf.shape[a] != m * size for a, m in axes.items()):
                    raise RuntimeError("XPU graph batch axis does not scale with the batch size")
        except Exception:
            graph.reset()
            raise
        # Recording runs on default_capture_stream. replay runs on the current stream.
        # Both write captured, so the current stream waits for the recording stream.
        stream = torch.xpu.graph.default_capture_stream
        assert stream is not None
        torch.xpu.current_stream().wait_stream(stream)
        return _Recording(graph=graph, static_leaves=static_leaves, captured_leaves=captured_leaves)

    def _static_leaves(self, size: int, largest: int, buffer: list[Any]) -> list[Any]:
        """Returns the input leaves recorded for size. buffer holds the input leaves at batch size largest.

        A leaf without a batch axis is the buffer leaf. A leaf batched only along axis 0 is a prefix view of
        the buffer leaf, which is contiguous. Graphs share that memory because replays run one at a time.
        Any other batched leaf below largest gets its own tensor.
        """
        assert self._layout is not None
        own: list[Any] | None = None
        leaves = []
        for index, (shared, axes) in enumerate(zip(buffer, self._layout.input_axes, strict=True)):
            if not isinstance(shared, torch.Tensor) or not axes or size == largest:
                leaves.append(shared)
            elif axes.keys() == {0}:
                leaves.append(shared[: axes[0] * size])
            else:
                if own is None:
                    own = tree_flatten(_clone_tree(self._sample_of_size(size)))[0]
                leaves.append(own[index])
        return leaves

    def _replay_matches_eager(self) -> None:
        """Compares replay to eager on every tuning sample, a perturbed copy, and replayed windows."""
        assert self._data
        assert self._layout is not None
        for sample in self._data:
            for checked in (sample, _perturb(sample)):
                outputs = self._replay(checked)
                if outputs is None:
                    raise RuntimeError("XPU graph inputs do not match the captured call")
                _assert_close(outputs, self._eager(checked))
        if self._layout.has_batch_axis:
            self._windows_match_eager()

    def _windows_match_eager(self) -> None:
        """Replays windows of one perturbed batch on every graph and compares them to one eager run.

        Each graph replays its full size and a shorter prefix padded with zeros.
        A padded prefix fails when a row reads a later row or a padding row.
        When chunking is possible, the graph also replays a window that starts after the first row.
        Chunking copies that window to the front of the graph, so a row that reads an earlier row fails.
        A grid that recorded only one row is checked on a two-row batch, because that is the smallest chunked call.
        """
        assert self._layout is not None
        rows_in_reference = self.recorded_batch_sizes[-1]
        if self._layout.can_chunk and rows_in_reference == 1:
            rows_in_reference = 2
        reference = _perturb(self._sample_of_size(rows_in_reference))
        leaves, _ = tree_flatten(reference)
        expected, _ = tree_flatten(self._eager(reference))
        for size in self.recorded_batch_sizes:
            windows = [(0, size)]
            half = (size + 1) // 2
            if half != size:
                windows.append((0, half))
            if self._layout.can_chunk and size > 1:
                start = size // 2
                windows.append((start, size - start))
            elif self._layout.can_chunk and rows_in_reference > size:
                windows.append((1, 1))
            for start, rows in windows:
                outputs = self._replay_rows(leaves, start, rows, size)
                window = [
                    ref[_rows_index(ref.ndim, axes, start, rows)] if axes else ref
                    for ref, axes in zip(expected, self._layout.output_axes, strict=True)
                ]
                _assert_close(outputs, window)

    def _autocast(self) -> torch.autocast:
        """Returns the autocast context used for warmup, capture, and the eager check."""
        # The autocast cache aliases weight casts. Those addresses are part of the recording.
        return torch.autocast(
            device_type=self._device.type,
            dtype=self._config.autocast_dtype,
            enabled=self._config.autocast_enabled,
            cache_enabled=False,
        )

    def _release_graphs(self) -> None:
        """Drops the graphs, the static tensors they captured, and their memory pool."""
        for recording in self._recordings.values():
            recording.graph.reset()
        self._recordings = {}
        self._recorded_module = None
        self._layout = None
        self._pool = None

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
            self.STATE_INPUT_SPEC: None if self._input_spec is None else self._input_spec.to_dict(),
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
        input_spec = state_dict.get(cls.STATE_INPUT_SPEC)
        backend._input_spec = None if input_spec is None else SampleMetadata.from_dict(input_spec)
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


def _clone_tree(tree: Any) -> Any:
    """Clones every tensor. An object that holds tensors is copied, because replay writes that object again."""
    return tree_map(_clone_leaf, tree)


def _clone_leaf(leaf: Any) -> Any:
    if isinstance(leaf, torch.Tensor):
        return leaf.detach().clone()
    if _state_tensors(leaf):
        return copy.deepcopy(leaf)
    return leaf


def _nested_values(item: Any) -> list[Any] | None:
    """Values stored in item, or None when item holds none."""
    if isinstance(item, dict):
        return list(item.values())
    if isinstance(item, list | tuple):
        return list(item)
    try:
        return list(vars(item).values())
    except TypeError:
        return None


def _state_tensors(value: Any) -> list[torch.Tensor]:
    """Returns tensors stored inside value. A tensor itself is not one of them."""
    nested = _nested_values(value)
    if nested is None:
        return []
    found: list[torch.Tensor] = []
    seen = {id(value)}
    pending = list(reversed(nested))
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            found.append(item)
            continue
        if id(item) in seen:
            continue
        seen.add(id(item))
        children = _nested_values(item)
        if children:
            pending.extend(reversed(children))
    return found


def _batch_axes(one: Any, two: Any) -> BatchAxes:
    """Returns the batch axes of a leaf from its values at batch size 1 and 2."""
    if not isinstance(one, torch.Tensor) or not isinstance(two, torch.Tensor):
        return {}
    if one.ndim != two.ndim:
        raise RuntimeError("XPU graph tensor rank changes with the batch size")
    axes: BatchAxes = {}
    for axis, (size_one, size_two) in enumerate(zip(one.shape, two.shape, strict=True)):
        if size_one == size_two:
            continue
        if size_two != 2 * size_one:
            raise RuntimeError("XPU graph tensor axis does not scale with the batch size")
        axes[axis] = size_one
    return axes


def _leaf_fits(leaf: Any, static: Any, axes: BatchAxes, batch_size: int) -> bool:
    """Returns True when leaf can be copied into static with batch_size rows."""
    if isinstance(static, torch.Tensor) != isinstance(leaf, torch.Tensor):
        return False
    if not isinstance(static, torch.Tensor):
        # != of two non-tensors returns a bool. A tensor pair never reaches this line.
        return leaf == static
    expected_shape = tuple(axes[axis] * batch_size if axis in axes else size for axis, size in enumerate(static.shape))
    return (leaf.dtype, leaf.device, tuple(leaf.shape)) == (static.dtype, static.device, expected_shape)


def _rows_index(ndim: int, axes: BatchAxes, start: int, rows: int) -> tuple[slice, ...]:
    """Index of rows start to start + rows along every batch axis."""
    index = [slice(None)] * ndim
    for axis, multiplier in axes.items():
        index[axis] = slice(start * multiplier, (start + rows) * multiplier)
    return tuple(index)


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
        if isinstance(ref, torch.Tensor):
            _assert_tensor_close(got, ref)
            continue
        got_tensors = _state_tensors(got)
        ref_tensors = _state_tensors(ref)
        if not got_tensors and not ref_tensors:
            if got != ref:
                raise RuntimeError("XPU graph output does not match eager")
            continue
        if len(got_tensors) != len(ref_tensors):
            raise RuntimeError("XPU graph output does not match eager")
        for got_tensor, ref_tensor in zip(got_tensors, ref_tensors, strict=True):
            _assert_tensor_close(got_tensor, ref_tensor)


def _assert_tensor_close(got: Any, ref: torch.Tensor) -> None:
    """Raises when one replay tensor is too far from its eager tensor."""
    if not isinstance(got, torch.Tensor) or got.shape != ref.shape or got.dtype != ref.dtype:
        raise RuntimeError("XPU graph output tensor does not match eager")
    rtol, atol = (1e-2, 1e-2) if ref.dtype in (torch.float16, torch.bfloat16) else (1e-3, 1e-3)
    if torch.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
        return
    if got.is_floating_point() or got.is_complex():
        difference = (got - ref).abs().max().item()
        raise RuntimeError(f"XPU graph output differs from eager (max abs {difference})")
    raise RuntimeError("XPU graph output differs from eager")
