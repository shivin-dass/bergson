import math
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from shutil import rmtree
from typing import Any, cast

import psutil
import torch
import torch.distributed as dist
import torch.distributed.tensor  # noqa: F401 — register DTensor for torch.load
import torchopt
from torch import nn
from torch.distributed._functional_collectives import (
    all_reduce as differentiable_all_reduce,
)
from torch.distributed._functional_collectives import (
    wait_tensor,
)
from torch.distributed.tensor import DTensor
from torchopt.pytree import tree_flatten_with_path, tree_iter, tree_map
from torchopt.typing import GradientTransformation, OptState
from tqdm.auto import tqdm

from ..data import sorted_checkpoints
from ..distributed import grad_tree
from .config import MagicSaveMode
from .data_stream import DataStream, Microbatch, step_inputs
from .fsdp import shallow_copy
from .rtl_tqdm import RtlTqdm
from .swap import swap_parameters


_SAVE_EXECUTOR: ThreadPoolExecutor | None = None


def _save_executor() -> ThreadPoolExecutor:
    """Single background writer thread for checkpoint saves.

    One worker is enough: callers already serialize saves (train() waits on the
    pending save before issuing the next; backward() drains its futures before
    each load), so extra workers would only add interleaving risk.
    """
    global _SAVE_EXECUTOR
    if _SAVE_EXECUTOR is None:
        _SAVE_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt_save"
        )
    return _SAVE_EXECUTOR


def _rank_and_world() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_file(path: str, rank: int) -> str:
    """This rank's shard file inside a step_N.ckpt checkpoint directory."""
    return os.path.join(path, f"rank_{rank}.pt")


def _is_local_main(rank: int) -> bool:
    """True on one rank per node — the rank that owns node-local file cleanup.

    Checkpoint shards live on node-local disks, so deletion must happen once
    per node, not once globally (global rank 0 can't see other nodes' files).
    Falls back to global rank 0 when LOCAL_RANK is unset (single-node spawn),
    where per-node and global cleanup coincide.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    return rank == 0 if local_rank is None else int(local_rank) == 0


def _consensus_valid_checkpoints(
    ckpt_list: list[tuple[int, str]], rank: int
) -> tuple[list[tuple[int, str]], list[str]]:
    """Split checkpoints into (valid, locally-present-but-invalid paths).

    A checkpoint is valid only if EVERY rank has its own shard file — a crash
    mid-save can leave some ranks' (or a whole node's) files missing, and the
    step_N.ckpt dirs themselves can exist on one node but not another. So each
    rank shares its {step: has-my-shard} map and validity is the AND over the
    union of steps seen anywhere; every rank returns the identical valid list.
    """
    local = {idx: os.path.exists(_rank_file(path, rank)) for idx, path in ckpt_list}

    if dist.is_initialized():
        world = dist.get_world_size()
        all_maps: list[dict[int, bool] | None] = [None] * world
        dist.all_gather_object(all_maps, local)
        indices = sorted({i for m in all_maps if m for i in m})
        agreed = {
            i: all(m is not None and m.get(i, False) for m in all_maps) for i in indices
        }
    else:
        agreed = local

    paths = dict(ckpt_list)
    valid = [(i, paths[i]) for i in sorted(agreed) if agreed[i] and i in paths]
    invalid_local = [paths[i] for i in sorted(agreed) if not agreed[i] and i in paths]
    return valid, invalid_local


def _add(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    """Sum two VJP results, either of which may be None.

    `torch.autograd.grad(..., allow_unused=True)` returns None for inputs the
    output didn't depend on, and the positions of those inputs are load-bearing
    (they index into params / opt-state leaves), so Nones must be carried
    through rather than dropped.
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _maybe_get_cuda_rng_state() -> torch.Tensor:
    """ "Get the CUDA RNG state if CUDA is initialized, otherwise return zeros."""
    if torch.cuda.is_initialized():
        return torch.cuda.random.get_rng_state()

    # This corresponds to a manual seed of 0
    return torch.zeros(16, dtype=torch.uint8)


@dataclass
class SaveFuture:
    """Wraps a background checkpoint-save future.

    result() re-raises any exception from the writer thread, so a failed save
    surfaces at the next synchronization point instead of silently producing a
    checkpoint that is missing this rank's shard.
    """

    fut: Future
    debug_name: str = ""
    debug_pbar: RtlTqdm | tqdm | None = None

    def result(self):
        start = time.monotonic()
        result = self.fut.result()
        elapsed = time.monotonic() - start

        if self.debug_name and (not dist.is_initialized() or dist.get_rank() == 0):
            print_fn = self.debug_pbar.write if self.debug_pbar else print
            print_fn(f"Waiting for {self.debug_name} took {elapsed:.2f} seconds")

        return result


@dataclass
class BackwardState:
    param_grads: dict[str, torch.Tensor]

    opt_grads: list[torch.Tensor]
    """PyTree of the same structure as the optimizer state, containing gradients for
    each of the optimizer state tensors."""

    weight_grads: torch.Tensor


@dataclass
class TrainerState:
    # Differentiable state
    params: dict[str, torch.Tensor]
    opt_state: OptState

    # Non-differentiable state
    buffers: dict[str, torch.Tensor]
    batch_index: int = 0
    cuda_rng_state: torch.Tensor = field(default_factory=_maybe_get_cuda_rng_state)
    cpu_rng_state: torch.Tensor = field(default_factory=torch.random.get_rng_state)

    def copy_(self, other: "TrainerState"):
        for k in self.params.keys():
            self.params[k].copy_(other.params[k])
        for k in self.buffers.keys():
            self.buffers[k].copy_(other.buffers[k])
        self.batch_index = other.batch_index
        self.cuda_rng_state.copy_(other.cuda_rng_state)
        self.cpu_rng_state.copy_(other.cpu_rng_state)

    def to(self, device: torch.device | str) -> "TrainerState":
        params = {k: p.to(device) for k, p in self.params.items()}
        buffers = {k: b.to(device) for k, b in self.buffers.items()}
        opt_state = tree_map(
            lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, self.opt_state
        )
        return TrainerState(params, opt_state, buffers, self.batch_index)

    def load(self, path: str):
        """Load state from this rank's shard file inside checkpoint dir `path`.

        Each rank reads only its own `rank_{r}.pt` — no cross-rank or
        cross-node filesystem access. Mirrors the old DCP semantics: tensors
        are copied in-place into the current state (so callers holding
        references keep them) and `batch_index` is NOT restored — callers
        (resume(), backward()) set it explicitly before loading.
        """
        rank, world = _rank_and_world()
        saved = torch.load(
            _rank_file(path, rank), map_location="cpu", weights_only=True
        )

        saved_world = int(saved.pop("__world_size"))
        if saved_world != world:
            raise RuntimeError(
                f"Checkpoint {path} was saved at world_size={saved_world} but is "
                f"being loaded at world_size={world}. Per-rank shard checkpoints "
                "are same-run scratch and cannot be resharded."
            )

        state = self.state_dict()
        with torch.no_grad():
            for k, v in saved.items():
                if k == ".batch_index":
                    continue
                target = state[k]
                if isinstance(target, DTensor):
                    target.to_local().copy_(v)
                elif isinstance(target, torch.Tensor):
                    target.copy_(v)

    def save(
        self,
        path: str,
        debug_pbar: RtlTqdm | tqdm | None = None,
        threads: int = 8,
    ) -> SaveFuture:
        """Save this rank's shards to `path/rank_{r}.pt` in a background thread.

        DTensors are saved as their local shard only; every rank writes its own
        complete file (params, opt state, buffers, RNG) to its own node's disk,
        so no rank ever needs another node's filesystem. The GPU→CPU copy is
        synchronous — with inplace training the next step mutates these tensors,
        so staging must finish before this method returns. Only the disk write
        is deferred, and it lands atomically via tmp + os.replace.
        """
        rank, world = _rank_and_world()

        def _stage(v):
            if isinstance(v, DTensor):
                return v.to_local().detach().to("cpu", copy=True)
            if isinstance(v, torch.Tensor):
                return v.detach().to("cpu", copy=True)
            return v

        cpu_state = {k: _stage(v) for k, v in self.state_dict().items()}
        cpu_state["__world_size"] = world
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        os.makedirs(path, exist_ok=True)
        rank_path = _rank_file(path, rank)

        def _write():
            tmp_path = rank_path + ".tmp"
            torch.save(cpu_state, tmp_path)
            os.replace(tmp_path, rank_path)

        fut = _save_executor().submit(_write)
        return SaveFuture(
            fut,
            debug_name=path if debug_pbar is not None else "",
            debug_pbar=debug_pbar,
        )

    def detach_(self) -> "TrainerState":
        for k, p in self.params.items():
            self.params[k] = p.detach()

        def _detach_leaf(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t.detach()
            return t

        self.opt_state = tree_map(_detach_leaf, self.opt_state)
        return self

    @property
    def requires_grad(self) -> bool:
        p_val = any(p.requires_grad for p in self.params.values())
        opt_val = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in tree_iter(self.opt_state)
        )
        return p_val or opt_val

    @requires_grad.setter
    def requires_grad(self, value: bool):
        for p in self.params.values():
            p.requires_grad = value

        for t in tree_iter(self.opt_state):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t.requires_grad = value

    def differentiable_tensors(self) -> list[torch.Tensor]:
        ps = list(self.params.values())
        os = [
            t
            for t in tree_iter(self.opt_state)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
        ]
        return ps + os

    @contextmanager
    def activate(self, model: nn.Module):
        cpu_state = torch.random.get_rng_state()
        torch.random.set_rng_state(self.cpu_rng_state)

        with swap_parameters(model, self.params, self.buffers, strict=True) as p:
            yield p

        torch.random.set_rng_state(cpu_state)

    def state_dict(self) -> dict:
        # Convert to dict manually because dataclasses.asdict does a deep copy
        state = {
            **self.params,
            **self.buffers,
            ".batch_index": torch.tensor(self.batch_index),
            ".cuda_rng_state": self.cuda_rng_state,
            ".cpu_rng_state": self.cpu_rng_state,
        }

        # Flatten opt_state PyTree into the top-level dict with "opt_state/" prefix so
        # that it can be saved with DCP, which doesn't support nested structures.
        paths, elements, _ = tree_flatten_with_path(self.opt_state)
        str_paths = ["opt_state/" + ".".join(map(str, p)) for p in paths]
        opt_state = dict(zip(str_paths, elements))
        state.update(opt_state)

        return state

    def size_in_bytes(self) -> int:
        """Roughly, the amount of space needed to save the state dict."""
        state = self.state_dict()
        return sum(
            t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
            for t in state.values()
        )


class Trainer:
    """Stateless, functional trainer for a model and optimizer."""

    @classmethod
    def initialize(
        cls,
        model: nn.Module,
        optimizer: GradientTransformation,
    ) -> tuple["Trainer", TrainerState]:
        """Convenience method for initializing the trainer and state."""
        # Create new tensor objects for the parameters and buffers to ensure that they
        # are not modified in place. Only trainable params go into the state; frozen
        # params stay in the nn.Module.
        params = shallow_copy(
            {
                k: v
                for k, v in model.named_parameters(remove_duplicate=False)
                if v.requires_grad
            }
        )
        buffers = shallow_copy(dict(model.named_buffers(remove_duplicate=False)))
        opt_state = optimizer.init(params)

        state = TrainerState(params, opt_state, buffers)
        return cls(model, optimizer), state

    def __init__(self, model: nn.Module, optimizer: GradientTransformation):
        # Move only trainable parameters to the meta device, leaving frozen params
        # on device so they don't need to be managed by TrainerState.
        for mod in model.modules():
            for p_name, param in list(mod.named_parameters(recurse=False)):
                if param.requires_grad:
                    mod.register_parameter(
                        p_name, nn.Parameter(param.data.to("meta"), requires_grad=True)
                    )

        self.model = model
        self.optimizer = optimizer

    def microbatch_grads(
        self,
        state: TrainerState,
        inputs: dict[str, Any],
        *,
        trace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Gradients of one (micro)batch's loss wrt the params in `state`.

        Does NOT touch the RNG state or all-reduce: callers own both, because
        accumulation needs the same RNG for each microbatch in Phase A and
        Phase C, and one all-reduce on the accumulated gradient.
        """
        # Trainable params live on the meta device and are swapped in from state.
        # Frozen params remain on-device in the model and are left untouched.
        with swap_parameters(
            self.model,
            state.params,
            state.buffers,
            preserve_graph=trace,
        ) as params:
            outputs = self.model(**inputs)

            # Currently we support two output types: HuggingFace, and "raw loss"
            # - HuggingFace models output a dict/dataclass with a "loss" field
            # - Raw loss models output a single scalar loss value as a Tensor
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = outputs

            assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
            self._last_loss = loss.detach().item()
            return grad_tree(loss, params, create_graph=trace)

    def accumulate_grads(
        self,
        state: TrainerState,
        micros: list[Microbatch],
        *,
        fsdp: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor | None], list[tuple[torch.Tensor, torch.Tensor]]
    ]:
        """Phase A: accumulate `g_total = sum_k w_k * g_k` without tracing.

        Each microbatch's graph is freed as soon as its gradient is taken
        (`create_graph=False`), so peak activation memory is one microbatch's
        rather than the whole effective batch's.

        Returns `(g_total, rng_states)`, where `rng_states[k]` is the (CPU,
        CUDA) RNG state snapshot taken immediately before microbatch k's
        forward. Phase C restores them so its traced recomputation of `g_k`
        sees byte-identical randomness.

        `g_total` has exactly the same keys as `state.params`, including `None`
        at positions the loss doesn't depend on (`allow_unused=True` yields
        those). Dropping them would make the pytree narrower than
        `params`/`opt_state`, and torchopt zips those trees positionally: the
        optimizer would silently pair each gradient with the wrong parameter's
        moments. torchopt treats `None` as a leaf and skips it, so keeping the
        holes is both structurally correct and a no-op numerically.
        """
        g_total: dict[str, torch.Tensor | None] = {}
        rng_states = []

        for micro in micros:
            # Snapshot before the forward, and re-seed from the step's state so
            # microbatch k's randomness is a pure function of (step, k) — never
            # of how many microbatches ran before it.
            torch.random.set_rng_state(state.cpu_rng_state)
            rng_states.append(
                (torch.random.get_rng_state(), _maybe_get_cuda_rng_state())
            )

            grads = self.microbatch_grads(state, micro.inputs, trace=False)
            for k, g in grads.items():
                prev = g_total.get(k)
                if g is None:
                    g_total.setdefault(k, None)
                elif prev is None:
                    g_total[k] = g * micro.weight
                else:
                    prev.add_(g, alpha=micro.weight)
            del grads

        # One all-reduce on the accumulated gradient: equivalent to averaging
        # each microbatch's gradient separately (the sum is linear) and K times
        # cheaper in collectives.
        if dist.is_initialized() and not fsdp:
            for g in g_total.values():
                if g is not None:
                    dist.all_reduce(g, op=dist.ReduceOp.AVG)

        return g_total, rng_states

    def apply_grads(
        self,
        state: TrainerState,
        grads: dict[str, torch.Tensor | None],
        *,
        inplace: bool = False,
    ) -> TrainerState:
        """Run the optimizer update, returning the next state.

        Split out of `step` so the backward pass's Phase B can re-run exactly
        this computation differentiably on detached leaves.
        """
        updates, new_state = self.optimizer.update(
            grads, state.opt_state, inplace=inplace, params=state.params
        )
        new_params = torchopt.apply_updates(state.params, updates, inplace=inplace)
        return TrainerState(
            new_params,
            new_state,
            state.buffers,
            state.batch_index + 1,
        )

    def step(
        self,
        state: TrainerState,
        inputs: dict[str, Any] | list[Microbatch],
        *,
        inplace: bool = False,
        trace: bool = False,
        fsdp: bool = False,
    ) -> TrainerState:
        """Perform a single training step on `state`, returning the new state.

        Args:
            state: The current trainer state, containing model parameters, optimizer
                state, and RNG states.
            inputs: A batch of training data to use for this step. It will be unpacked
                as keyword arguments to the model's forward method. A list of
                `Microbatch`es instead runs the step with gradient accumulation:
                gradients are accumulated across the microbatches and a single
                optimizer update is applied, so peak activation memory is one
                microbatch's while the update is the full effective batch's.
            inplace: Whether to perform in-place updates during this step. In-place
                updates can reduce memory usage but can cause problems with autograd.
            trace: Whether to trace this step with autograd to allow for a backward
                pass later. Tracing can add overhead, so it should only be enabled if
                backward passes will be needed. Not supported together with
                accumulation — the backward pass replays accumulated steps with the
                three-phase VJP in `backward`, which is the whole point of
                accumulation (tracing all microbatches at once would hold every
                microbatch's double-backward graph alive simultaneously, i.e. the
                memory accumulation exists to avoid).
            fsdp: Whether the model is wrapped with FSDP. If False and distributed
                training is being used, the trainer will perform its own all-reduce of
                gradients. If True, the trainer will assume that FSDP is handling
                gradient synchronization, and will not perform any all-reduces itself.
        """
        if isinstance(inputs, list):
            assert not trace, (
                "trace=True with gradient accumulation would defeat its purpose; "
                "Trainer.backward replays accumulated steps with a split VJP"
            )
            torch.random.set_rng_state(state.cpu_rng_state)
            grads, _ = self.accumulate_grads(state, inputs, fsdp=fsdp)
            return self.apply_grads(state, grads, inplace=inplace)

        torch.random.set_rng_state(state.cpu_rng_state)
        grads = self.microbatch_grads(state, inputs, trace=trace)

        if dist.is_initialized() and not fsdp:
            if trace:
                # Use differentiable all_reduce to preserve autograd graph
                grads = {
                    k: wait_tensor(
                        differentiable_all_reduce(
                            g / dist.get_world_size(),
                            "sum",
                            dist.distributed_c10d._get_default_group(),
                        )
                    )
                    for k, g in grads.items()
                }
            else:
                for g in grads.values():
                    dist.all_reduce(g, op=dist.ReduceOp.AVG)

        return self.apply_grads(state, grads, inplace=inplace)

    def resume(self, state: TrainerState, save_dir: str) -> TrainerState:
        """Resume training from the most recent checkpoint in `save_dir`.

        This method modifies `state` in-place to load the most recent checkpoint, and
        returns it for convenience.
        """
        ckpt_list = sorted_checkpoints(save_dir)
        rank, _ = _rank_and_world()

        # Filter out incomplete checkpoints (some rank's shard missing after a
        # mid-save crash) and clean them up. Validity is agreed across ranks so
        # everyone resumes from the same step; deletion is per-node since each
        # node only sees its own shard files.
        valid_ckpts, invalid_paths = _consensus_valid_checkpoints(ckpt_list, rank)
        if _is_local_main(rank):
            for path in invalid_paths:
                if os.path.exists(path):
                    rmtree(path) if os.path.isdir(path) else os.remove(path)

        # Load the most recent trainer state
        if valid_ckpts:
            last_idx, last_path = valid_ckpts[-1]
            state.batch_index = last_idx
            state.load(last_path)
            state.detach_()

        return state

    def train(
        self,
        state: TrainerState,
        data: DataStream,
        *,
        debug: bool = False,
        inplace: bool = False,
        save_dir: str | None = None,
        save_mode: MagicSaveMode = "sqrt",
        trace: bool = False,
        log_fn: Callable[[int, float], None] | None = None,
        resume: bool = False,
        fsdp: bool = False,
        grad_accum_steps: int = 1,
    ) -> TrainerState:
        """Train the model on the given data stream, starting from the given state.

        Args:
            state: The initial trainer state to start training from.
            data: The training data stream to iterate over.
            debug: Whether to print debug information about checkpoint loading times.
            inplace: Whether to perform in-place updates during training. In-place
                updates can reduce memory usage but may cause issues with some
                optimizers or models.
            save_dir: The directory in which to save checkpoints during training.
            save_mode: The strategy for how often to save checkpoints.
            trace: Whether to trace the training step with autograd to allow for
                backward passes. Tracing can add overhead, so it should only be enabled
                if backward passes will be needed.
            log_fn: An optional function to call after each training step with the step
                index and the most recent loss value, for logging purposes.
            resume: Whether to resume from a previously interrupted training run by
                loading the most recent checkpoint from `save_dir`.
            fsdp: Flag to pass to `Trainer.step`, indicating whether the model is
                wrapped with FSDP.
            grad_accum_steps: Number of microbatches to split each step's batch into
                (1 = no accumulation). The optimizer step, and therefore the saved
                checkpoint trajectory, is over the whole batch either way.

        Returns:
            The final trainer state after training.
        """
        # Make sure the save directory exists
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # Always save the first state
        next_save = 0
        n = len(data)

        start = 0
        if resume and save_dir is not None:
            state = self.resume(state, save_dir)
            start = state.batch_index

        pending_save: SaveFuture | None = None

        main = not dist.is_initialized() or dist.get_rank() == 0
        pbar = tqdm(range(start, n), desc="Training", disable=not main)

        for i in pbar:
            # Save checkpoint BEFORE each step. Step 0 is the initial state prior to
            # any updates, step 1 is the state after the first update, etc.
            if save_dir and i == next_save:
                # Wait for the previous save before starting a new one to avoid
                # multiple concurrent DCP saves with separate Gloo groups, which can
                # deadlock when background threads call distributed operations.
                if pending_save is not None:
                    pending_save.result()

                p = os.path.join(save_dir, f"step_{i}.ckpt")
                pending_save = state.save(p, debug_pbar=pbar if debug else None)

                match save_mode:
                    case "all":
                        # Save next step
                        next_save += 1
                    case "sqrt":
                        chunk_size = math.isqrt(n)

                        # If we're in the last chunk, save every step
                        if i >= n - chunk_size:
                            next_save += 1

                        # Otherwise, save sqrt(n) steps from now
                        else:
                            next_save += chunk_size
                    case "log":
                        # Cut the remaining steps in half to get to the next save point
                        next_save = max(1, (n - next_save) // 2) + next_save
                    case other:
                        raise ValueError(f"Unsupported save mode: {other}")

            x = step_inputs(data, i, grad_accum_steps)
            state = self.step(state, x, inplace=inplace, trace=trace, fsdp=fsdp)

            if log_fn is not None:
                log_fn(i, self._last_loss)

        if pending_save is not None:
            pending_save.result()

        return state

    def save_backward_state(self, bwd_state, path, expected_idx, last_idx):
        tmp_path = path + ".tmp"
        torch.save(
            {
                "expected_idx": expected_idx,
                "last_idx": last_idx,
                "param_grads": bwd_state.param_grads,
                "opt_grads": bwd_state.opt_grads,
                "weight_grads": bwd_state.weight_grads,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load_backward_state(self, path, ckpt_list, device, main: bool):
        saved = torch.load(path, map_location=device, weights_only=True)
        bwd_state = BackwardState(
            saved["param_grads"],
            saved["opt_grads"],
            saved["weight_grads"],
        )
        expected_idx = saved["expected_idx"]
        last_idx = saved["last_idx"]

        # Filter to valid checkpoints we still need to process. Validity needs
        # cross-rank consensus (a crash can leave one node's shards missing);
        # deletion is per-node because shards live on node-local disks.
        rank, _ = _rank_and_world()
        needed = [(idx, p) for idx, p in ckpt_list if idx <= expected_idx]
        ckpt_list, invalid_paths = _consensus_valid_checkpoints(needed, rank)
        if _is_local_main(rank):
            for p in invalid_paths:
                if os.path.isdir(p):
                    rmtree(p)

        if not ckpt_list and expected_idx >= 0:
            raise RuntimeError(
                f"Cannot resume backward: no valid checkpoints found "
                f"for step {expected_idx}"
            )

        if main:
            print(f"Resuming backward pass from step {expected_idx}")

        return bwd_state, ckpt_list, expected_idx, last_idx

    def _accumulated_vjp(
        self,
        fwd_state: TrainerState,
        micros: list[Microbatch],
        bwd_state: BackwardState,
        weights: torch.Tensor,
        flat_i: list[torch.Tensor],
        *,
        fsdp: bool = False,
    ) -> BackwardState:
        """VJP through one accumulated step, split at the accumulated gradient.

        A single VJP through the whole accumulated step would need every
        microbatch's double-backward graph alive at once — exactly the memory
        accumulation exists to avoid. Instead we cut the step at `g_total`,
        which is legal because accumulation is *linear* in the microbatch
        gradients even though the optimizer isn't:

            g_total = sum_k w_k * g_k(params_in)          [linear seam]
            params_out, opt_out = Update(g_total, params_in, opt_in)

        so, writing `v = dL/dg_total`,

            dL/dparams_in = [direct from Update] + sum_k w_k (dg_k/dparams_in)^T v
            dL/dopt_in    = [direct from Update]
            dL/dweights   = sum_k w_k (dg_k/dweights)^T v

        Phase A recomputes `g_total` untraced (one microbatch of activations at
        a time). Phase B re-runs ONLY the optimizer update differentiably on
        detached leaves — elementwise ops on param-sized tensors, so the graph
        is negligible — and yields both `v` and the direct terms. Any
        nonlinearity confined to the update (Adam's second moment, and
        `clip_grad_norm`'s coefficient, which torchopt computes as a Python
        float and is therefore straight-through) is handled exactly there.
        Phase C then recomputes each `g_k` traced, one at a time, and pushes
        `w_k * v` back through it, freeing the graph before the next.

        Peak batch-dependent memory is thus ONE microbatch's double-backward
        graph, independent of the effective batch size, and the result is
        mathematically identical to the unaccumulated whole-batch VJP.
        """
        # ---- Phase A: accumulate g_total untraced, snapshotting RNG per micro.
        torch.random.set_rng_state(fwd_state.cpu_rng_state)
        g_total, rng_states = self.accumulate_grads(fwd_state, micros, fsdp=fsdp)

        p_keys = list(bwd_state.param_grads.keys())
        p_grads = list(bwd_state.param_grads.values())
        o_grads = bwd_state.opt_grads
        w_grads = bwd_state.weight_grads
        del bwd_state

        # ---- Phase B: re-run the optimizer update on detached leaves.
        # `flat_i` (params, then float opt-state leaves — the
        # `differentiable_tensors()` order) already requires grad; g_total is
        # made a leaf so the VJP can report dL/dg_total separately.
        # `None` entries are kept so the tree still lines up with the optimizer
        # state (see `accumulate_grads`), but only the real tensors can be
        # differentiated with respect to.
        g_leaves = {
            k: (g.detach().requires_grad_() if g is not None else None)
            for k, g in g_total.items()
        }
        del g_total
        g_keys = [k for k, g in g_leaves.items() if g is not None]

        state_f = self.apply_grads(fwd_state, g_leaves, inplace=False)

        # inplace=False above, so this is a pure function of the leaves.
        opt_vjp = list(
            torch.autograd.grad(
                state_f.differentiable_tensors(),
                [g_leaves[k] for k in g_keys] + flat_i,
                grad_outputs=p_grads + o_grads,
                allow_unused=True,
            )
        )
        del p_grads, state_f

        # dL/dg_total, keyed like the gradient dict, and the update rule's
        # DIRECT contributions to dL/d(params_in, opt_in): the identity path,
        # weight decay, and the moment updates. Phase C adds the paths that go
        # through each microbatch's loss.
        v = dict(zip(g_keys, opt_vjp[: len(g_keys)]))
        state_grads = opt_vjp[len(g_keys) :]
        del opt_vjp, g_leaves

        # ---- Phase C: per microbatch, recompute g_k traced and push w_k * v
        # back through it, freeing each graph before the next.
        weight_grads = w_grads
        for micro, (cpu_rng, cuda_rng) in zip(micros, rng_states, strict=True):
            # Byte-identical randomness to this microbatch's Phase A forward.
            torch.random.set_rng_state(cpu_rng)
            if torch.cuda.is_initialized():
                torch.cuda.random.set_rng_state(cuda_rng)

            grads = self.microbatch_grads(fwd_state, micro.inputs, trace=True)

            if dist.is_initialized() and not fsdp:
                # Phase A all-reduced g_total once; the transpose of that
                # (linear) average applies per microbatch here.
                grads = {
                    k: wait_tensor(
                        differentiable_all_reduce(
                            g / dist.get_world_size(),
                            "sum",
                            dist.distributed_c10d._get_default_group(),
                        )
                    )
                    if g is not None
                    else None
                    for k, g in grads.items()
                }

            # Only the params g_k actually depends on differentiably; g_k is
            # None for params this microbatch's loss doesn't touch, and v is
            # None for params the optimizer update left untouched.
            live = [
                k
                for k, g in grads.items()
                if g is not None and v.get(k) is not None
            ]
            outs = [grads[k] for k in live]
            v_outs = [v[k] * micro.weight for k in live]

            micro_vjp = torch.autograd.grad(
                outs,
                flat_i + [weights],
                grad_outputs=v_outs,
                allow_unused=True,
            )
            del grads, outs, v_outs

            state_grads = [
                _add(a, b)
                for a, b in zip(state_grads, micro_vjp[:-1], strict=True)
            ]
            weight_grads = _add(weight_grads, micro_vjp[-1])
            del micro_vjp

        param_grads = dict(zip(p_keys, state_grads[: len(p_keys)]))
        return BackwardState(param_grads, state_grads[len(p_keys) :], weight_grads)

    def backward(
        self,
        ckpt_dir: str,
        data: DataStream,
        bwd_state: BackwardState,
        fwd_state: TrainerState,
        *,
        cleanup: bool = True,
        debug: bool = False,
        inplace: bool = False,
        fsdp: bool = False,
        resume: bool = False,
        save_every: int = 0,
        save_mode: MagicSaveMode = "sqrt",
        grad_accum_steps: int = 1,
    ) -> BackwardState:
        """Run a backward pass through the training trajectory saved at `ckpt_dir`.

        Args:
            ckpt_dir: Directory containing checkpoints saved during the forward pass.
            data: The training data stream, needed to replay forward steps.
            bwd_state: The initial backward state, containing gradients for the query
                loss function.
            fwd_state: Forward state containing the model parameters and optimizer
                state from the end of the forward trajectory.
            cleanup: Whether to delete checkpoints in `ckpt_dir` as soon as they are
                no longer needed. If False, only the temporary checkpoints created
                during the backward pass will be deleted, and all original checkpoints
                will be preserved.
            debug: Whether to print debug information about checkpoint loading times.
            inplace: Whether to perform in-place updates during forward and backward
                steps. In-place updates can reduce memory usage but may cause issues
                with some optimizers or models.
            fsdp: The `fsdp` flag that will be passed to the trainer's `step` method.
            resume: Whether to resume from a previously interrupted backward pass by
                loading the backward state and skipping checkpoints that have already
                been processed.
            save_every: If > 0, save the backward state every N steps to allow resuming
                from interruptions. Backward checkpoints are saved in `ckpt_dir` with
                the name `backward_rank{rank}.pt`.
            save_mode: The save mode that was used during the forward trajectory, which
                determines how checkpoints are spaced and thus how the backward pass
                should step forward through the trajectory when replaying.
            grad_accum_steps: Number of microbatches per step. MUST match the value
                used for the forward pass, otherwise the replayed states won't match
                the trajectory the checkpoints came from. When > 1, each step's VJP is
                split at the accumulated gradient (see `_accumulated_vjp`).

        Returns:
            The final backward state after processing the entire trajectory.
        """
        ckpts = sorted_checkpoints(ckpt_dir)
        ckpt_paths = [path for _, path in ckpts]
        preserve_paths = {ckpt_paths[-1]} if cleanup else set(ckpt_paths)

        ckpt_list = cast(list[tuple[int, str | TrainerState]], ckpts)
        state_size = fwd_state.size_in_bytes()

        main = not dist.is_initialized() or dist.get_rank() == 0
        rank = dist.get_rank() if dist.is_initialized() else 0

        bwd_ckpt_path = os.path.join(ckpt_dir, f"backward_rank{rank}.pt")

        if resume and os.path.exists(bwd_ckpt_path):
            bwd_state, ckpt_list, expected_idx, last_idx = self.load_backward_state(
                bwd_ckpt_path, ckpt_list, data.device, main
            )
        else:
            expected_idx, _ = ckpt_list[-1]
            last_idx = expected_idx

        main_pbar = RtlTqdm(
            desc="Backward",
            total=last_idx + 1,
            initial=last_idx - expected_idx,
            disable=not main,
            position=0,
            # Get rid of jitters in the ETA due to rematerialization
            smoothing=0,
        )
        sub_pbar = None

        save_futures: list[SaveFuture] = []
        while ckpt_list:
            # Make sure everything has been saved
            for fut in save_futures:
                fut.result()
            save_futures.clear()

            idx, ckpt = ckpt_list[-1]
            fwd_state.batch_index = idx
            fwd_state.detach_()  # Detach so that replay steps can use in-place ops

            start = time.monotonic()
            fwd_state.load(ckpt) if isinstance(ckpt, str) else fwd_state.copy_(ckpt)
            elapsed = time.monotonic() - start

            if debug and main:
                name = ckpt if isinstance(ckpt, str) else f"in-memory checkpoint {idx}"
                main_pbar.write(f"Loaded checkpoint {name} in {elapsed:.2f} seconds")

            # Only delete this checkpoint if it's the one we expected to load. If it's
            # not, we need to keep it around, and step forward through training
            if idx == expected_idx:
                del ckpt_list[-1]

                if isinstance(ckpt, str) and ckpt not in preserve_paths:
                    # Unlike the old collective dcp.load, per-rank loads don't
                    # synchronize — without a barrier the deleting rank can
                    # race ahead and remove shards a sibling rank hasn't read.
                    if dist.is_initialized():
                        dist.barrier()

                    # Delete once per node: shard files are node-local, so
                    # global rank 0 cannot clean up the other nodes' copies.
                    if _is_local_main(rank) and os.path.exists(ckpt):
                        rmtree(ckpt) if os.path.isdir(ckpt) else os.remove(ckpt)

            # Step forward in training if needed
            next_save = (
                (expected_idx - idx) // 2 + idx
                if save_mode == "log"
                else min(expected_idx, idx + 1)
            )
            while idx < expected_idx:
                if sub_pbar is None:
                    sub_pbar = tqdm(
                        total=expected_idx - idx,
                        desc=f"Rematerializing steps {idx} to {expected_idx}",
                        disable=not main,
                        leave=False,
                        position=1,
                        smoothing=0,
                    )

                fwd_state = self.step(
                    fwd_state,
                    step_inputs(data, fwd_state.batch_index, grad_accum_steps),
                    inplace=inplace,
                    trace=False,
                    fsdp=fsdp,
                )
                idx += 1
                sub_pbar.update()

                # Save checkpoints for states we will need later
                if idx == next_save and idx < expected_idx:
                    # Switch from RAM disk to checkpoint dir if needed
                    num_copies = dist.get_world_size() if dist.is_initialized() else 1
                    fits_in_ram = (
                        psutil.virtual_memory().available > num_copies * state_size
                    )
                    if dist.is_initialized():
                        flag = torch.tensor(
                            int(fits_in_ram),
                            dtype=torch.int32,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                        fits_in_ram = bool(flag.item())
                    if fits_in_ram:
                        ckpt_list.append((idx, fwd_state.to("cpu").detach_()))
                    else:
                        ckpt = os.path.join(ckpt_dir, f"step_{idx}.ckpt")
                        ckpt_list.append((idx, ckpt))

                        save_futures.append(
                            fwd_state.save(
                                ckpt, debug_pbar=main_pbar if debug else None
                            )
                        )

                    # Advance next_save according to the save mode
                    next_save = (
                        (expected_idx - idx) // 2 + idx
                        if save_mode == "log"
                        else min(expected_idx, idx + 1)
                    )

            if sub_pbar is not None:
                sub_pbar.close()
                sub_pbar = None

            # The index we expect on the next iteration is one less than the current
            expected_idx = idx - 1

            fwd_state.detach_()
            fwd_state.requires_grad = True
            data.requires_grad = True

            flat_i = fwd_state.differentiable_tensors()

            if grad_accum_steps > 1:
                bwd_state = self._accumulated_vjp(
                    fwd_state,
                    data.microbatches(fwd_state.batch_index, grad_accum_steps),
                    bwd_state,
                    data.weights,
                    flat_i,
                    fsdp=fsdp,
                )
                main_pbar.update()
            else:
                # Re-do the training step
                state_f = self.step(
                    fwd_state,
                    data[fwd_state.batch_index],
                    trace=True,
                    fsdp=fsdp,
                )
                main_pbar.update()

                # Carefully consume the bwd state to save memory
                flat_f = state_f.differentiable_tensors()
                p_grads = list(bwd_state.param_grads.values())
                o_grads = bwd_state.opt_grads

                p_keys = list(bwd_state.param_grads.keys())
                w_grads = bwd_state.weight_grads
                del bwd_state

                # grad_outputs is the gradient of the loss wrt the next TrainerState.
                # We're doing a VJP to get the gradient wrt the current TrainerState,
                # AND the example weights for this batch.
                inps = flat_i + [data.weights]
                result = list(
                    torch.autograd.grad(
                        flat_f,
                        inps,
                        grad_outputs=p_grads + o_grads,
                        allow_unused=True,
                    )
                )
                del p_grads

                # Accumulate parameter gradients
                param_grads = {k: result[i] for i, k in enumerate(p_keys)}
                del result[: len(p_keys)]

                weight_grads = result[-1] + w_grads
                bwd_state = BackwardState(param_grads, result[:-1], weight_grads)

            # Save backward state for resume
            steps_done = last_idx - expected_idx
            if save_every > 0 and steps_done % save_every == 0:
                self.save_backward_state(
                    bwd_state, bwd_ckpt_path, expected_idx, last_idx
                )

        for fut in save_futures:
            fut.result()

        # Clean up backward state file on successful completion
        if os.path.exists(bwd_ckpt_path):
            os.remove(bwd_ckpt_path)

        main_pbar.close()
        return bwd_state
