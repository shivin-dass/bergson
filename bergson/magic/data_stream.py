from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset

from ..data import pad_and_tensor


@dataclass
class Microbatch:
    """One microbatch of a training step, for gradient accumulation.

    `weight` is this microbatch's share of the parent step's loss denominator,
    defined so that the step's loss decomposes exactly as

        loss_step == sum_k weight_k * loss_k

    which is what makes `g_total = sum_k weight_k * g_k` equal to the gradient
    of the unaccumulated full-batch step. Every loss in this codebase is a
    weighted sum over units divided by a *batch-dependent* denominator (row
    count, or valid-token count), so the weights are NOT 1/K in general: a
    microbatch's gradient is already an average over its own units, and
    recombining them needs each one's share of the denominator. Only when all
    microbatches carry the same denominator does this reduce to 1/K.

    Streams own this computation because the denominator convention belongs to
    the (stream, model-wrapper) pair, not to the Trainer.
    """

    inputs: dict
    weight: float


def step_inputs(data, i: int, grad_accum_steps: int) -> dict | list[Microbatch]:
    """The inputs for step `i`: one batch, or a list of microbatches."""
    if grad_accum_steps > 1:
        return data.microbatches(i, grad_accum_steps)

    return data[i]


class DataStream:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        input_key: str = "text",
        weight_shape: tuple[int, ...] | None = None,
    ):
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = torch.device(device)
        self.input_key = input_key
        self.n = len(dataset)
        self.num_batches = self.n // batch_size

        # If a shape isn't provided, assume that each sequence contains one document
        if weight_shape is None:
            weight_shape = (self.n,)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.weights = torch.nn.Parameter(torch.ones(*weight_shape, device=device))

    @property
    def requires_grad(self) -> bool:
        return self.weights.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.weights.requires_grad = value

    def _prepare_batch(self, batch):
        """Convert batch to tensors. Override this method for non-token data.

        Args:
            batch: Dict from dataset[indices], containing data columns

        Returns:
            Tuple of (x, y, valid_mask) where:
                x: Input tensor (e.g., observations, input_ids)
                y: Label tensor (e.g., actions, labels)
                valid_mask: Boolean tensor indicating valid positions
        """
        x, y, valid_mask = pad_and_tensor(
            batch["input_ids"],
            labels=batch.get("labels"),
            device=self.device,
        )
        return x, y, valid_mask

    def _assemble(self, i: int) -> tuple[dict, Any]:
        """Assemble step `i`'s batch, minus the weight gather.

        Returns the batch and the index into `self.weights` that produces its
        `example_weight`. The gather is left to the caller so that microbatches
        can each take their own — see `microbatches`.
        """
        if i < 0 or i >= len(self):
            raise IndexError("DataStream index out of range")

        rng = range(
            i * self.batch_size,
            min((i + 1) * self.batch_size, len(self.dataset)),
        )
        indices = list(rng)[self.rank :: self.world_size]

        batch = self.dataset[indices]
        x, y, valid_mask = self._prepare_batch(batch)

        # If the weights are 1D, we assume they correspond to documents and look for
        # "doc_ids" in the batch to index them. If they're 2D, they correspond to tokens
        if self.weights.ndim == 2:
            # Truncate to the max sequence length in the batch to avoid indexing errors
            indices = (indices, slice(None, x.shape[1]))
        elif "doc_ids" in batch:
            indices = torch.tensor(batch["doc_ids"], device=self.device)
            # doc_ids may be longer than the per-batch padded seq_len (unpacked
            # path stores doc_ids at dataset-wide max_len); truncate to match.
            if indices.ndim == 2:
                indices = indices[:, : x.shape[1]]

        return {
            "input_ids": x,
            "labels": y,
            "valid_mask": valid_mask,
        }, indices

    def __getitem__(self, i: int) -> dict:
        batch, indices = self._assemble(i)
        batch["example_weight"] = self.weights[indices]
        return batch

    def microbatches(self, i: int, k: int) -> list[Microbatch]:
        """Split step `i` into `k` row-slices of the assembled batch.

        Slicing the assembled batch (rather than re-assembling each slice)
        keeps every row's padded width — and therefore its per-token losses —
        identical to the unaccumulated step.

        Each microbatch gathers its OWN `example_weight` from the weights
        Parameter. Slicing a single shared gather would instead leave all K
        microbatches hanging off one autograd node, whose saved tensors are
        freed by the first microbatch's backward — the accumulated backward
        pass differentiates each microbatch separately, so they must not share
        graph nodes.

        `weighted_causal_lm_ce` divides by `valid_mask[:, :-1].sum()` (or by
        T-1 when no mask is given), so a slice's weight is its share of the
        batch's valid positions.
        """
        batch, indices = self._assemble(i)
        rows = batch["input_ids"].shape[0]
        if rows % k:
            raise ValueError(
                f"batch of {rows} rows is not divisible into {k} microbatches"
            )

        valid = batch["valid_mask"]
        denom = valid[:, :-1].sum()

        size = rows // k
        micros = []
        for start in range(0, rows, size):
            sl = slice(start, start + size)
            inputs = {
                key: (v[sl] if isinstance(v, torch.Tensor) else v)
                for key, v in batch.items()
            }
            # Per-microbatch gather: row-sliced for row-indexed weights, and
            # for 2-D (per-token) weights the row list is sliced in place.
            if isinstance(indices, tuple):
                rows_idx, col_slice = indices
                inputs["example_weight"] = self.weights[rows_idx[sl], col_slice]
            else:
                inputs["example_weight"] = self.weights[indices[sl]]

            weight = (valid[sl, :-1].sum() / denom).item()
            micros.append(Microbatch(inputs, weight))

        return micros

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.num_batches

    def __reversed__(self):
        for i in reversed(range(len(self))):
            yield self[i]
