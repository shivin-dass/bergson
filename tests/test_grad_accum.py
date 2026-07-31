"""Gradient accumulation must not change MAGIC scores.

`grad_accum_steps=K` splits each step's batch into K microbatches to cut peak
activation memory, executing the same optimizer step. The forward trajectory
and the influence scores must therefore match an unaccumulated run at the same
effective batch size (up to float non-associativity), which is what these
tests pin down. See HANDOFF_W8.10_grad_accum.md.
"""

import tempfile

import pytest
import torch
import torchopt
from datasets import Dataset
from torch import nn
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.data_stream import Microbatch
from bergson.utils.math import weighted_causal_lm_ce

MODEL_NAME = "trl-internal-testing/tiny-Phi3ForCausalLM"


@pytest.fixture
def accum_dataset():
    """8 rows of equal length, so 8 = 2x4 = 4x2 splits evenly."""
    torch.manual_seed(0)
    rows = torch.randint(1, 1000, (8, 6)).tolist()
    return Dataset.from_dict(
        {
            "input_ids": rows,
            "labels": rows,
            "attention_mask": [[1] * 6] * 8,
        }
    )


def _run_magic(dataset, batch_size: int, grad_accum_steps: int, clip_norm: float = 0.0):
    """Full MAGIC run (train + query grads + backward) -> (scores, final params)."""
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    # Match the production optimizer (vla_foundry cached_worker): adamw,
    # optionally behind clip_grad_norm, whose coefficient is nonlinear in the
    # accumulated gradient and so exercises the Phase B seam.
    optimizer = torchopt.adamw(1e-3, betas=(0.95, 0.975), eps_root=1e-2)
    if clip_norm > 0:
        optimizer = torchopt.chain(torchopt.clip_grad_norm(clip_norm), optimizer)

    trainer, fwd_state = Trainer.initialize(model, optimizer)
    stream = DataStream(dataset, batch_size=batch_size, device="cpu")

    with tempfile.TemporaryDirectory() as ckpt_dir:
        fwd_state = trainer.train(
            fwd_state,
            stream,
            inplace=True,
            save_dir=ckpt_dir,
            save_mode="all",
            grad_accum_steps=grad_accum_steps,
        )
        final_params = {k: p.detach().clone() for k, p in fwd_state.params.items()}

        with fwd_state.activate(model) as params:
            batch = stream[0]
            del batch["example_weight"]
            loss = model(**batch).loss
            query_grads = {
                k: g.detach().clone() for k, g in grad_tree(loss, params).items()
            }
            opt_grads = [
                torch.zeros_like(buf)
                for buf in tree_iter(fwd_state.opt_state)
                if isinstance(buf, torch.Tensor) and buf.is_floating_point()
            ]
            bwd_state = BackwardState(
                query_grads, opt_grads, torch.zeros_like(stream.weights)
            )

        stream.requires_grad = True
        bwd_state = trainer.backward(
            ckpt_dir,
            stream,
            bwd_state,
            fwd_state,
            inplace=True,
            cleanup=True,
            save_mode="all",
            grad_accum_steps=grad_accum_steps,
        )

    return bwd_state.weight_grads.detach().cpu(), final_params


@pytest.mark.parametrize("clip_norm", [0.0, 1e-3])
@pytest.mark.parametrize("grad_accum_steps", [2, 4])
def test_accumulated_scores_match_unaccumulated(
    accum_dataset, grad_accum_steps, clip_norm
):
    """K microbatches at effective batch B == one batch of B, scores and all.

    The clip_norm=1e-3 case is chosen small enough that clipping actually binds
    (an unclipped run would leave the coefficient at 1.0 and the nonlinear part
    of Phase B untested).
    """
    batch_size = 4

    ref_scores, ref_params = _run_magic(
        accum_dataset, batch_size, grad_accum_steps=1, clip_norm=clip_norm
    )
    acc_scores, acc_params = _run_magic(
        accum_dataset,
        batch_size,
        grad_accum_steps=grad_accum_steps,
        clip_norm=clip_norm,
    )

    # The forward trajectory must match first: a mismatch here means the
    # accumulated step itself is wrong, not the VJP.
    for k in ref_params:
        torch.testing.assert_close(
            acc_params[k],
            ref_params[k],
            atol=1e-6,
            rtol=1e-5,
            msg=lambda m, k=k: f"param {k}: {m}",
        )

    assert ref_scores.abs().sum() > 0, "reference scores are all zero (degenerate)"
    torch.testing.assert_close(acc_scores, ref_scores, atol=1e-7, rtol=1e-4)


def test_microbatch_weights_sum_to_one(accum_dataset):
    """Microbatch weights partition the step's loss denominator.

    `g_total = sum_k w_k g_k` reproduces the full-batch gradient only if the
    w_k are each microbatch's share of the batch's denominator (here, valid
    token positions) — with equal-length rows that is exactly 1/K.
    """
    stream = DataStream(accum_dataset, batch_size=4, device="cpu")
    micros = stream.microbatches(0, 2)

    assert len(micros) == 2
    assert sum(m.weight for m in micros) == pytest.approx(1.0)
    for m in micros:
        assert m.weight == pytest.approx(0.5)
        assert m.inputs["input_ids"].shape[0] == 2
        # example_weight rows must follow the sliced rows, still attached to
        # the weights Parameter (that path is what produces the scores).
        assert m.inputs["example_weight"].shape[0] == 2
        assert m.inputs["example_weight"].grad_fn is not None

    # Each microbatch must own its gather. Sharing one node across microbatches
    # makes the second microbatch's backward hit buffers the first one freed.
    assert micros[0].inputs["example_weight"].grad_fn is not (
        micros[1].inputs["example_weight"].grad_fn
    )


def test_microbatch_grads_match_full_batch(accum_dataset):
    """Phase A's accumulated gradient equals the full batch's gradient."""
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)

    trainer, state = Trainer.initialize(model, torchopt.adamw(1e-3))
    stream = DataStream(accum_dataset, batch_size=8, device="cpu")

    torch.random.set_rng_state(state.cpu_rng_state)
    full = trainer.microbatch_grads(state, stream[0])
    accumulated, _ = trainer.accumulate_grads(state, stream.microbatches(0, 4))

    assert set(accumulated) == {k for k, g in full.items() if g is not None}
    for k, g in accumulated.items():
        torch.testing.assert_close(
            g, full[k], atol=1e-7, rtol=1e-5, msg=lambda m, k=k: f"grad {k}: {m}"
        )


class _PartlyUnusedModel(nn.Module):
    """Trainable params the loss doesn't touch, between ones it does.

    Mirrors the VLA setup, where only the first `num_vlm_layers_to_use` Qwen
    layers take part in the loss while every layer's params stay trainable and
    tracked by the optimizer.
    """

    def __init__(self):
        super().__init__()
        self.used = nn.Linear(4, 4)
        self.unused = nn.Linear(7, 3)  # distinct shapes: a mis-zip would throw
        self.also_used = nn.Linear(4, 2)

    def forward(self, x):
        return self.also_used(self.used(x)).pow(2).mean()


def test_unused_params_keep_their_slot_in_the_accumulated_grad():
    """`g_total` must keep None holes so torchopt's positional zip stays aligned.

    Dropping the None entries makes the gradient pytree narrower than
    params/opt_state, and torchopt zips them by position: every gradient after
    the first hole gets paired with the wrong parameter's moments. That
    misalignment is invisible on a model whose every param is used, and fails as
    a shape-broadcast error (or, with matching shapes, silently) on one whose
    params aren't.
    """
    torch.manual_seed(0)
    model = _PartlyUnusedModel()
    trainer, state = Trainer.initialize(model, torchopt.adamw(1e-3))

    x = torch.randn(4, 4)
    micros = [Microbatch({"x": x[:2]}, 0.5), Microbatch({"x": x[2:]}, 0.5)]

    g_total, _ = trainer.accumulate_grads(state, micros)

    assert set(g_total) == set(state.params), "gradient tree lost a param's slot"
    assert g_total["unused.weight"] is None
    assert g_total["unused.bias"] is None
    assert g_total["used.weight"] is not None

    # The real check: the optimizer update must run and leave unused params be.
    new_state = trainer.apply_grads(state, g_total)
    torch.testing.assert_close(
        new_state.params["unused.weight"], state.params["unused.weight"]
    )
    assert not torch.allclose(
        new_state.params["used.weight"], state.params["used.weight"]
    )


def test_uneven_row_lengths_weight_by_valid_tokens(accum_dataset):
    """With ragged rows, weights follow valid-token counts, not 1/K.

    `weighted_causal_lm_ce` divides by the batch's valid positions, so a
    microbatch holding fewer real tokens must contribute proportionally less —
    plain 1/K would silently mis-scale the accumulated gradient.
    """
    rows = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14], [15, 16]]
    ds = Dataset.from_dict({"input_ids": rows, "labels": rows})
    stream = DataStream(ds, batch_size=4, device="cpu")

    micros = stream.microbatches(0, 2)
    weights = [m.weight for m in micros]

    assert sum(weights) == pytest.approx(1.0)
    # First microbatch holds the two long rows: 2x5 valid positions out of
    # 2x5 + 2x1 (padding positions carry label -100 and don't count).
    assert weights[0] == pytest.approx(10 / 12)
    assert weights[1] == pytest.approx(2 / 12)
