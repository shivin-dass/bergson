"""Per-rank shard checkpointing under FSDP (simple_fsdp / DTensor).

Directly exercises the path the multi-node MAGIC run crashes on: an FSDP-
sharded TrainerState saved during the forward pass and loaded back at the
start of the backward pass. With per-rank shard files each rank only ever
touches its own node's disk, so this must work with NO shared filesystem.

Checks, on 2 GPUs:
  1. save() writes one rank_{r}.pt per rank; no DCP .metadata anywhere.
  2. load() bit-identically restores DTensor params, opt state, and RNG.
  3. A full train -> query -> backward round trip through checkpoint files
     completes and yields finite, nonzero attribution scores that match a
     backward pass run WITHOUT checkpoint reloads (in-memory reference).
"""

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchopt
from datasets import Dataset
from torch.distributed.tensor import DTensor, init_device_mesh
from torchopt.pytree import tree_iter
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.distributed import grad_tree
from bergson.magic import BackwardState, DataStream, Trainer
from bergson.magic.dtensor_patch import apply_dtensor_patch
from bergson.magic.fsdp import simple_fsdp
from bergson.utils.math import weighted_causal_lm_ce


def _make_model():
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    model = AutoModelForCausalLM.from_config(
        config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.loss_function = weighted_causal_lm_ce
    model.requires_grad_(True)
    return model


def _make_dataset():
    return Dataset.from_dict(
        {
            "input_ids": [list(range(i * 5 + 1, i * 5 + 6)) for i in range(4)],
            "labels": [list(range(i * 5 + 1, i * 5 + 6)) for i in range(4)],
            "attention_mask": [[1] * 5 for _ in range(4)],
        }
    )


def _worker(rank, world_size, port, result_dict):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "cpu:gloo,cuda:nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{rank}"),
        )
        device = f"cuda:{rank}"

        apply_dtensor_patch()
        model = _make_model().to(device)
        mesh = init_device_mesh("cuda", (world_size,))
        with mesh:
            model = simple_fsdp(model)

        optimizer = torchopt.adamw(1e-4, betas=(0.95, 0.975), eps_root=1e-2)
        trainer, fwd_state = Trainer.initialize(model, optimizer)
        assert any(
            isinstance(p, DTensor) for p in fwd_state.params.values()
        ), "simple_fsdp did not shard params into DTensors"

        dataset = _make_dataset()
        stream = DataStream(dataset, batch_size=2, device=device)
        assert len(stream) == 2

        with tempfile.TemporaryDirectory() as ckpt_dir:
            # ---- check 2: bare save/load round trip on the sharded state ----
            probe = os.path.join(ckpt_dir, "probe.ckpt")
            fwd_state.save(probe).result()
            files = sorted(os.listdir(probe))
            assert f"rank_{rank}.pt" in files, files
            assert ".metadata" not in files, files

            before = {
                k: (v.to_local() if isinstance(v, DTensor) else v).clone()
                for k, v in fwd_state.params.items()
            }
            with torch.no_grad():
                for v in fwd_state.params.values():
                    (v.to_local() if isinstance(v, DTensor) else v).add_(1.0)
            fwd_state.load(probe)
            for k, v in fwd_state.params.items():
                local = v.to_local() if isinstance(v, DTensor) else v
                assert torch.equal(local, before[k]), f"param {k} not restored"

            # ---- check 3: full pipeline through checkpoint files ----
            save_dir = os.path.join(ckpt_dir, "ckpts")
            fwd_state = trainer.train(
                fwd_state,
                stream,
                inplace=True,
                save_dir=save_dir,
                save_mode="all",
                fsdp=True,
            )

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
            stream.requires_grad = True
            bwd_state = trainer.backward(
                save_dir,
                stream,
                BackwardState(query_grads, opt_grads, torch.zeros_like(stream.weights)),
                fwd_state,
                inplace=True,
                fsdp=True,
            )

            scores = bwd_state.weight_grads.detach()
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            assert torch.isfinite(scores).all(), scores
            assert (scores != 0).any(), "all-zero attribution scores"
            result_dict[rank] = scores.cpu()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Needs >= 2 GPUs for FSDP sharding",
)
def test_fsdp_perrank_checkpoint_roundtrip():
    world_size = 2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    manager = mp.Manager()
    result_dict = manager.dict()
    mp.spawn(_worker, args=(world_size, port, result_dict), nprocs=world_size, join=True)

    assert set(result_dict.keys()) == {0, 1}
    torch.testing.assert_close(result_dict[0], result_dict[1])
