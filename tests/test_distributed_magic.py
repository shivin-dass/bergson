"""Test that FSDP and DDP produce equivalent MAGIC attribution scores.

Runs the CLI's run_magic twice (once FSDP, once DDP) with a tiny model
and asserts the resulting scores match.

Requires at least 2 CUDA devices.
"""

import tempfile

import pytest
import torch

from bergson.config import DataConfig, DistributedConfig
from bergson.magic.cli import MagicConfig, run_magic


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
def test_fsdp_ddp_scores_match():
    """FSDP and DDP should produce equivalent attribution scores."""
    world_size = min(torch.cuda.device_count(), 4)

    data = DataConfig(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train[:1024]",
        chunk_length=32,
    )
    dist_cfg = DistributedConfig(nproc_per_node=world_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        ddp_cfg = MagicConfig(
            run_path=f"{tmpdir}/ddp",
            model="trl-internal-testing/tiny-Phi3ForCausalLM",
            fsdp=False,
            data=data,
            query=data,
            batch_size=8,
            num_epochs=1,
            overwrite=True,
            num_subsets=2,
            distributed=dist_cfg,
        )
        fsdp_cfg = MagicConfig(
            run_path=f"{tmpdir}/fsdp",
            model="trl-internal-testing/tiny-Phi3ForCausalLM",
            fsdp=True,
            data=data,
            query=data,
            batch_size=8,
            num_epochs=1,
            overwrite=True,
            num_subsets=2,
            distributed=dist_cfg,
        )

        run_magic(ddp_cfg)
        run_magic(fsdp_cfg)

        ddp_scores = torch.load(f"{tmpdir}/ddp/scores.pt", weights_only=True)
        fsdp_scores = torch.load(f"{tmpdir}/fsdp/scores.pt", weights_only=True)

    assert (
        fsdp_scores.shape == ddp_scores.shape
    ), f"Shape mismatch: FSDP {fsdp_scores.shape} vs DDP {ddp_scores.shape}"

    atol = 1e-4
    rtol = 1e-3
    if not torch.allclose(fsdp_scores, ddp_scores, atol=atol, rtol=rtol):
        diff = (fsdp_scores - ddp_scores).abs()
        ratio = fsdp_scores.abs().mean() / ddp_scores.abs().mean()
        pytest.fail(
            f"FSDP and DDP scores differ.\n"
            f"  FSDP: mean|s|={fsdp_scores.abs().mean():.6f}, "
            f"sum={fsdp_scores.sum():.6f}\n"
            f"  DDP:  mean|s|={ddp_scores.abs().mean():.6f}, "
            f"sum={ddp_scores.sum():.6f}\n"
            f"  Ratio (FSDP/DDP): {ratio:.4f}\n"
            f"  Max abs diff: {diff.max():.6f}, "
            f"Mean abs diff: {diff.mean():.6f}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
def test_fsdp_ddp_scores_match_with_grad_clipping():
    """Gradient clipping is consistent across FSDP shards and DDP replicas.

    The global norm is reduced differently in each mode (FSDP sums partial squared
    norms across shards; DDP computes it from replicated grads), so matching scores
    confirm the cross-shard reduction is correct. The unclipped DDP run is the
    control that the chosen threshold actually clips.

    Scores here are tiny (~1e-6), so the checks are scale-free: clipping must change
    the scores by an O(1) fraction, while FSDP and DDP must agree to far tighter than
    that change — a broken cross-shard reduction would move scores by ~their own size.
    """
    world_size = min(torch.cuda.device_count(), 4)
    # tiny-Phi3 grad norms on this data are ~0.35, so 0.2 clips on every step.
    max_grad_norm = 0.2

    data = DataConfig(
        dataset="Salesforce/wikitext",
        subset="wikitext-2-raw-v1",
        split="train[:512]",
        chunk_length=32,
    )
    dist_cfg = DistributedConfig(nproc_per_node=world_size)

    def cfg(name, *, fsdp, clip):
        return MagicConfig(
            run_path=f"{tmpdir}/{name}",
            model="trl-internal-testing/tiny-Phi3ForCausalLM",
            fsdp=fsdp,
            data=data,
            query=data,
            batch_size=8,
            num_epochs=1,
            overwrite=True,
            num_subsets=2,
            max_grad_norm=max_grad_norm if clip else None,
            distributed=dist_cfg,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        run_magic(cfg("ddp_noclip", fsdp=False, clip=False))
        run_magic(cfg("ddp", fsdp=False, clip=True))
        run_magic(cfg("fsdp", fsdp=True, clip=True))

        ddp_noclip = torch.load(f"{tmpdir}/ddp_noclip/scores.pt", weights_only=True)
        ddp_scores = torch.load(f"{tmpdir}/ddp/scores.pt", weights_only=True)
        fsdp_scores = torch.load(f"{tmpdir}/fsdp/scores.pt", weights_only=True)

    assert fsdp_scores.shape == ddp_scores.shape

    scale = ddp_noclip.abs().max()
    clip_effect = (ddp_scores - ddp_noclip).abs().max()
    fsdp_ddp_diff = (fsdp_scores - ddp_scores).abs().max()

    assert clip_effect > 0.1 * scale, (
        f"clip barely changed the scores ({clip_effect:.2e} vs scale {scale:.2e}); "
        "lower max_grad_norm so it bites"
    )
    assert fsdp_ddp_diff < 0.05 * clip_effect, (
        f"clipped FSDP and DDP scores differ too much: {fsdp_ddp_diff:.2e} "
        f"(clip effect {clip_effect:.2e}) — cross-shard norm reduction is likely wrong"
    )


if __name__ == "__main__":
    test_fsdp_ddp_scores_match()
    test_fsdp_ddp_scores_match_with_grad_clipping()
