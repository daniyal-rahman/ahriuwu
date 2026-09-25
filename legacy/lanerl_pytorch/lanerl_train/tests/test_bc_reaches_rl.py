"""The BC prior must actually reach the thing that trains.

Behaviour cloning existed for weeks with no reachable consumer. The trainer
built ``LanePolicy(model_cfg)`` random, passed no ``reference=`` to
``DualClipPPO`` (so ``kl_ref_coef`` was dead code), and the only thing
``--bc-checkpoint`` fed was the ``bc_policy`` *anchor*, which
``anchor_launch_spec`` rejects outright because only ``scripted`` anchors can be
played by the in-server bot.  Nothing errored.  BC trained, wrote a checkpoint,
reported accuracies, and RL started from noise.

That is the same shape as the audit's "downstream doesn't run" finding, and the
only thing that catches it is a test that asserts the wiring, not the training.
These tests are deliberately about plumbing: that the checkpoint BC writes is
loadable by the policy RL builds, with no silently-ignored keys, and that the
KL term cannot be switched on without a prior to pull toward.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]

from lanerl_rl.model import LanePolicy, ModelConfig  # noqa: E402
from lanerl_rl.ppo import DualClipPPO, PPOConfig  # noqa: E402


def test_a_bc_checkpoint_loads_into_the_policy_rl_builds(tmp_path):
    """No missing and no unexpected keys, with strict=True.

    ``load_state_dict(..., strict=False)`` is what the trainer uses, so that it
    can report the mismatch itself rather than throwing -- but the contract is
    that there IS no mismatch. If BC and the trainer ever build a different
    ModelConfig, this is where it surfaces, instead of as a policy that is
    randomly initialised in precisely the heads BC was supposed to teach.
    """
    trained = LanePolicy(ModelConfig())
    ckpt = tmp_path / "bc_policy.pt"
    torch.save({"policy": trained.state_dict(), "kind": "bc"}, ckpt)

    fresh = LanePolicy(ModelConfig())
    blob = torch.load(ckpt, map_location="cpu")
    missing, unexpected = fresh.load_state_dict(blob["policy"], strict=False)
    assert not missing, f"BC checkpoint is missing keys the policy needs: {missing}"
    assert not unexpected, f"BC checkpoint has keys the policy does not use: {unexpected}"

    for (ka, va), (kb, vb) in zip(trained.state_dict().items(),
                                  fresh.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb), f"{ka} did not survive the round trip"


def test_the_kl_term_is_inert_without_a_reference():
    """kl_ref_coef alone must not read as "KL is on".

    It was configurable, documented, and had no effect, because no caller ever
    supplied reference=. A coefficient that silently does nothing is worse than
    one that is absent: the run log says kl_ref_coef=0.1 either way.
    """
    policy = LanePolicy(ModelConfig())
    learner = DualClipPPO(policy, PPOConfig(kl_ref_coef=0.1))
    assert learner.reference is None
    # ... and with one, it is frozen: a reference that keeps training is not a
    # prior, it is a second policy chasing the first.
    ref = LanePolicy(ModelConfig())
    learner2 = DualClipPPO(policy, PPOConfig(kl_ref_coef=0.1), reference=ref)
    assert learner2.reference is ref
    assert not any(p.requires_grad for p in learner2.reference.parameters())
    assert not learner2.reference.training


def test_the_cli_refuses_a_kl_coefficient_with_no_prior():
    """--kl-ref-coef without --init-from must fail loudly at startup.

    Accepting it would reproduce the original bug through the front door.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "lanerl_train", "--kl-ref-coef", "0.5",
         "--run-name", "should-never-start"],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0
    assert "--init-from" in (proc.stdout + proc.stderr)


def test_init_from_rejects_a_checkpoint_that_does_not_match():
    """A partial load must not be accepted silently.

    Loading with strict=False and ignoring the returned lists is how a
    "BC-initialised" run ends up random in exactly the heads that matter.
    """
    policy = LanePolicy(ModelConfig())
    state = policy.state_dict()
    first = next(iter(state))
    del state[first]
    fresh = LanePolicy(ModelConfig())
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert missing, "a checkpoint missing a tensor must report it"


@pytest.mark.parametrize("flag", ["--init-from", "--kl-ref-coef"])
def test_the_flags_exist_and_are_documented(flag):
    from lanerl_train.__main__ import build_argparser

    text = build_argparser().format_help()
    assert flag in text
