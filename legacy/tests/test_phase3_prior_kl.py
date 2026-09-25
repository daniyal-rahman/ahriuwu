"""Phase-3 PMPO prior-KL regression tests: the offset the prior is sliced at.

WHY THIS FILE EXISTS
--------------------
Phase 3 anchors the policy to a frozen "behavioural prior" that is literally
``copy.deepcopy(policy_head)``. That deepcopy is the whole test oracle:

    at Phase-3 step 0 the prior IS the policy, so KL[pi || prior] must be
    EXACTLY 0.0 -- not small, not 1e-3, exactly 0.

It was not. ``train_imagination.py`` sliced the policy at ``MTP_OFFSET`` (1, the
first BC-trained head) and the prior at a hardcoded ``0`` (a head BC never
trains, still at zero-init = exact uniform). The KL measured 7.29 nats on a real
checkpoint, and since ``KL(pi || uniform) = logK - H(pi)``, the ``--pmpo-beta``
term was an ENTROPY BONUS: Phase 3 was actively erasing the behaviour-cloned
policy rather than anchoring to it. Fixed in ``af72fa2``.

WHY THE OLD SMOKE TEST DID NOT CATCH IT
---------------------------------------
It perturbed MTP head 0 off zero-init first, with the comment "so the prior KL is
non-degenerate". Head 0 being at exact zero-init is precisely the condition that
makes a mis-sliced prior visible; perturbing it manufactured a plausible-looking
1.9e-3 and hid a 7.29-nat bug. A test built around the bug is worse than no test,
so these three assertions are chosen so that the BUG FAILS THEM:

  1. step-0 KL is exactly 0            -- buggy code gives ~7 nats.        FAILS
  2. perturbing head 0 alone keeps 0   -- buggy code reads head 0.         FAILS
  3. perturbing head MTP_OFFSET moves it -- guards against the lazy fix of
     returning a constant 0, which would satisfy 1 and 2.                  PASSES

(1) and (2) alone are satisfiable by ``kl = 0``; (3) alone is satisfied by the
bug. Only all three together pin the slice to the offset the policy reads.

Heads are left at the state a REAL checkpoint has them in: offset 0 at exact
zero-init (dead -- it appears in no BC loss), offsets >= 1 trained. Never perturb
head 0 here.

Run:  PYTHONPATH=src python tests/test_phase3_prior_kl.py
      (or: pytest tests/test_phase3_prior_kl.py)
"""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_imagination as TI  # noqa: E402
from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM  # noqa: E402
from ahriuwu.models import PolicyHead  # noqa: E402

OFFSET = TI.MTP_OFFSET
MODES = [("axis", False), ("axis", True), ("joint_noop", False)]
_LABEL = {("axis", False): "axis", ("axis", True): "axis+gate", ("joint_noop", False): "joint_noop"}

D, BINS, MTP, B, H = 32, 7, 4, 3, 2


def _make_head(mode, gate, seed=0):
    """A PolicyHead in the state every real Phase-2 checkpoint is in: MTP offset 0
    at exact zero-init (BC never trains it), offsets >= 1 trained."""
    torch.manual_seed(seed)
    head = PolicyHead(input_dim=D, num_abilities=len(ABILITY_KEYS), hidden_dim=16,
                      mtp_length=MTP, movement_dim=MOVEMENT_DIM, movement_bins=BINS,
                      movement_gate=gate, movement_mode=mode)
    with torch.no_grad():
        for n in range(1, MTP):          # n >= 1 ONLY. Head 0 stays at zero-init.
            head.heads[n].weight.normal_(0, 0.5)
            head.heads[n].bias.normal_(0, 0.5)
            head.movement_heads[n].weight.normal_(0, 0.5)
            head.movement_heads[n].bias.normal_(0, 0.5)
            if gate:
                head.gate_heads[n].weight.normal_(0, 0.5)
                head.gate_heads[n].bias.normal_(0, 0.5)
    return head


def _roll(head, seed=1):
    """A minimal imagined rollout, in the shapes imagine() produces."""
    torch.manual_seed(seed)
    prev = torch.randint(0, BINS, (B, H, MOVEMENT_DIM))
    if head.movement_mode == "joint_noop":
        prev = head.joint_encode(prev[..., 0], prev[..., 1])
        idx = torch.randint(0, head.movement_classes, (B, H))
    else:
        idx = torch.randint(0, BINS, (B, H, MOVEMENT_DIM))
    return {
        "agent_outs": torch.randn(B, H, D),
        "ability_acts": torch.randint(0, 2, (B, H, len(ABILITY_KEYS))).float(),
        "movement_acts": torch.rand(B, H, MOVEMENT_DIM),
        "movement_idx": idx,
        "movement_prev": prev,
        "rewards": torch.rand(B, H) * 0.01,
        "values": torch.zeros(B, H),
    }


def _kl(head, prior, roll):
    """The KL train_imagination actually optimises, via the real run_step."""
    from ahriuwu.models import RunningRMS, ValueHead
    args = TI.argparse.Namespace(gamma=0.997, lambda_=0.95, pmpo_alpha=0.5,
                                 pmpo_beta=0.3, num_buckets=41)
    vh = ValueHead(input_dim=D, hidden_dim=16, num_buckets=args.num_buckets)
    rms = {"value": RunningRMS(), "policy": RunningRMS()}
    _, info = TI.run_step(roll, head, prior, vh, args, "cpu", torch.float32, rms)
    return info["kl"].item()


def test_step0_kl_is_exactly_zero():
    """The deepcopy invariant: prior IS the policy at step 0, so KL == 0 exactly.

    Fails at ~7 nats if the prior is sliced at an offset the policy does not read.
    """
    for mode, gate in MODES:
        head = _make_head(mode, gate)
        prior = copy.deepcopy(head)
        prior.eval()
        prior.requires_grad_(False)
        kl = _kl(head, prior, _roll(head))
        assert kl == 0.0, (
            f"[{_LABEL[(mode, gate)]}] prior is a deepcopy of the policy, so the "
            f"step-0 KL must be EXACTLY 0.0, got {kl:.6e}. A nonzero value here "
            f"means the prior is read at a different MTP offset than the policy.")
        print(f"OK: [{_LABEL[(mode, gate)]}] step-0 KL == {kl:.1f} exactly.")


def test_untrained_offset_zero_does_not_reach_the_kl():
    """Perturbing ONLY the dead head 0 must leave the KL at exactly 0.

    This is the direct falsifier for the original bug. With the prior sliced at 0
    the perturbation lands squarely in the KL and this assertion fails; with both
    sides sliced at MTP_OFFSET, head 0 is simply not read.
    """
    for mode, gate in MODES:
        head = _make_head(mode, gate)
        prior = copy.deepcopy(head)
        prior.eval()
        prior.requires_grad_(False)
        with torch.no_grad():   # policy's head 0 only -- the offset nothing reads
            head.heads[0].weight.normal_(0, 1.0)
            head.movement_heads[0].weight.normal_(0, 1.0)
            if gate:
                head.gate_heads[0].weight.normal_(0, 1.0)
        kl = _kl(head, prior, _roll(head))
        assert kl == 0.0, (
            f"[{_LABEL[(mode, gate)]}] MTP offset 0 is not the offset Phase 3 "
            f"reads (MTP_OFFSET={OFFSET}), so perturbing it must not move the KL; "
            f"got {kl:.6e}. The prior is being sliced at 0.")
        print(f"OK: [{_LABEL[(mode, gate)]}] head 0 perturbed, KL still {kl:.1f}.")


def test_read_offset_does_reach_the_kl():
    """Perturbing head MTP_OFFSET must move the KL off zero.

    Without this, a KL hardcoded to 0 would pass the two tests above. This is the
    sufficiency half: the term has to actually be live at the offset PMPO scores.
    """
    for mode, gate in MODES:
        head = _make_head(mode, gate)
        prior = copy.deepcopy(head)
        prior.eval()
        prior.requires_grad_(False)
        with torch.no_grad():
            head.heads[OFFSET].weight.add_(torch.randn_like(head.heads[OFFSET].weight) * 0.5)
            head.movement_heads[OFFSET].weight.add_(
                torch.randn_like(head.movement_heads[OFFSET].weight) * 0.5)
        kl = _kl(head, prior, _roll(head))
        assert kl > 1e-4, (
            f"[{_LABEL[(mode, gate)]}] perturbing the offset PMPO reads "
            f"(MTP_OFFSET={OFFSET}) must produce a nonzero KL, got {kl:.6e}. "
            f"The KL term is dead.")
        print(f"OK: [{_LABEL[(mode, gate)]}] head {OFFSET} perturbed, KL = {kl:.4f} > 0.")


def test_gate_is_inside_the_gated_kl():
    """For a gated head the movement law is the sticky MIXTURE, so the gate is
    part of the policy. Perturbing only the gate must move the KL -- otherwise the
    "should I issue an order at all" decision is unregularised and free to drift
    to always-fire while the KL term still reads low."""
    head = _make_head("axis", True)
    prior = copy.deepcopy(head)
    prior.eval()
    prior.requires_grad_(False)
    with torch.no_grad():
        head.gate_heads[OFFSET].bias.add_(2.0)
    kl = _kl(head, prior, _roll(head))
    assert kl > 1e-4, (
        f"the movement gate is a factor of the gated policy but does not reach "
        f"the KL (got {kl:.6e}); it would train unanchored.")
    print(f"OK: [axis+gate] gate perturbed, KL = {kl:.4f} > 0.")


if __name__ == "__main__":
    test_step0_kl_is_exactly_zero()
    test_untrained_offset_zero_does_not_reach_the_kl()
    test_read_offset_does_reach_the_kl()
    test_gate_is_inside_the_gated_kl()
    print("\nAll Phase-3 prior-KL regression tests passed.")
