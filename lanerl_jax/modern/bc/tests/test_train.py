"""The BC trainer's pieces on a synthetic game: windows cover every decision once, the loss is finite and
masked where it should be, and evaluation reports per-button recall."""
import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.actions import MODERN_BUTTON_INDEX as B
from lanerl_jax.modern.bc import train as T
from lanerl_jax.modern.obs import MODERN_WORLD_SELF_DIM
from lanerl_jax.modern.rl.policy import LanePolicy, make_policy_config
from lanerl_jax.modern.train import DEFAULT_BUTTONS_OFF, SMALL_POLICY


def _game(name, n=50, seed=0, group="melee", kit=True):
    rng = np.random.default_rng(seed)
    button = rng.choice([B["noop"], B["move"], B["q"]], n, p=[0.7, 0.25, 0.05]).astype(np.int32)
    click = button == B["move"]
    return T.Game(name, "train", group, kit, rng.random((n, 32, 20)).astype(np.float16), rng.random((n, 32)) > 0.5,
                  rng.random((n, MODERN_WORLD_SELF_DIM)).astype(np.float32), rng.random((n, 6)).astype(np.float32),
                  button, rng.integers(0, 96, n).astype(np.int32), rng.integers(0, 54, n).astype(np.int32),
                  rng.random(n) > 0.1, click, click & (rng.random(n) > 0.5), click & (rng.random(n) > 0.5),
                  champion="Garen", role="top")


def test_windows_cover_every_decision_once():
    games = [_game("a", 50), _game("b", 70)]
    for rng in (None, np.random.default_rng(3)):
        items = T.windows(games, 16, rng)
        seen = {i: np.zeros(len(g.button), int) for i, g in enumerate(games)}
        for gi, s in items:
            e = min(s + 16, len(games[gi].button))
            seen[gi][s:e] += 1
        assert all((v >= 1).all() for v in seen.values())
        if rng is None:
            assert all((v == 1).all() for v in seen.values())


def test_loss_is_finite_and_masks_burn_in_and_self_casts():
    games = [_game("a", 40)]
    off = tuple(sorted(B[b] for b in DEFAULT_BUTTONS_OFF))
    policy = LanePolicy(make_policy_config("modern-world-v1", core="gru", core_norm=True, core_residual=True, **SMALL_POLICY), off)
    b = T.batch_of(games, T.windows(games, 16, None), 16, 4, None)
    assert b["weight"][:, :4].sum() == 0 and b["click"][b["button"] == B["q"]].sum() == 0
    jb = {k: jnp.asarray(v) for k, v in b.items()}
    params = policy.init(jax.random.key(0), jb["entities"][0, 0], jb["mask"][0, 0], jb["self"][0, 0],
                         jb["global_"][0, 0], policy.initial_carry(()))
    loss = T.make_loss(policy, 1.0, np.ones(20, np.float32))
    total, info = loss(params, jb)
    assert np.isfinite(float(total)) and float(info["button_acc"]) <= 1.0
    grads = jax.grad(lambda p: loss(p, jb)[0])(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))
    ev = T.evaluate(jax.jit(loss.per_decision), params, games, 16, 4, 4)
    assert set(ev) == {"all", "melee", "kit_known", "champion:Garen", "role:top"}
    assert "move" in ev["all"]["recall"]
    assert "move" in ev["all"]["joint_button_click_exact_by"]
    assert all(0. <= x <= 1. for x in ev["all"]["joint_button_click_exact_by"].values())
    assert ev["all"]["decisions"] == int(b["weight"].sum())
    import json
    json.dumps(ev)  # native numeric metrics must serialize as numbers, not numpy scalars or strings


def test_dropout_zeroes_enemy_identity():
    games = [_game("a", 40)]
    rng = np.random.default_rng(0)
    b = T.batch_of(games, T.windows(games, 40, None), 40, 0, rng, enemy_id_dropout=1.0, stat_dropout=1.0)
    assert np.all(b["self"][:, :, T.enemy_id_columns("modern-world-v1")] == 0.0)
    assert np.all(b["self"][:, 1:, T.stat_columns("modern-world-v1")] == b["self"][:, :1, T.stat_columns("modern-world-v1")])


def test_hold_weight_scales_held_decisions():
    games = [_game("a", 40)]
    b1 = T.batch_of(games, T.windows(games, 40, None), 40, 0, None, hold_weight=1.0)
    b3 = T.batch_of(games, T.windows(games, 40, None), 40, 0, None, hold_weight=0.3)
    held = b1["held"][0]
    assert held.any() and np.allclose(b3["weight"][0][held], 0.3 * b1["weight"][0][held])
    assert np.array_equal(b3["weight"][0][~held], b1["weight"][0][~held])


def test_timing_targets_spread_a_press_onto_ready_neighbours_only():
    n = 30
    g = _game("a", n)
    col = T._self_columns("modern-world-v1")
    button = np.full(n, B["move"], np.int32)
    button[15] = B["q"]
    button[12] = B["stop"]                          # not noop/move: keeps its own target
    sv = g.self_vec.copy()
    sv[:, col["cd_q"]] = 0.0
    sv[16:, col["cd_q"]] = 0.9                      # on cooldown after the press
    sv[10, col["cd_q"]] = 0.5                       # not ready at one earlier decision
    g = g._replace(button=button, self_vec=sv, valid=np.ones(n, bool), held=np.zeros(n, bool))
    s = T.timing_targets(g, 0.2, 0.5, "modern-world-v1")
    spread = np.flatnonzero(s.alt_m > 0)
    assert set(spread) == {11, 13, 14}
    assert np.all(s.alt_b[spread] == B["q"])
    assert np.isclose(float(s.own_m[15]) + float(s.alt_m.astype(np.float32).sum()), 1.0, atol=1e-2)
    assert s.alt_m[14] > s.alt_m[13] > s.alt_m[11] and float(s.own_m[15]) > float(s.alt_m[14])
    assert np.allclose(s.own_m[spread].astype(np.float32) + s.alt_m[spread].astype(np.float32), 1.0, atol=1e-3)
    off = tuple(sorted(B[b] for b in DEFAULT_BUTTONS_OFF))
    policy = LanePolicy(make_policy_config("modern-world-v1", core="gru", core_norm=True, core_residual=True, **SMALL_POLICY), off)
    b = T.batch_of([s], T.windows([s], n, None), n, 0, None)
    jb = {k: jnp.asarray(v) for k, v in b.items()}
    params = policy.init(jax.random.key(0), jb["entities"][0, 0], jb["mask"][0, 0], jb["self"][0, 0],
                         jb["global_"][0, 0], policy.initial_carry(()))
    loss = T.make_loss(policy, 1.0, np.ones(20, np.float32))
    d = loss.per_decision(params, jb)
    lb = jax.nn.log_softmax(loss.forward(params, jb).button, -1)[0]
    want = -(b["own_m"][0, 14] * lb[14, B["move"]] + b["alt_m"][0, 14] * lb[14, B["q"]])
    assert np.isclose(float(d["ce_raw"][0, 14]), float(want), rtol=1e-4)


def test_dilate_marks_within_k():
    x = np.zeros((1, 12), bool)
    x[0, 5] = True
    assert np.flatnonzero(T._dilate(x, 2)[0]).tolist() == [3, 4, 5, 6, 7]
