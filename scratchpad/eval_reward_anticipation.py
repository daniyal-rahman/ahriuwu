#!/usr/bin/env python3
"""Does the Phase-2 reward head ANTICIPATE income (last-hits), or only report them?

Go/no-go for "v1.5": if the head can say "gold is coming in the next 3-10 frames
(150-500ms)" from causally-available latents, a policy can use it to time an AA
and Bronze farming is plausible WITHOUT any minion-HP computer vision. If it only
fires ON/AFTER the gold arrives, the plan is dead.

Verified before writing this:
  * dynamics temporal attention is CAUSAL (layers.py Attention mode="temporal"
    -> _causal_mask_mod / self.causal_mask), agent-token self-attn is causal,
    agent cross-attn is per-frame. So token t sees frames <= t ONLY.
  * MTP offset semantics (train_agent_finetune.reward_mtp_loss):
        pred = reward_logits[:, :T-n, n, :] ; tgt = symlog(rewards)[:, n:]
    -> head n at token t predicts reward at t+n. n = 0..8.
  * reward[t] = 1e-3 * (gold_total[t] - gold_total[t-1])  (rewards/reward.py)
    i.e. reward at t = income that ALREADY landed between t-1 and t.
  * labels fps = 20 -> 1 frame = 50 ms; latents are stride-1 (frame_indices
    diff == 1), so offset n frames = 50*n ms.

Adversarial controls this runs (the point of the script):
  A) AUTOCORRELATION BASELINE. Minions die in bunches; "income recently" predicts
     "income soon" with zero perception. The head must BEAT sum(r[t-k..t]).
  B) CLEAN population: only frames with NO income in [t-lookback, t]. Kills the
     "gold is currently flowing" cue, so a hit here is real anticipation.
  C) ACTION LEAK. The action token at frame t is an INPUT. If the human's AA
     press at t is what drives the "anticipation", the signal is circular (a
     policy cannot use "I already pressed AA" to decide to press AA). Re-run
     with abilities zeroed and compare.
  D) SHUFFLE control: labels permuted -> AUC must collapse to ~0.5.
  E) per-offset collapse check: if pred[t,n] == pred[t,0] for all n, the MTP
     heads are degenerate and "0..8-frames-ahead" is vacuous.

Usage:
  PYTHONPATH=src python scratchpad/eval_reward_anticipation.py \
      --matches NA1_5549981347 --latents-dir /srv/nfs/datasets/replay_latents_v7_heldout \
      --tag heldout
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "src")
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.models import create_dynamics, RewardHead, DiffusionSchedule
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset

FPS = 20.0


def load_phase2(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    cfg = ck.get("dynamics_config") or {}
    dyn_state = ck["dynamics_state_dict"]
    if any(k.startswith("_orig_mod.") for k in dyn_state):
        dyn_state = {k.replace("_orig_mod.", ""): v for k, v in dyn_state.items()}
    use_actions = cfg.get("use_actions", any("action_embed." in k for k in dyn_state))
    dyn = create_dynamics(
        size=a.get("model_size", "medium"), latent_dim=cfg.get("latent_dim", a.get("latent_dim", 32)),
        use_agent_tokens=True, use_actions=use_actions, num_tasks=1,
        agent_layers=a.get("agent_layers", 4), use_qk_norm=not a.get("no_qk_norm", False),
        soft_cap=a.get("soft_cap", 50.0) or None,
        num_register_tokens=a.get("num_register_tokens", 8),
        num_kv_heads=a.get("num_kv_heads", None),
    ).to(device)
    miss, unexp = dyn.load_state_dict(dyn_state, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert len(miss) + len(unexp) <= 10, f"ARCH MISMATCH miss={miss[:5]} unexp={unexp[:5]}"
    dyn.eval().requires_grad_(False)
    rh = RewardHead(input_dim=dyn.model_dim, hidden_dim=a.get("hidden_dim", 256),
                    num_buckets=a.get("num_buckets", 255),
                    mtp_length=a.get("mtp_length", 9)).to(device)
    rh.load_state_dict(ck["reward_head_state_dict"])
    rh.eval().requires_grad_(False)
    print(f"[ckpt] {path}\n       gs={ck.get('global_step')} epoch={ck.get('epoch')} "
          f"use_actions={use_actions} mtp={a.get('mtp_length')} train_seq_len={a.get('seq_len')} "
          f"tau_ctx={a.get('tau_ctx', 0.9)}", flush=True)
    return dyn, rh, int(a.get("mtp_length", 9))


def auc(pos, neg):
    """Rank AUC: P(pred_pos > pred_neg), ties = 0.5."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    # average ranks so exact ties (very common with a saturated head) score 0.5
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(len(allv), dtype=np.float64)
    # tie correction
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    rp = ranks[: len(pos)]
    return float((rp.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def collect(dyn, rh, mtp, match, latents_dir, labels_root, seq_len, max_windows,
            dev, tau_mode, ablate_abilities, seed=0):
    """Run the frozen stack over non-overlapping windows.

    Returns list of dicts with per-window (T,) rewards and (T, mtp) predictions.
    Windows are non-overlapping so nothing is double-counted in the AUCs.
    """
    schedule = DiffusionSchedule(device=dev)
    tmp = tempfile.mkdtemp()
    src = glob.glob(f"{latents_dir}/{match}.pt")[0]
    os.symlink(os.path.abspath(src), f"{tmp}/{match}.pt")
    ds = ReplayLatentSequenceDataset(
        latents_dir=tmp, labels_root=labels_root, outcomes={match: False},
        sequence_length=seq_len, stride=seq_len)
    n = len(ds)
    idx = np.arange(n) if max_windows <= 0 or max_windows >= n else \
        np.linspace(0, n - 1, max_windows).astype(int)
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in idx:
        s = ds[int(i)]
        z0 = s["latents"].float().unsqueeze(0).to(dev)
        T = z0.shape[1]
        if tau_mode == "clean":
            tau = torch.ones(1, T, device=dev)
            z_noisy = z0
        else:  # "train": tau ~ U[0.9, 1.0], the Phase-2 context regime
            tau = 0.9 + torch.rand(1, T, generator=g).to(dev) * 0.1
            z_noisy, _ = schedule.add_noise(z0, tau)
        actions = None
        if dyn.use_actions:
            actions = {"movement": s["actions"]["movement"].float().unsqueeze(0).to(dev)}
            for k in ABILITY_KEYS:
                v = s["actions"][k].long().unsqueeze(0).to(dev)
                if ablate_abilities:
                    v = torch.zeros_like(v)
                actions[k] = v
            if "cursor_valid" in s["actions"]:
                actions["cursor_valid"] = s["actions"]["cursor_valid"].unsqueeze(0).to(dev)
        d_one = torch.ones(1, dtype=torch.long, device=dev)
        with torch.no_grad():
            _, agent_out = dyn(z_noisy, tau, step_size=d_one, actions=actions)
            pred = rh.predict(agent_out)[0]           # (T, mtp) original scale
        out.append({
            "r": s["rewards"].numpy().astype(np.float64),          # (T,)
            "p": pred.float().cpu().numpy().astype(np.float64),    # (T, mtp)
            "aa": s["actions"]["AA"].numpy(),                      # (T,)
            "start": int(s["start_frame"]),
        })
    return out


# ───────────────────────────── metrics ─────────────────────────────

def window_arrays(windows, mtp):
    """Stack per-window arrays, keeping window identity so labels never cross
    a window boundary (each window is a contiguous chunk of real time, but
    windows are non-adjacent when subsampled)."""
    return windows


def anticipation_table(windows, mtp, thresh, lookback, horizons, rng):
    """For each horizon k: build income_within_k, score with several predictors."""
    rows = []
    for k in horizons:
        # collect over all windows
        lab, cur, recent, longrecent = [], [], [], []
        mtp_cum, mtp_max, mtp_off = [], [], {n: [] for n in range(mtp)}
        clean_mask = []
        for w in windows:
            r, p = w["r"], w["p"]
            T = len(r)
            hi = T - k                      # t must have t+k < T
            if hi <= lookback:
                continue
            t = np.arange(lookback, hi)
            # future income in (t, t+k]
            csum = np.concatenate([[0.0], np.cumsum(r)])
            fut = csum[t + k + 1] - csum[t + 1]
            lab.append(fut >= thresh)
            cur.append(p[t, 0])
            mtp_cum.append(p[t, 1:min(k, mtp - 1) + 1].sum(axis=1) if mtp > 1 else p[t, 0] * 0)
            mtp_max.append(p[t, 1:].max(axis=1) if mtp > 1 else p[t, 0] * 0)
            for n in range(mtp):
                mtp_off[n].append(p[t, n])
            # BASELINE A: income already seen in the immediate past (autocorrelation)
            recent.append(csum[t + 1] - csum[np.maximum(t + 1 - k, 0)])
            longrecent.append(csum[t + 1] - csum[np.maximum(t + 1 - lookback, 0)])
            # CLEAN: no income at all in [t-lookback, t]
            clean_mask.append((csum[t + 1] - csum[t + 1 - lookback]) <= 0.0)
        if not lab:
            continue
        cat = lambda xs: np.concatenate(xs)
        lab = cat(lab); cur = cat(cur); recent = cat(recent)
        longrecent = cat(longrecent); clean = cat(clean_mask)
        mtp_cum = cat(mtp_cum); mtp_max = cat(mtp_max)
        offs = {n: cat(v) for n, v in mtp_off.items()}

        def scored(mask):
            m = mask
            pos, neg = lab & m, (~lab) & m
            d = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
            if d["n_pos"] < 10 or d["n_neg"] < 10:
                return d
            d["auc_cur(n=0)"] = auc(cur[pos], cur[neg])
            d["auc_mtp_cum"] = auc(mtp_cum[pos], mtp_cum[neg])
            d["auc_mtp_max"] = auc(mtp_max[pos], mtp_max[neg])
            d["auc_base_recent"] = auc(recent[pos], recent[neg])
            d["auc_base_long"] = auc(longrecent[pos], longrecent[neg])
            sh = rng.permutation(len(lab))
            d["auc_shuffle"] = auc(cur[lab[sh] & m], cur[(~lab[sh]) & m])
            for n in range(mtp):
                d[f"auc_off{n}"] = auc(offs[n][pos], offs[n][neg])
            return d

        allm = np.ones_like(lab, dtype=bool)
        rows.append({"k": k, "ms": int(1000 * k / FPS),
                     "ALL": scored(allm), "CLEAN": scored(clean),
                     "clean_frac": float(clean.mean())})
    return rows


def per_offset_exact(windows, mtp, thresh, lookback, rng):
    """The head's own training target: pred[t,n] vs reward[t+n] >= thresh.
    Restricted to CLEAN frames (no income in [t-lookback, t]) so it must be
    real anticipation, not 'gold currently flowing'."""
    rows = []
    for n in range(mtp):
        pos_a, neg_a, pos_c, neg_c = [], [], [], []
        for w in windows:
            r, p = w["r"], w["p"]
            T = len(r)
            if T - n <= lookback:
                continue
            t = np.arange(lookback, T - n)
            csum = np.concatenate([[0.0], np.cumsum(r)])
            y = r[t + n] >= thresh
            clean = (csum[t + 1] - csum[t + 1 - lookback]) <= 0.0
            s = p[t, n]
            pos_a.append(s[y]); neg_a.append(s[~y])
            pos_c.append(s[y & clean]); neg_c.append(s[(~y) & clean])
        f = lambda xs: np.concatenate(xs) if xs else np.array([])
        pa, na, pc, nc = f(pos_a), f(neg_a), f(pos_c), f(neg_c)
        rows.append({
            "n": n, "ms": int(1000 * n / FPS),
            "auc_all": auc(pa, na), "npos_all": len(pa),
            "auc_clean": auc(pc, nc), "npos_clean": len(pc),
        })
    return rows


def offset_collapse(windows, mtp):
    P = np.concatenate([w["p"] for w in windows], axis=0)  # (N, mtp)
    c = np.corrcoef(P.T)
    return {
        "std_per_offset": P.std(axis=0).tolist(),
        "mean_per_offset": P.mean(axis=0).tolist(),
        "corr_off0_vs_offn": [float(c[0, n]) for n in range(mtp)],
        "mean_abs_diff_off0_offn": [float(np.abs(P[:, n] - P[:, 0]).mean()) for n in range(mtp)],
    }


def event_triggered_average(windows, mtp, thresh, lead_max=15):
    """Mean head score at lead time d BEFORE an income event, vs the global mean.

    Score = sum(pred[t,1:]) (the head's total 'income coming in 50-400ms' mass).
    d>0 = d frames BEFORE the event. This is the picture that decides the plan:
    if the curve only rises at d<=0 the head is a reporter, not a predictor.
    """
    all_s, lead = [], {d: [] for d in range(-3, lead_max + 1)}
    for w in windows:
        r, p = w["r"], w["p"]
        T = len(r)
        s = p[:, 1:].sum(axis=1)
        all_s.append(s)
        ev = np.where(r >= thresh)[0]
        for e in ev:
            for d in lead:
                t = e - d
                if 0 <= t < T:
                    lead[d].append(s[t])
    a = np.concatenate(all_s)
    mu, sd = a.mean(), a.std()
    rows = []
    for d in sorted(lead, reverse=True):
        v = np.array(lead[d])
        if len(v) == 0:
            continue
        # percentile of the mean lead-score within the global score distribution
        pct = float((a < v.mean()).mean())
        rows.append({"lead_frames": d, "lead_ms": int(1000 * d / FPS), "n": len(v),
                     "mean": float(v.mean()), "z": float((v.mean() - mu) / (sd + 1e-12)),
                     "pctile": pct})
    return {"global_mean": float(mu), "global_std": float(sd), "rows": rows}


def operating_point(windows, mtp, thresh, k, lookback, aa_only=False):
    """If we fired an AA whenever score > threshold, what do we catch / waste?"""
    lab, score, aa = [], [], []
    for w in windows:
        r, p = w["r"], w["p"]
        T = len(r)
        hi = T - k
        if hi <= lookback:
            continue
        t = np.arange(lookback, hi)
        csum = np.concatenate([[0.0], np.cumsum(r)])
        fut = csum[t + k + 1] - csum[t + 1]
        lab.append(fut >= thresh)
        score.append(p[t, 1:min(k, mtp - 1) + 1].sum(axis=1))
        aa.append(w["aa"][t])
    lab = np.concatenate(lab); score = np.concatenate(score); aa = np.concatenate(aa)
    out = []
    for fire_rate in [0.005, 0.0062, 0.01, 0.02, 0.05, 0.10, 0.20]:
        thr = np.quantile(score, 1 - fire_rate)
        fire = score >= thr
        tp = int((fire & lab).sum()); fp = int((fire & ~lab).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(int(lab.sum()), 1)
        out.append({"fire_rate": fire_rate, "thr": float(thr), "precision": prec,
                    "recall": rec, "tp": tp, "fp": fp,
                    "fp_per_sec": fp / (len(lab) / FPS)})
    return {"base_rate": float(lab.mean()), "n": int(len(lab)),
            "human_aa_rate": float(aa.mean()), "rows": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--latents-dir", required=True)
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=40, help="per match; <=0 = all")
    ap.add_argument("--thresh", type=float, default=0.010,
                    help="income event threshold in reward units (0.010 = 10 gold)")
    ap.add_argument("--lookback", type=int, default=6,
                    help="frames of no-income required for the CLEAN population")
    ap.add_argument("--tau-mode", default="clean", choices=["clean", "train"])
    ap.add_argument("--ablate-abilities", action="store_true",
                    help="zero all ability inputs (tests the action leak)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help=".npz of raw per-window r/p/aa for re-analysis")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    dyn, rh, mtp = load_phase2(args.phase2_ckpt, dev)
    windows = []
    for m in args.matches:
        w = collect(dyn, rh, mtp, m, args.latents_dir, args.labels_root,
                    args.seq_len, args.max_windows, dev, args.tau_mode,
                    args.ablate_abilities)
        print(f"  {m}: {len(w)} windows x {args.seq_len} frames", flush=True)
        windows += w
    nframes = sum(len(w["r"]) for w in windows)
    allr = np.concatenate([w["r"] for w in windows])
    print(f"\n=== {args.tag} | seq_len={args.seq_len} tau={args.tau_mode} "
          f"ablate_abilities={args.ablate_abilities} | frames={nframes} "
          f"({nframes/FPS:.0f}s) ===")
    print(f"reward: nonzero={float((allr>0).mean()):.3%} "
          f">={args.thresh}={float((allr>=args.thresh).mean()):.3%} "
          f"mean={allr.mean():.6f} p99={np.quantile(allr,0.99):.5f} max={allr.max():.4f}")

    res = {"tag": args.tag, "args": vars(args), "mtp": mtp, "frames": nframes}

    print("\n--- (E) MTP offset collapse check ---")
    oc = offset_collapse(windows, mtp)
    res["offset_collapse"] = oc
    print("  n      : " + " ".join(f"{n:>7d}" for n in range(mtp)))
    print("  std    : " + " ".join(f"{v:7.4f}" for v in oc["std_per_offset"]))
    print("  corr@0 : " + " ".join(f"{v:7.3f}" for v in oc["corr_off0_vs_offn"]))
    print("  |d|@0  : " + " ".join(f"{v:7.4f}" for v in oc["mean_abs_diff_off0_offn"]))

    print("\n--- (1) per-offset exact target: pred[t,n] vs reward[t+n]>=thresh ---")
    print("  n   ms    AUC_all  npos   AUC_clean  npos   (clean = no income in "
          f"[t-{args.lookback},t])")
    po = per_offset_exact(windows, mtp, args.thresh, args.lookback, rng)
    res["per_offset"] = po
    for r in po:
        print(f"  {r['n']}  {r['ms']:4d}   {r['auc_all']:.3f}  {r['npos_all']:5d}    "
              f"{r['auc_clean']:.3f}   {r['npos_clean']:5d}")

    print("\n--- (2) anticipation curve: income_within_k ---")
    horizons = [1, 2, 3, 5, 8, 10]
    at = anticipation_table(windows, mtp, args.thresh, args.lookback, horizons, rng)
    res["anticipation"] = at
    for pop in ["ALL", "CLEAN"]:
        print(f"\n  [{pop}]  k  ms   npos  nneg  | cur(n=0) mtp_cum mtp_max | "
              f"base_recent base_long | shuffle")
        for r in at:
            d = r[pop]
            if "auc_cur(n=0)" not in d:
                print(f"        {r['k']:2d} {r['ms']:4d}  (too few: pos={d['n_pos']} neg={d['n_neg']})")
                continue
            print(f"        {r['k']:2d} {r['ms']:4d} {d['n_pos']:6d} {d['n_neg']:6d} |"
                  f"  {d['auc_cur(n=0)']:.3f}   {d['auc_mtp_cum']:.3f}   {d['auc_mtp_max']:.3f}  |"
                  f"   {d['auc_base_recent']:.3f}     {d['auc_base_long']:.3f}   |  {d['auc_shuffle']:.3f}")

    print("\n--- (2b) event-triggered average: head score vs lead time before event ---")
    eta = event_triggered_average(windows, mtp, args.thresh)
    res["event_triggered"] = eta
    print(f"  global mean score = {eta['global_mean']:.5f}  std = {eta['global_std']:.5f}")
    print("  lead(frames)  lead(ms)   n   mean_score      z   pctile_of_global")
    for r in eta["rows"]:
        print(f"    {r['lead_frames']:+4d}      {r['lead_ms']:+6d}  {r['n']:4d}  "
              f"{r['mean']:9.5f}  {r['z']:+6.2f}   {r['pctile']:.3f}")

    print("\n--- (3) operating point (score = sum pred[t,1..k], k=5 -> 250ms) ---")
    op = operating_point(windows, mtp, args.thresh, 5, args.lookback)
    res["operating_point_k5"] = op
    print(f"  base rate (income within 250ms) = {op['base_rate']:.3%}   "
          f"human AA press rate = {op['human_aa_rate']:.3%}   n={op['n']}")
    print("  fire_rate  precision  recall    tp     fp   fp/sec")
    for r in op["rows"]:
        print(f"   {r['fire_rate']:7.3%}    {r['precision']:.3f}    {r['recall']:.3f} "
              f"{r['tp']:6d} {r['fp']:6d}  {r['fp_per_sec']:6.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"\nwrote {args.out}")
    if args.dump:
        np.savez_compressed(
            args.dump,
            r=np.stack([w["r"] for w in windows]),
            p=np.stack([w["p"] for w in windows]),
            aa=np.stack([w["aa"] for w in windows]),
            start=np.array([w["start"] for w in windows]),
        )
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
