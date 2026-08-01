#!/usr/bin/env python3
"""Generate MANIFEST.json + README.md for a dataset (or every dataset under /mnt/nfs/datasets).

Goal: any BASIC question about a dataset (how many games, how many frames, what fps, what
resolution, what format, HUD on/off, what labels, where it came from) is auto-answered by a
committed manifest — never hand-parsed from the raw files (that is how the "4 fps vs 20 fps"
misreport happened; the truth was in labels.json all along).

Usage:
    python scripts/gen_dataset_manifest.py /mnt/nfs/datasets/yt_pretrain_garen
    python scripts/gen_dataset_manifest.py --all
    python scripts/gen_dataset_manifest.py --all --fast     # skip per-item frame counting

Facts that CANNOT be derived from files (purpose, source, HUD status, gotchas) come from the
PROV registry below, overridable/extendable by a hand-written <dataset>/.provenance.json.
When you build a NEW dataset: have the pipeline drop a labels.json (like the YT pipeline does)
and/or a .provenance.json, then run this. Re-run any time to refresh.
"""
import os, sys, json, glob, struct, datetime, subprocess, statistics
from collections import Counter

DATASETS_ROOT = "/mnt/nfs/datasets"
IMG_EXT = (".jpg", ".jpeg", ".png")

# --- provenance for known datasets (facts not derivable from the files themselves) ---
PROV = {
  "yt_pretrain_garen": {
    "purpose": "YouTube pretraining corpus for the v7 tokenizer — Garen gameplay VODs.",
    "provenance": "yt-dlp pipeline (scripts/download_yt_frames.py) run on the GCP spot fleet + laptopserver; "
                  "extract at 20 fps, HUD region blacked, downscale 1280x720 -> 352x352 (INTER_AREA squish). FROZEN corpus.",
    "hud": "BLACKED OUT in the on-screen HUD region (~35% of the frame zeroed). Training loss MUST exclude these "
           "regions (valid-mask fix) or the tokenizer learns to paint them black.",
    "related": ["r2:ahriuwu-yt-pretrain (mirror)", "ahriuwu_yt_holdout (6-game holdout subset of this)"],
    "gotchas": ["Frames inside each .tar are NOT lexically ordered — sort on read.",
                "HUD blacked -> exclude from loss."],
  },
  "lol_replays_16_9_772": {
    "purpose": "Action-labeled LoL replay corpus (in-client replays, HUD DISABLED) — frames + per-frame "
               "camera/memory/click labels for the dynamics + agent work.",
    "provenance": "Rendered from .rofl replays with the in-client HUD toggled OFF; 16:9 -> 352x352.",
    "hud": "DISABLED (rendered off) -> full game view, NO blacked region. The clean action-labeled set.",
    "fps": 20, "fps_authority": "pipeline (eval_dynamics.py FPS=20)",
    "related": ["dynamics_replay_latents_v7_dim32 (pretok of this through the step-6000 tokenizer, on R2)"],
    "gotchas": ["Per-game dir holds frames/ + labels.json (per-frame, large) + raw_cam.json + raw_mem.json + clicks.json."],
  },
  "lol_replays_16_9_772_smoketest": {
    "purpose": "3-game smoke-test subset of lol_replays_16_9_772.",
    "hud": "DISABLED", "fps": 20, "related": ["lol_replays_16_9_772 (parent)"],
  },
  "ahriuwu_yt_holdout": {
    "purpose": "6-game held-out YT subset for rollout eval (before/after PSNR on fixed clips 1 & 16).",
    "provenance": "Flat-JPG extraction of 6 yt_pretrain_garen video IDs, kept OUT of training.",
    "hud": "same as yt_pretrain_garen (BLACKED)", "fps": 20, "fps_authority": "parent corpus labels.json",
    "related": ["yt_pretrain_garen (parent corpus)"],
  },
  "_ckpt": {"purpose": "MISFILED: a tokenizer checkpoint (belongs under /mnt/nfs/checkpoints), not a dataset.",
            "needs_attention": "move tokenizer_yt1500.pt to /mnt/nfs/checkpoints/"},
  "bci-falcon": {"purpose": "BCI Falcon dataset (separate brain-computer-interface project).", "needs_annotation": True},
  "pd-kaggle":  {"purpose": "Parkinson's AMP-PD Kaggle competition data (separate project).", "needs_annotation": True},
  "pd-ppmi":    {"purpose": "Parkinson's PPMI data (separate project).", "needs_annotation": True},
}

def sh(cmd):
    try: return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""

def du_bytes(path):
    out = sh(f"du -sb '{path}'")
    try: return int(out.split()[0])
    except Exception: return None

def human(n):
    if n is None: return "?"
    for u in ("B","KiB","MiB","GiB","TiB"):
        if n < 1024: return f"{n:.0f} {u}" if u=="B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PiB"

def img_size(path):
    """Read (w,h) from a PNG/JPEG header without PIL."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24]); return w, h
            if head[:2] == b"\xff\xd8":  # JPEG: walk markers
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff": b = f.read(1)
                    marker = f.read(1)
                    if not marker: break
                    if 0xC0 <= marker[0] <= 0xCF and marker[0] not in (0xC4,0xC8,0xCC):
                        f.read(3); h, w = struct.unpack(">HH", f.read(4)); return w, h
                    seg = f.read(2)
                    if len(seg) < 2: break
                    f.seek(struct.unpack(">H", seg)[0] - 2, 1)
    except Exception: return None
    return None

def stat_block(counts):
    counts = [c for c in counts if c is not None]
    if not counts: return {}
    return {"total": sum(counts), "per_item_min": min(counts),
            "per_item_mean": round(statistics.mean(counts)), "per_item_max": max(counts),
            "n_items": len(counts)}

def detect_layout(path):
    if glob.glob(f"{path}/*.labels.json") and glob.glob(f"{path}/*.tar"): return "tar+labels"
    subdirs = [d for d in glob.glob(f"{path}/*") if os.path.isdir(d)]
    if subdirs:
        s0 = subdirs[0]
        if os.path.isdir(f"{s0}/frames"): return "frame-subdirs"
        if any(x.lower().endswith(IMG_EXT) for x in os.listdir(s0)[:50]): return "flat-frame-subdirs"
        return "subdirs-generic"
    if glob.glob(f"{path}/*.pt"): return "checkpoint-blob"
    return "generic"

def build(path, fast=False):
    name = os.path.basename(path.rstrip("/"))
    prov = dict(PROV.get(name, {}))
    p_over = os.path.join(path, ".provenance.json")
    if os.path.exists(p_over):
        try: prov.update(json.load(open(p_over)))
        except Exception: pass
    layout = detect_layout(path)
    m = {"name": name, "path": path, "layout": layout,
         "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
         "generated_by": "scripts/gen_dataset_manifest.py",
         "total_size_bytes": du_bytes(path)}
    m.update({k: prov[k] for k in prov})

    if layout == "tar+labels":
        labels = sorted(glob.glob(f"{path}/*.labels.json"))
        frames, fpss, targets, dist = [], Counter(), Counter(), {"rank": Counter(), "patch": Counter(), "champion": Counter()}
        field_keys = set()
        for lf in labels:
            try: d = json.load(open(lf))
            except Exception: continue
            field_keys |= set(d.keys())
            if "n_frames" in d: frames.append(d["n_frames"])
            if "fps" in d: fpss[d["fps"]] += 1
            if "target" in d: targets[d["target"]] += 1
            for k in ("rank", "patch", "champion"):
                if d.get(k) is not None: dist[k][d[k]] += 1
        m["item_kind"] = "games"; m["item_count"] = len(glob.glob(f"{path}/*.tar"))
        m["frames"] = stat_block(frames)
        m["media"] = {"format": "jpg", "fps": (fpss.most_common(1)[0][0] if fpss else prov.get("fps")),
                      "resolution": (f"{targets.most_common(1)[0][0]}x{targets.most_common(1)[0][0]}" if targets else None)}
        if len(fpss) > 1: m["media"]["fps_WARNING"] = f"mixed fps across games: {dict(fpss)}"
        m["label_fields"] = sorted(field_keys)
        m["distributions"] = {k: dict(v.most_common(8)) for k, v in dist.items() if v}

    elif layout in ("frame-subdirs", "flat-frame-subdirs"):
        subdirs = sorted(d for d in glob.glob(f"{path}/*") if os.path.isdir(d))
        counts, sample_res, aux = [], None, set()
        for sd in subdirs:
            fdir = f"{sd}/frames" if layout == "frame-subdirs" and os.path.isdir(f"{sd}/frames") else sd
            if not fast:
                try: counts.append(sum(1 for x in os.listdir(fdir) if x.lower().endswith(IMG_EXT)))
                except Exception: pass
            if sample_res is None:
                try:
                    imgs = [x for x in os.listdir(fdir) if x.lower().endswith(IMG_EXT)]
                    if imgs: sample_res = img_size(os.path.join(fdir, sorted(imgs)[0]))
                except Exception: pass
            if layout == "frame-subdirs":
                for x in os.listdir(sd):
                    if x.endswith(".json"): aux.add(x)
        m["item_kind"] = "games"; m["item_count"] = len(subdirs)
        if counts: m["frames"] = stat_block(counts)
        fmt = "png" if layout == "frame-subdirs" else "jpg"
        m["media"] = {"format": fmt, "fps": prov.get("fps"),
                      "resolution": (f"{sample_res[0]}x{sample_res[1]}" if sample_res else None)}
        if aux: m["per_game_files"] = ["frames/"] + sorted(aux)

    elif layout == "checkpoint-blob":
        pts = glob.glob(f"{path}/*.pt")
        m["item_kind"] = "files"; m["item_count"] = len(pts)
        m["contents"] = [os.path.basename(p) for p in pts][:20]

    else:  # generic / subdirs-generic
        _ours = {"MANIFEST.json", "README.md", ".provenance.json"}
        entries = [e for e in (os.listdir(path) if os.path.isdir(path) else []) if e not in _ours]
        m["item_kind"] = "entries"; m["item_count"] = len(entries)
        ext = Counter(os.path.splitext(x)[1].lower() for x in entries if not os.path.isdir(os.path.join(path, x)))
        m["top_level_sample"] = sorted(entries)[:12]
        if ext: m["file_types"] = dict(ext.most_common(8))

    m["total_size"] = human(m["total_size_bytes"])
    return m

def render_readme(m):
    L = [f"# {m['name']} — dataset manifest", "",
         f"> Auto-generated by `{m['generated_by']}` on {m['generated']}. Re-run to refresh; do not hand-edit.",
         f"> Machine-readable copy: `MANIFEST.json`.", ""]
    if m.get("purpose"): L += [f"**Purpose:** {m['purpose']}", ""]
    L += [f"- **Path:** `{m['path']}`", f"- **Layout:** {m['layout']}",
          f"- **{m.get('item_kind','items').capitalize()}:** {m.get('item_count','?')}",
          f"- **Total size:** {m.get('total_size','?')}"]
    fr = m.get("frames")
    if fr: L.append(f"- **Frames:** {fr['total']:,} total  (per game: min {fr['per_item_min']:,} · "
                    f"mean {fr['per_item_mean']:,} · max {fr['per_item_max']:,})")
    md = m.get("media", {})
    if md:
        bits = []
        if md.get("fps") is not None: bits.append(f"fps **{md['fps']}**")
        if md.get("resolution"): bits.append(f"res **{md['resolution']}**")
        if md.get("format"): bits.append(f"format **{md['format']}**")
        if md.get("fps_WARNING"): bits.append(f"⚠️ {md['fps_WARNING']}")
        if bits: L.append("- **Media:** " + " · ".join(bits))
    if m.get("hud"): L.append(f"- **HUD:** {m['hud']}")
    if m.get("per_game_files"): L.append(f"- **Per-game files:** " + ", ".join(f"`{x}`" for x in m["per_game_files"]))
    if m.get("label_fields"): L.append(f"- **Label fields:** " + ", ".join(m["label_fields"]))
    if m.get("contents"): L.append(f"- **Contents:** " + ", ".join(f"`{x}`" for x in m["contents"]))
    if m.get("top_level_sample"): L.append(f"- **Top-level:** " + ", ".join(m["top_level_sample"]))
    if m.get("file_types"): L.append(f"- **File types:** " + ", ".join(f"{k or '(none)'}×{v}" for k,v in m["file_types"].items()))
    dist = m.get("distributions", {})
    for k, v in dist.items():
        if v: L.append(f"- **{k}:** " + ", ".join(f"{kk}×{vv}" for kk, vv in v.items()))
    if m.get("provenance"): L += ["", f"**Provenance:** {m['provenance']}"]
    if m.get("gotchas"): L += ["", "**Gotchas:**"] + [f"- {g}" for g in m["gotchas"]]
    if m.get("related"): L += ["", "**Related:** " + ", ".join(m["related"])]
    if m.get("needs_annotation"): L += ["", "> ⚠️ **Needs provenance annotation** — drop a `.provenance.json` here and re-run."]
    if m.get("needs_attention"): L += ["", f"> ⚠️ **{m['needs_attention']}**"]
    return "\n".join(L) + "\n"

def process(path, fast=False):
    m = build(path, fast=fast)
    json.dump(m, open(os.path.join(path, "MANIFEST.json"), "w"), indent=2, default=str)
    open(os.path.join(path, "README.md"), "w").write(render_readme(m))
    fr = m.get("frames", {})
    print(f"  ✓ {m['name']:<32} {m.get('item_count','?')} {m.get('item_kind','')}, "
          f"{m.get('total_size','?')}, frames={fr.get('total','-')}, "
          f"fps={m.get('media',{}).get('fps','-')}, res={m.get('media',{}).get('resolution','-')}")
    return m

REPO_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

def write_index(mans):
    """Top-level docs/DATASETS.md — one table over all datasets. Regenerated on every --all run."""
    ah = {"yt_pretrain_garen","lol_replays_16_9_772","lol_replays_16_9_772_smoketest","ahriuwu_yt_holdout","_ckpt"}
    L = ["# Datasets index", "",
         f"> Auto-generated by `scripts/gen_dataset_manifest.py --all` on "
         f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}. Each dataset also has its own "
         f"`MANIFEST.json` + `README.md`. Re-run to refresh; do not hand-edit.", "",
         "| dataset | items | size | frames | fps | res | HUD | purpose |",
         "|---|---|---|---|---|---|---|---|"]
    def row(m):
        fr = m.get("frames", {}); md = m.get("media", {})
        hud = (m.get("hud","") or "")[:24]
        frtot = fr.get("total") if fr else None
        frcell = f"{frtot:,}" if isinstance(frtot, int) else "-"
        return (f"| `{m['name']}` | {m.get('item_count','?')} {m.get('item_kind','')} | {m.get('total_size','?')} "
                f"| {frcell} | {md.get('fps') or '-'} | {md.get('resolution') or '-'} "
                f"| {hud or '-'} | {(m.get('purpose','') or '')[:70]} |")
    for m in sorted(mans, key=lambda m: (m["name"] not in ah, m["name"])):
        L.append(row(m))
    L += ["", "**ahriuwu training data:** `yt_pretrain_garen` (tokenizer pretrain), "
          "`lol_replays_16_9_772` (action-labeled, HUD off), `ahriuwu_yt_holdout` (rollout eval).",
          "Other entries (`bci-*`, `pd-*`) are separate projects.",
          "", "To (re)generate: `python scripts/gen_dataset_manifest.py --all`. "
          "New dataset → have the build pipeline drop a `labels.json` per item and/or a `.provenance.json`, then run this."]
    os.makedirs(REPO_DOCS, exist_ok=True)
    open(os.path.join(REPO_DOCS, "DATASETS.md"), "w").write("\n".join(L) + "\n")
    print(f"  ✓ index -> docs/DATASETS.md ({len(mans)} datasets)")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    fast = "--fast" in sys.argv
    if "--all" in sys.argv:
        targets = sorted(d for d in glob.glob(f"{DATASETS_ROOT}/*")
                         if os.path.isdir(d) and os.path.basename(d) != "lost+found")
    else:
        targets = args
    if not targets:
        print(__doc__); sys.exit(1)
    print(f"generating manifests for {len(targets)} dataset(s):")
    mans = []
    for t in targets:
        try: mans.append(process(t, fast=fast))
        except Exception as e: print(f"  ✗ {t}: {e}")
    if "--all" in sys.argv and mans:
        write_index(mans)
