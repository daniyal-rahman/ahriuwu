"""DAgger step for the BC diagnostic: relabel recorded (obs) with the scripted
last-hitter (a pure function of the observation, so labelling is offline) and
merge several npz files into one aggregated dataset.

    python -m lanerl_jax.train.bc_dagger out.npz demos.npz clone_rollout.npz [...]
"""
import sys
import numpy as np, jax, jax.numpy as jnp
from .scripted_policy import scripted_act
from ..obs.builder import Observation


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    label = jax.jit(jax.vmap(jax.vmap(lambda e, m, s, g: scripted_act(
        Observation(e, m, s, g, jnp.full((e.shape[0],), -1, jnp.int32)), None))))
    parts = {k: [] for k in ("entities", "mask", "self", "global", "action", "done")}
    for p in paths:
        d = np.load(p)
        T, N = d["action"].shape[:2]
        acts = []
        for t0 in range(0, T, 256):     # chunk to bound memory
            a = label(jnp.asarray(d["entities"][t0:t0+256], jnp.float32), jnp.asarray(d["mask"][t0:t0+256]),
                      jnp.asarray(d["self"][t0:t0+256]), jnp.asarray(d["global"][t0:t0+256]))
            acts.append(np.stack([np.asarray(x) for x in a], -1).astype(np.int16))
        relabel = np.concatenate(acts)
        agree = (relabel == d["action"]).all(-1).mean()
        print(f"{p}: {T}x{N} steps; original actions agree with the scripted label on {agree:.3f}", flush=True)
        for k in parts:
            parts[k].append(relabel if k == "action" else d[k])
    # datasets may have different N: keep time-major by concatenating along T after padding N? Simpler: treat each
    # file's agents as extra columns only if T matches; otherwise stack along T with N aligned by truncation.
    n_min = min(x.shape[1] for x in parts["action"])
    merged = {k: np.concatenate([x[:, :n_min] for x in v], axis=0) for k, v in parts.items()}
    np.savez_compressed(out, **merged)
    print("wrote", out, {k: v.shape for k, v in merged.items()})


if __name__ == "__main__":
    main()
