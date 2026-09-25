"""Profile one ServerCollector step loop: where does a decision's wall time go?"""
import cProfile, pstats, sys, time, io
from pathlib import Path
import numpy as np, jax
from lanerl_jax.train.server_train import ServerCollector
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.trainer import _sample

E, T, STEPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
TEAMS = tuple(int(t) for t in sys.argv[5].split(",")) if len(sys.argv) > 5 else (0,)
c = ServerCollector(E, out, 49900 + 10*T + 100*len(TEAMS), 600., True, T, teams=TEAMS, server_dir=
    Path("/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0"))
print(f"setup {time.perf_counter()-t0:.1f}s", flush=True)
policy = LanePolicy(PolicyConfig())
obs, stats = c.observe()
params = policy.init(jax.random.key(0), obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
@jax.jit
def act(params, obs, key):
    lg = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
    a, lp, u = _sample(lg, key, ~obs.entity_pad_mask)
    return a, lp, u, lg.value
key = jax.random.key(1)
for _ in range(3):  # warm
    key, k = jax.random.split(key); a = act(params, obs, k)
    c.step(np.stack(jax.device_get(a[0]), -1)); obs, stats = c.observe()
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable()
ta = ts = to = 0.
for _ in range(STEPS):
    key, k = jax.random.split(key)
    t1 = time.perf_counter(); a = act(params, obs, k); h = np.stack(jax.device_get(a[0]), -1); t2 = time.perf_counter()
    c.step(h); t3 = time.perf_counter()
    obs, stats = c.observe(); t4 = time.perf_counter()
    ta += t2-t1; ts += t3-t2; to += t4-t3
pr.disable(); tot = time.perf_counter()-t0
print(f"envs={E} step_ticks={T} steps={STEPS}: {tot/STEPS*1000:.1f} ms/step  act={ta/STEPS*1000:.1f} step(server)={ts/STEPS*1000:.1f} observe={to/STEPS*1000:.1f}  -> {E*STEPS/tot:.0f} decisions/s", flush=True)
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumtime').print_stats(35); print(s.getvalue()[:6000])
c.close()
