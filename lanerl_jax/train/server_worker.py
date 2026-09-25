"""Worker-process entry for `server_train.MultiProcessCollector`.

Imported by the spawned child BEFORE anything else, so the CPU-only JAX
setting below is in place before `server_train` (and therefore `jax`) is
imported. Workers encode observations on the CPU; only the learner process
uses the GPU.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def worker_main(conn, kwargs):
    import numpy as np
    import jax
    from .server_train import ServerCollector
    collector = None
    try:
        collector = ServerCollector(**kwargs)
        conn.send(("ready", collector.n))
        while True:
            cmd, arg = conn.recv()
            if cmd == "observe":
                obs, stats = collector.observe()
                conn.send(("ok", (tuple(np.asarray(x) for x in obs), stats)))
            elif cmd == "step":
                conn.send(("ok", collector.step(arg)))
            elif cmd == "restart":
                collector.restart_done(arg)
                conn.send(("ok", list(collector.episodes)))
            elif cmd == "ranks":
                conn.send(("ok", collector.spell_ranks()))
            elif cmd == "close":
                conn.send(("ok", None))
                break
    except BaseException as exc:  # report, then die; the parent raises
        import traceback
        conn.send(("error", traceback.format_exc()))
    finally:
        if collector is not None:
            collector.close()
