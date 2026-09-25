import sys, time, torch, numpy as np
sys.path.insert(0, "scripts")
from agent_infer import GarenAgent
TOK = "rollout_stage/transformer_tokenizer_latest.pt"
dev = "cuda"
def bench(ctx, M=25):
    a = GarenAgent("x", tokenizer_ckpt=TOK, context=ctx, device=dev, init_only=True)
    frame = np.random.rand(352, 352, 3).astype(np.float32)
    a.reset()
    for _ in range(6):
        a.act_from_latent(a.encode_frame(frame), temperature=0.0)
    torch.cuda.synchronize()
    te = ta = 0.0
    for _ in range(M):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lat = a.encode_frame(frame)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        a.act_from_latent(lat, temperature=0.0)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        te += t1 - t0; ta += t2 - t1
    te = te/M*1000; ta = ta/M*1000; tot = te + ta
    print(f"  ctx={ctx:2d}: encode {te:5.1f} + dyn+policy {ta:6.1f} = {tot:6.1f} ms/frame -> {1000/tot:5.1f} fps  [{'FITS' if tot<=50 else 'OVER'} 50ms]", flush=True)
    del a; torch.cuda.empty_cache()
print(f"=== {torch.cuda.get_device_name(0)} (cap {torch.cuda.get_device_capability(0)}) ===", flush=True)
for ctx in (8, 16, 32):
    try: bench(ctx)
    except torch.cuda.OutOfMemoryError: print(f"  ctx={ctx}: OOM (fits in {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB? no)", flush=True); torch.cuda.empty_cache()
    except Exception as e: print(f"  ctx={ctx}: ERR {str(e)[:90]}", flush=True); torch.cuda.empty_cache()
