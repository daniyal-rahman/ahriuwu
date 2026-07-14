import sys, time, torch, numpy as np, contextlib
sys.path.insert(0, "scripts")
from agent_infer import GarenAgent
TOK = "rollout_stage/transformer_tokenizer_latest.pt"; dev = "cuda"
def bench(ctx, use_bf16, M=25):
    a = GarenAgent("x", tokenizer_ckpt=TOK, context=ctx, device=dev, init_only=True)
    frame = np.random.rand(352, 352, 3).astype(np.float32); a.reset()
    cm = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    for _ in range(6):
        with cm(): a.act_from_latent(a.encode_frame(frame), 0.0)
    torch.cuda.synchronize(); tot = 0.0
    for _ in range(M):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with cm(): a.act_from_latent(a.encode_frame(frame), 0.0)
        torch.cuda.synchronize(); tot += time.perf_counter() - t0
    tot = tot/M*1000
    print(f"  {'bf16' if use_bf16 else 'fp32'} ctx={ctx:2d}: {tot:6.1f} ms/frame -> {1000/tot:5.1f} fps  [{'FITS 20fps' if tot<=50 else 'over'}]", flush=True)
    del a; torch.cuda.empty_cache()
print(f"=== {torch.cuda.get_device_name(0)} ===", flush=True)
for bf in (False, True):
    for ctx in (8, 16, 32):
        try: bench(ctx, bf)
        except Exception as e: print(f"  {'bf16' if bf else 'fp32'} ctx={ctx}: ERR {str(e)[:60]}", flush=True); torch.cuda.empty_cache()
